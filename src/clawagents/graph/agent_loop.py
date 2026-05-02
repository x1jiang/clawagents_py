"""ClawAgents ReAct Agent Loop

Single-loop ReAct executor inspired by deepagents/openclaw architecture.
Eliminates the separate Understand/Verify phases that added 2 unnecessary
LLM round-trips per iteration.

Flow: LLM → tool calls → LLM → tool calls → ... → final text answer

Robustness features retained:
  - Tool loop detection
  - Context-window guard with auto-compaction
  - Parallel tool execution
  - Tool-output truncation
  - Structured event callbacks (on_event)

Efficiency features (learned from deepagents/openclaw):
  - Adaptive token estimation multiplier (auto-calibrates after overflow)
  - Tool argument truncation in older messages (saves tokens)
  - Single-pass message filtering
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from clawagents.providers.llm import LLMProvider, LLMMessage, LLMResponse, NativeToolSchema, NativeToolCall, strip_thinking_tokens
from clawagents.tools.registry import ToolRegistry, ParsedToolCall, ToolResult
from clawagents.run_context import RunContext
from clawagents.session.heartbeat import (
    DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S,
    run_with_heartbeat,
)
from clawagents.usage import Usage, RequestUsage
from clawagents.lifecycle import RunHooks, AgentHooks
from clawagents.guardrails import (
    InputGuardrail,
    OutputGuardrail,
    GuardrailBehavior,
    GuardrailTripwireTriggered,
    GuardrailResult,
)
from clawagents.stream_events import (
    StreamEvent,
    stream_event_from_kind,
)
from clawagents.context.carryover import get_compaction_carryover
from clawagents.handoffs import Handoff, HandoffInputData
from clawagents.prompts import build_system_prompt
from clawagents.tracing import handoff_span

logger = logging.getLogger(__name__)


# ─── Model Control Token Sanitization ─────────────────────────────────────
_MODEL_CONTROL_TOKEN_RE = re.compile(r'<[｜|][^>]*?[｜|]>')

def _sanitize_assistant_text(text: str) -> str:
    """Strip leaked model control tokens from assistant text (GLM-5, DeepSeek, etc.)."""
    return _MODEL_CONTROL_TOKEN_RE.sub('', text).strip()


# ─── Dangling Tool Call Repair (learned from deepagents) ──────────────────
# When native function calling is used and the agent loop is interrupted mid-execution,
# the next LLM call sees tool_calls without matching tool results — most APIs reject this.
# This pass inserts synthetic "cancelled" responses for any dangling tool calls.

def _patch_dangling_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    if not messages:
        return messages

    # Build set of all tool_call_ids that have a matching role="tool" response
    responded_ids: set[str] = set()
    for msg in messages:
        if msg.role == "tool" and msg.tool_call_id:
            responded_ids.add(msg.tool_call_id)

    patched: list[LLMMessage] = []
    for i, msg in enumerate(messages):
        patched.append(msg)

        # Text-mode: look for assistant messages with JSON tool calls without a following [Tool Result]
        if msg.role == "assistant" and isinstance(msg.content, str) and msg.content.startswith('{"tool":'):
            _next_msg = messages[i + 1] if i + 1 < len(messages) else None
            _next_content = _next_msg.content if _next_msg is not None else None
            has_result = (
                _next_msg is not None
                and _next_msg.role == "user"
                and isinstance(_next_content, str)
                and _next_content.startswith("[Tool Result]")
            )
            if not has_result:
                patched.append(LLMMessage(
                    role="user",
                    content="[Tool Result] Tool call was cancelled — the agent was interrupted before it could complete.",
                ))

        # Native tool calls: inject synthetic role="tool" for any missing responses
        elif msg.role == "assistant" and msg.tool_calls_meta:
            for tc in msg.tool_calls_meta:
                tc_id = tc.get("id")
                if tc_id and tc_id not in responded_ids:
                    patched.append(LLMMessage(
                        role="tool",
                        content="Tool call was cancelled — the agent was interrupted before it could complete.",
                        tool_call_id=tc_id,
                    ))
                    responded_ids.add(tc_id)

    return patched


# ─── Tool Result Eviction (learned from deepagents) ───────────────────────
# When tool output exceeds a threshold, write the full result to a file and
# replace it with a head/tail preview + file path.

_EVICTION_CHARS_THRESHOLD = 80_000  # ~20K tokens
def _get_eviction_dir() -> Path:
    return Path.cwd() / ".clawagents" / "large_results"


_PREVIEW_MAX_CHARS = 2000

def _create_content_preview(content: str, head_lines: int = 5, tail_lines: int = 5) -> str:
    lines = content.split("\n")
    if len(lines) <= head_lines + tail_lines + 2 and len(content) <= _PREVIEW_MAX_CHARS:
        return content

    if len(lines) <= head_lines + tail_lines + 2:
        half = _PREVIEW_MAX_CHARS // 2
        return (content[:half]
                + f"\n... [{len(content) - _PREVIEW_MAX_CHARS} chars truncated] ...\n"
                + content[-half:])

    head = "\n".join(f"{i + 1}: {l}" for i, l in enumerate(lines[:head_lines]))
    total = len(lines)
    tail = "\n".join(
        f"{total - tail_lines + i + 1}: {l}"
        for i, l in enumerate(lines[-tail_lines:])
    )
    omitted = total - head_lines - tail_lines
    return f"{head}\n... [{omitted} lines truncated] ...\n{tail}"


def _evict_large_tool_result(tool_name: str, output: str) -> str:
    if len(output) < _EVICTION_CHARS_THRESHOLD:
        return output

    try:
        _get_eviction_dir().mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)
        file_path = _get_eviction_dir() / f"{sanitized}_{ts}.txt"
        file_path.write_text(output, "utf-8")

        preview = _create_content_preview(output)
        return (
            f"[Result too large ({len(output)} chars) — saved to {file_path}]\n"
            f"Use read_file to access the full result. Preview:\n\n{preview}"
        )
    except Exception:
        half = _EVICTION_CHARS_THRESHOLD // 2
        return (
            output[:half]
            + f"\n\n... [truncated {len(output) - _EVICTION_CHARS_THRESHOLD} chars] ...\n\n"
            + output[-half:]
        )


def _tool_observation(result: ToolResult) -> str | list[dict[str, Any]]:
    if result.success:
        return result.output
    error = f"Error: {result.error}" if result.error else "Error: Tool failed"
    if isinstance(result.output, list):
        return [{"type": "text", "text": error}, *result.output]
    output = str(result.output or "").strip()
    return f"{error}\nOutput:\n{output}" if output else error


# ─── Model-Aware Context Budget (learned from deepagents) ─────────────────

# NOTE: Order matters for prefix matching below. List the *most specific*
# keys first so e.g. "gpt-5.4-medium" resolves to the "gpt-5.4" profile
# rather than falling back to "gpt-5".
_MODEL_PROFILES: dict[str, dict[str, int | float]] = {
    # ── OpenAI — GPT-5 family (400K context) ───────────────────────────
    "gpt-5.4-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.4-nano": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.4": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.3-codex": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.3-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.3": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.2-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.2": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.1-codex": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.1-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5.1": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5-codex": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5-mini": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5-nano": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    "gpt-5": {"max_input_tokens": 400_000, "budget_ratio": 0.85},
    # ── OpenAI — GPT-4.1 (1M context) ──────────────────────────────────
    "gpt-4.1-mini": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "gpt-4.1-nano": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "gpt-4.1": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    # ── OpenAI — GPT-4o (128K context) ─────────────────────────────────
    "gpt-4o-mini": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gpt-4o": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    # ── OpenAI — reasoning (o-series) ──────────────────────────────────
    "o4-mini": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o3-mini": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o3": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o1-pro": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    "o1-mini": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "o1": {"max_input_tokens": 200_000, "budget_ratio": 0.80},
    # ── Google — Gemini 3.x (1M–2M context) ────────────────────────────
    "gemini-3.1-pro": {"max_input_tokens": 2_000_000, "budget_ratio": 0.90},
    "gemini-3.1-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3.1": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-pro": {"max_input_tokens": 2_000_000, "budget_ratio": 0.90},
    "gemini-3-flash-preview": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    # ── Google — Gemini 2.5 ────────────────────────────────────────────
    "gemini-2.5-pro": {"max_input_tokens": 2_000_000, "budget_ratio": 0.90},
    "gemini-2.5-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    # ── Anthropic — Claude 4.x ─────────────────────────────────────────
    "claude-opus-4-7": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-opus-4-5": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-opus-4": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-4.6-sonnet": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-4.5-sonnet": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-sonnet-4-5": {"max_input_tokens": 1_000_000, "budget_ratio": 0.85},
    "claude-sonnet-4": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    # ── Anthropic — Claude 3.x ─────────────────────────────────────────
    "claude-3-7-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-3-5-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-3-5-haiku": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    # ── Ollama / local OpenAI-compatible models ────────────────────────
    # NOTE: prefix-matching walks in insertion order. Put specific tags
    # (``gemma4:e4b``) before generic families (``gemma4``) before legacy
    # prefixes (``gemma3``/``gemma``) so "gemma4:e4b" doesn't collapse to
    # the 8K Gemma-1 default.
    # ── Google — Gemma 4 (released 2026-04-02; Apache-2.0) ─────────────
    "gemma4:e2b": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gemma4:e4b": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gemma4:26b": {"max_input_tokens": 256_000, "budget_ratio": 0.85},
    "gemma4:31b": {"max_input_tokens": 256_000, "budget_ratio": 0.85},
    "gemma4": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    # ── Google — Gemma 3n (edge/mobile 32K) ────────────────────────────
    "gemma3n:e4b": {"max_input_tokens": 32_000, "budget_ratio": 0.80},
    "gemma3n:e2b": {"max_input_tokens": 32_000, "budget_ratio": 0.80},
    "gemma3n": {"max_input_tokens": 32_000, "budget_ratio": 0.80},
    # ── Google — Gemma 3 / 2 / 1 ───────────────────────────────────────
    "gemma3": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gemma2": {"max_input_tokens": 8_192, "budget_ratio": 0.75},
    "gemma": {"max_input_tokens": 8_192, "budget_ratio": 0.75},
    "llama3.3": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "llama3.2": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "llama3.1": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "qwen2.5-coder": {"max_input_tokens": 32_768, "budget_ratio": 0.80},
    "qwen2.5": {"max_input_tokens": 32_768, "budget_ratio": 0.80},
    "deepseek-r1": {"max_input_tokens": 64_000, "budget_ratio": 0.75},
    "mistral": {"max_input_tokens": 32_768, "budget_ratio": 0.80},
    "phi4": {"max_input_tokens": 16_384, "budget_ratio": 0.75},
}


def _resolve_context_budget(model_name: str, context_window: int) -> tuple[int, float]:
    """Return (effective_window, budget_ratio) based on model profile."""
    profile = _MODEL_PROFILES.get(model_name)
    if not profile:
        # Try prefix match
        for k, v in _MODEL_PROFILES.items():
            if model_name.startswith(k):
                profile = v
                break
    if profile:
        return int(profile["max_input_tokens"]), float(profile["budget_ratio"])
    return context_window, 0.75

AgentStatus = Literal["running", "done", "error", "max_iterations"]

EventKind = Literal[
    "tool_call",
    "tool_result",
    "retry",
    "agent_done",
    "warn",
    "error",
    "context",
    "final_content",
    "approval_required",
    "tool_skipped",
    "turn_started",
    "assistant_message",
    "assistant_delta",
    "tool_started",
    "usage",
    "guardrail_tripped",
    "compact_progress",
    "final_output",
]

OnEvent = Callable[[EventKind, dict[str, Any]], None]

# Hook types for extensibility without middleware overhead
BeforeLLMHook = Callable[[list["LLMMessage"]], list["LLMMessage"]]
AfterToolHook = Callable[[str, dict[str, Any], "ToolResult"], "ToolResult"]


@dataclass
class HookResult:
    """Rich result from a BeforeToolHook.

    Allows hooks to deny execution with a reason, rewrite tool arguments,
    or inject messages into the conversation — instead of a bare bool.
    """
    allowed: bool = True
    reason: str = ""
    updated_args: dict[str, Any] | None = None
    messages: list[Any] | None = None  # list[LLMMessage] — forward-ref safe


# BeforeToolHook is backward-compatible: old hooks returning bool still work.
BeforeToolHook = Callable[[str, dict[str, Any]], "bool | HookResult"]


def _default_on_event(kind: EventKind, data: dict[str, Any]) -> None:
    """Default event handler: write to stderr (CLI mode)."""
    if kind == "tool_call":
        sys.stderr.write(f"\U0001f527 {data['name']}\n")
    elif kind == "retry":
        sys.stderr.write(f"[retry] {data['reason']}\n")
    elif kind == "agent_done":
        sys.stderr.write(
            f"\n\u2713 {data['tool_calls']} tool calls"
            f" \u00b7 {data['iterations']} iterations"
            f" \u00b7 {data['elapsed']:.1f}s\n"
        )
    elif kind == "final_content":
        sys.stdout.write(data["content"])
        sys.stdout.write("\n")
        sys.stdout.flush()
    elif kind == "warn":
        sys.stderr.write(f"[warn] {data['message']}\n")
    elif kind == "error":
        sys.stderr.write(f"[error] {data['phase']}: {data['message']}\n")
    elif kind == "context":
        sys.stderr.write(f"[context] {data['message']}\n")
    elif kind == "compact_progress":
        phase = data.get("phase", "")
        message = data.get("message", "")
        sys.stderr.write(f"[compact] {phase}: {message}\n")
    sys.stderr.flush()


# ── Guardrail + Session helpers ──────────────────────────────────────────

async def _run_input_guardrails(
    guardrails: list[InputGuardrail],
    ctx: RunContext,
    task: str,
) -> Optional[str]:
    """Run input guardrails. Raises GuardrailTripwireTriggered on RAISE_EXCEPTION.

    Returns a rewrite string if any guardrail rewrites the input, else ``None``.
    """
    rewrite_prefix: list[str] = []
    for gr in guardrails:
        result: GuardrailResult = await gr.run(ctx, task)
        if result.behavior == GuardrailBehavior.ALLOW:
            continue
        if result.behavior == GuardrailBehavior.RAISE_EXCEPTION:
            raise GuardrailTripwireTriggered(gr.name, "input", result)
        if result.behavior == GuardrailBehavior.REJECT_CONTENT:
            rewrite_prefix.append(
                f"[Input Guardrail '{gr.name}']: "
                f"{result.replacement_output or result.message or 'rejected'}"
            )
    return "\n".join(rewrite_prefix) if rewrite_prefix else None


async def _run_output_guardrails(
    guardrails: list[OutputGuardrail],
    ctx: RunContext,
    output: str,
) -> tuple[str, Optional[str]]:
    """Run output guardrails. Raises on RAISE_EXCEPTION.

    Returns ``(possibly-rewritten output, tripped name or None)``.
    """
    for gr in guardrails:
        result: GuardrailResult = await gr.run(ctx, output)
        if result.behavior == GuardrailBehavior.ALLOW:
            continue
        if result.behavior == GuardrailBehavior.RAISE_EXCEPTION:
            raise GuardrailTripwireTriggered(gr.name, "output", result)
        if result.behavior == GuardrailBehavior.REJECT_CONTENT:
            return (
                result.replacement_output or result.message or f"[blocked by {gr.name}]",
                gr.name,
            )
    return (output, None)


async def _session_get_items(session: Any, limit: int | None = None) -> list[LLMMessage]:
    """Fetch prior messages from a Session-protocol backend (async or sync)."""
    get_items = getattr(session, "get_items", None)
    if get_items is None:
        return []
    accepts_limit = False
    if limit is not None:
        try:
            sig = inspect.signature(get_items)
            accepts_limit = "limit" in sig.parameters or any(
                p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_limit = True
    res = get_items(limit=limit) if accepts_limit else get_items()
    if asyncio.iscoroutine(res):
        res = await res
    out: list[LLMMessage] = []
    for item in res or []:
        if isinstance(item, LLMMessage):
            out.append(item)
        elif isinstance(item, dict) and "role" in item:
            out.append(LLMMessage(
                role=item.get("role", "user"),
                content=item.get("content", ""),
                tool_call_id=item.get("tool_call_id"),
                tool_calls_meta=item.get("tool_calls_meta"),
                thinking=item.get("thinking"),
            ))
    return out


async def _session_add_items(session: Any, items: list[LLMMessage]) -> None:
    """Persist messages to a Session-protocol backend (async or sync).

    Passes ``LLMMessage`` instances directly so both built-in backends
    (``InMemorySession``, ``SQLiteSession``) and user-supplied backends
    that accept dict payloads keep working.
    """
    add = getattr(session, "add_items", None)
    if add is None:
        return
    res = add(items)
    if asyncio.iscoroutine(res):
        await res


def _coerce_output_type(raw: str, output_type: type) -> Any:
    """Best-effort parse of final assistant text into ``output_type``.

    Supports:
    - ``str`` (pass-through)
    - Pydantic v1/v2 BaseModel subclasses
    - ``@dataclass`` classes
    - Any class with a ``model_validate_json`` / ``parse_raw`` class-method
    - ``dict`` / ``list`` (json-loaded)

    Returns the parsed value, or ``raw`` if parsing fails.
    """
    if output_type is str:
        return raw
    if output_type in (dict, list):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    # Pydantic v2
    if hasattr(output_type, "model_validate_json"):
        try:
            return output_type.model_validate_json(raw)
        except Exception:
            pass
    # Pydantic v1
    if hasattr(output_type, "parse_raw"):
        try:
            return output_type.parse_raw(raw)
        except Exception:
            pass
    # Dataclass
    try:
        import dataclasses as _dc
        if _dc.is_dataclass(output_type):
            data = json.loads(raw)
            if isinstance(data, dict):
                return output_type(**data)
    except Exception:
        pass
    return raw


@dataclass
class AgentState:
    messages: list[LLMMessage]
    current_task: str
    status: AgentStatus
    result: str
    iterations: int
    max_iterations: int
    tool_calls: int
    trajectory_file: str = ""
    session_file: str = ""
    # New-style aggregate state populated by the loop and exposed to callers.
    usage: Usage = field(default_factory=Usage)
    run_context: RunContext = field(default_factory=RunContext)
    final_output: Any = None
    guardrail_triggered: Optional[str] = None


BASE_SYSTEM_PROMPT = """You are a ClawAgent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls.

## Core Behavior
- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- If the request is ambiguous, ask questions before acting.

## Doing Tasks
When the user asks you to do something:
1. Think briefly about your approach, then act immediately using tools.
2. After getting tool results, continue using more tools or provide the final answer.
3. When done, provide the final answer directly. Do NOT ask if the user wants more.

Keep working until the task is fully complete.

## Efficiency Rules
- NEVER re-read a file you already have in context. Use the data from previous tool results.
- NEVER call the same tool with the same arguments twice. If you already have the result, use it.
- Batch independent tool calls into a single response when possible (use the array syntax).
- Prefer fewer, well-targeted tool calls over many exploratory ones.
- Use todo/planning tools only for broad or long-running tasks. Skip todo bookkeeping for bounded lookup, read, compare, or JSON-report tasks.
- Once tool results contain enough evidence to answer, stop calling tools and answer directly. Do not call tools only to mark progress complete."""


# ─── Adaptive Token Estimation (learned from deepagents) ──────────────────
# Now uses tiktoken for accurate BPE counting (with fallback to heuristic).

from clawagents.tokenizer import count_tokens, count_tokens_content, count_messages_tokens as _count_messages_tokens

# Keep _CHARS_PER_TOKEN for the Tier-3 preflight char-budget calculation only
_CHARS_PER_TOKEN = 4


def _estimate_tokens(content: str | list[dict], multiplier: float = 1.0, model: str | None = None) -> int:
    return count_tokens_content(content, model=model, multiplier=multiplier)


def _estimate_messages_tokens(messages: list[LLMMessage], multiplier: float = 1.0, model: str | None = None) -> int:
    return _count_messages_tokens(messages, model=model, multiplier=multiplier)


# ─── Tool Argument Truncation in Old Messages (learned from deepagents) ───

_MAX_ARG_LENGTH = 2000
_ARG_TRUNCATION_MARKER = "...(argument truncated)"
_RECENT_PROTECTED_COUNT = 20
_TRUNCATABLE_RE = re.compile(
    r'\{"tool":\s*"(write_file|edit_file|create_file)".*?"args":\s*\{'
)


def _truncate_old_tool_args(
    messages: list[LLMMessage], protect_recent: int = _RECENT_PROTECTED_COUNT,
) -> list[LLMMessage]:
    if len(messages) <= protect_recent:
        return messages

    cutoff = len(messages) - protect_recent
    result: list[LLMMessage] = []

    for i, m in enumerate(messages):
        if (
            i < cutoff
            and m.role == "assistant"
            and isinstance(m.content, str)
            and _TRUNCATABLE_RE.search(m.content)
            and len(m.content) > _MAX_ARG_LENGTH
        ):
            result.append(LLMMessage(
                role=m.role,
                content=m.content[:_MAX_ARG_LENGTH] + _ARG_TRUNCATION_MARKER,
            ))
        else:
            result.append(m)

    return result


# ─── Tool Loop Detection ──────────────────────────────────────────────────


class _ToolCallTracker:
    def __init__(
        self,
        window_size: int = 30,
        soft_limit: int = 3,
        hard_limit: int = 6,
        circuit_breaker_limit: int = 30,
    ):
        self._history: list[str] = []
        self._window_size = window_size
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._circuit_breaker_limit = circuit_breaker_limit
        self._result_hashes: dict[str, str] = {}
        self._no_progress_count = 0
        self._soft_warnings = 0

    def _key(self, tool_name: str, args: dict) -> str:
        try:
            return f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        except (TypeError, ValueError):
            return f"{tool_name}:{args}"

    @staticmethod
    def _hash_result(output: str) -> str:
        sample = output[:500]
        h = 0
        for ch in sample:
            h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        return str(h)

    def record(self, tool_name: str, args: dict) -> None:
        self._history.append(self._key(tool_name, args))
        if len(self._history) > self._window_size:
            self._history.pop(0)

    def record_result(self, tool_name: str, args: dict, output: str) -> None:
        """Record the result of a tool call for no-progress detection."""
        key = self._key(tool_name, args)
        result_hash = self._hash_result(output)
        prev_hash = self._result_hashes.get(key)
        if prev_hash == result_hash:
            self._no_progress_count += 1
        else:
            self._no_progress_count = max(0, self._no_progress_count - 1)
        self._result_hashes[key] = result_hash

    def is_ping_ponging(self) -> bool:
        """Detect A->B->A->B ping-pong oscillation (last 6 entries)."""
        if len(self._history) < 4:
            return False
        recent = self._history[-6:]
        if len(recent) < 4:
            return False
        unique = set(recent)
        if len(unique) != 2:
            return False
        for i in range(len(recent) - 1):
            if recent[i] == recent[i + 1]:
                return False
        return True

    def is_circuit_broken(self) -> bool:
        """Global circuit breaker: too many no-progress calls."""
        return self._no_progress_count >= self._circuit_breaker_limit

    def _count_occurrences(self, tool_name: str, args: dict) -> int:
        key = self._key(tool_name, args)
        return self._history.count(key)

    def is_soft_looping(self, tool_name: str, args: dict) -> bool:
        return self._count_occurrences(tool_name, args) >= self._soft_limit

    def is_hard_looping(self, tool_name: str, args: dict) -> bool:
        return self._count_occurrences(tool_name, args) >= self._hard_limit

    def is_soft_looping_batch(self, calls: list[ParsedToolCall]) -> bool:
        return any(self.is_soft_looping(c.tool_name, c.args) for c in calls)

    def is_hard_looping_batch(self, calls: list[ParsedToolCall]) -> bool:
        return any(self.is_hard_looping(c.tool_name, c.args) for c in calls)

    def record_batch(self, calls: list[ParsedToolCall]) -> None:
        for c in calls:
            self.record(c.tool_name, c.args)

    def bump_soft_warning(self) -> int:
        self._soft_warnings += 1
        return self._soft_warnings


# ─── Consecutive Failure Detection ────────────────────────────────────────
# Tracks tool-call success/failure to detect persistent failure streaks.
# When N consecutive tool calls fail, injects a "step back and rethink"
# message — lightweight online adaptation inspired by OpenClaw-RL's
# next-state reward signal.

_RETHINK_THRESHOLD = 3
_MAX_RETHINKS = 3

_RETHINK_MESSAGE = (
    "[System] Your last {n} tool calls all failed. "
    "Stop and reconsider your approach before trying again. "
    "Review the errors above, think about what went wrong, "
    "and try a fundamentally different strategy."
)


_SCORELESS_TOOLS: frozenset[str] = frozenset({
    "think", "todolist", "todo_write", "todo_read", "use_skill", "ask_user",
})


class _FailureTracker:
    """Track consecutive tool failures to trigger rethink injection.

    Scoreless tools (think, todolist, etc.) are excluded — their results
    are not meaningful signals for failure detection.
    """

    def __init__(self, threshold: int = _RETHINK_THRESHOLD, max_rethinks: int = _MAX_RETHINKS):
        self._results: list[bool] = []  # True = success, False = failure
        self._threshold = threshold
        self._max_rethinks = max_rethinks
        self._rethink_count = 0

    def record(self, success: bool, tool_name: str = "") -> None:
        if tool_name in _SCORELESS_TOOLS:
            return
        self._results.append(success)

    def record_batch(self, results: list[tuple[bool, str]]) -> None:
        for success, name in results:
            self.record(success, name)

    def should_rethink(self) -> bool:
        if self._rethink_count >= self._max_rethinks:
            return False
        if len(self._results) < self._threshold:
            return False
        return all(not s for s in self._results[-self._threshold:])

    def bump_rethink(self) -> int:
        self._rethink_count += 1
        self._results.clear()
        return self._rethink_count

    @property
    def consecutive_failures(self) -> int:
        count = 0
        for s in reversed(self._results):
            if not s:
                count += 1
            else:
                break
        return count


# ─── Pre-flight Context Guard ─────────────────────────────────────────────
# Runs once before the main loop to ensure the initial payload fits in the
# context window. Applies graduated shedding when the system prompt + tool
# descriptions + user task already exceed the budget.

_MAX_OVERFLOW_RETRIES = 3


def _preflight_context_check(
    messages: list[LLMMessage],
    context_window: int,
    tool_desc: str,
    native_schemas: list[NativeToolSchema] | None,
    registry: ToolRegistry | None,
    emit: OnEvent,
    model_name: Optional[str] = None,
) -> tuple[list[LLMMessage], str, list[NativeToolSchema] | None]:
    """Ensure the initial payload fits in the context budget.

    Returns (messages, tool_desc, native_schemas) — possibly modified via
    graduated shedding.

    Tiers:
      1. Truncate verbose tool parameter descriptions
      2. Drop text-based tool descriptions if native schemas are available
      3. Truncate the system prompt itself, keeping the core behavior section
    """
    effective_window, ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    budget = int(effective_window * ratio)

    native_schema_tokens = 0
    if native_schemas:
        schema_text = json.dumps([
            {"name": s.name, "description": s.description, "parameters": s.parameters}
            for s in native_schemas
        ])
        native_schema_tokens = _estimate_tokens(schema_text)

    def _payload_tokens() -> int:
        return _estimate_messages_tokens(messages) + native_schema_tokens

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    emit("context", {
        "message": f"pre-flight: initial payload ~{_payload_tokens()} tokens exceeds budget {budget}"
    })

    # ── Tier 1: Truncate parameter descriptions in tool_desc ──────────
    if tool_desc and registry:
        short_parts = ["## Available Tools\n"]
        for tool in registry.list():
            short_parts.append(f"### {tool.name}\n{tool.description}")
            if tool.parameters:
                short_parts.append("Parameters: " + ", ".join(
                    f"`{k}` ({v.get('type', 'string')}{'*' if v.get('required') else ''})"
                    for k, v in tool.parameters.items()
                ))
            short_parts.append("")
        short_desc = "\n".join(short_parts)
        sys_msg = messages[0]
        if isinstance(sys_msg.content, str):
            messages = [
                LLMMessage(role="system", content=sys_msg.content.replace(tool_desc, short_desc)),
                *messages[1:],
            ]
            tool_desc = short_desc
            emit("context", {"message": f"tier-1: shortened tool descriptions -> ~{_payload_tokens()} tokens"})
        else:
            emit("warn", {
                "message": "tier-1 shedding skipped: system message has multimodal content (list), cannot string-replace"
            })

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    # ── Tier 2: Drop text tool descriptions if native schemas exist ───
    if tool_desc and native_schemas:
        sys_msg = messages[0]
        if isinstance(sys_msg.content, str):
            messages = [
                LLMMessage(role="system", content=sys_msg.content.replace(tool_desc, "").strip()),
                *messages[1:],
            ]
            tool_desc = ""
            emit("context", {"message": f"tier-2: removed text tool descriptions -> ~{_payload_tokens()} tokens"})
        else:
            emit("warn", {
                "message": "tier-2 shedding skipped: system message has multimodal content (list), cannot string-replace"
            })

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    # ── Tier 3: Truncate system prompt, preserving core behavior ──────
    sys_content = messages[0].content
    max_sys_chars = int((budget - native_schema_tokens - _estimate_tokens(messages[1].content if len(messages) > 1 else "")) * _CHARS_PER_TOKEN * 0.8)
    if isinstance(sys_content, str):
        if max_sys_chars > 200 and len(sys_content) > max_sys_chars:
            truncated = sys_content[:max_sys_chars] + "\n\n...(system prompt truncated to fit context window)"
            messages = [LLMMessage(role="system", content=truncated), *messages[1:]]
            emit("context", {"message": f"tier-3: truncated system prompt -> ~{_payload_tokens()} tokens"})
    else:
        emit("warn", {
            "message": "tier-3 shedding skipped: system message has multimodal content (list), cannot truncate as string"
        })

    if _payload_tokens() > budget:
        emit("warn", {
            "message": (
                f"pre-flight: payload still ~{_payload_tokens()} tokens after all shedding "
                f"(budget {budget}). Consider increasing CONTEXT_WINDOW or reducing tools/instruction."
            )
        })

    return messages, tool_desc, native_schemas


# ─── Micro-Compact: clear old tool results (learned from Claude Code) ─────
# Unlike soft-trim which truncates, micro-compact completely replaces old tool
# result content with a placeholder. The model still sees the tool_use →
# tool_result structure (knows *what* it did) but not the raw output.
# This can effectively double the usable context window with zero LLM overhead.

_COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "read_file", "execute", "execute_command", "bash", "run_command",
    "grep", "glob", "ls", "tree", "web_fetch", "web_search",
    "search_files", "list_dir", "find_files",
})

_MICRO_COMPACT_KEEP_RECENT = 3  # keep last N compactable tool results intact


def _micro_compact_tool_results(
    messages: list[LLMMessage],
    keep_recent: int = _MICRO_COMPACT_KEEP_RECENT,
) -> list[LLMMessage]:
    """Clear old tool result content for compactable tools (keep last N).

    The model still sees the tool_use → tool_result pairs, just not the raw
    50KB grep/file output. This preserves the agent's sense of *what* it did
    while freeing massive amounts of context.
    """
    from clawagents.config.features import is_enabled
    if not is_enabled("micro_compact"):
        return messages

    # Collect compactable tool call IDs in order
    compactable_ids: list[str] = []
    # For text-based tool calls, track by message index
    compactable_text_indices: list[int] = []

    for i, msg in enumerate(messages):
        if msg.role == "assistant":
            # Native tool calls
            if msg.tool_calls_meta:
                for tc in msg.tool_calls_meta:
                    if tc.get("name", "") in _COMPACTABLE_TOOLS:
                        compactable_ids.append(tc["id"])
            # Text-based tool calls
            elif isinstance(msg.content, str):
                try:
                    import json as _json
                    parsed = _json.loads(msg.content)
                    if isinstance(parsed, dict) and parsed.get("tool") in _COMPACTABLE_TOOLS:
                        compactable_text_indices.append(i)
                    elif isinstance(parsed, list):
                        if any(isinstance(item, dict) and item.get("tool") in _COMPACTABLE_TOOLS for item in parsed):
                            compactable_text_indices.append(i)
                except (ValueError, TypeError):
                    pass

    # Keep the most recent N compactable tool results
    keep_ids = set(compactable_ids[-keep_recent:])
    keep_text_indices = set(compactable_text_indices[-keep_recent:])

    # Clear old compactable tool results
    result: list[LLMMessage] = []
    cleared = 0
    for i, msg in enumerate(messages):
        # Native tool results
        if msg.role == "tool" and msg.tool_call_id:
            if msg.tool_call_id in compactable_ids and msg.tool_call_id not in keep_ids:
                result.append(LLMMessage(
                    role="tool",
                    content="[Old tool result cleared to save context]",
                    tool_call_id=msg.tool_call_id,
                ))
                cleared += 1
                continue
        # Text-based tool results (user message following assistant tool call)
        elif msg.role == "user" and isinstance(msg.content, str) and msg.content.startswith("[Tool Result]"):
            if i > 0 and (i - 1) in compactable_text_indices and (i - 1) not in keep_text_indices:
                result.append(LLMMessage(
                    role="user",
                    content="[Tool Result] [Old tool result cleared to save context]",
                ))
                cleared += 1
                continue

        result.append(msg)

    return result


# ─── Soft-Trim: prune stale/low-value content before compaction ───────────

_SOFT_TRIM_BUDGET_FRACTION = 0.75  # soft-trim at 75% of the compaction budget_ratio
_SOFT_TRIM_RESULT_MAX_CHARS = 1000
_SOFT_TRIM_RESULT_KEEP_CHARS = 500
_SOFT_TRIM_RECENT_PROTECTED = 10

_IMAGE_DATA_RE = re.compile(r'^\[image\s*data?\]$', re.IGNORECASE)


def _soft_trim_messages(
    messages: list[LLMMessage],
    context_window: int,
    token_multiplier: float,
    emit: OnEvent,
    model_name: Optional[str] = None,
) -> list[LLMMessage]:
    """Remove stale/low-value content from context before hitting compaction threshold."""
    effective_window, budget_ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    soft_budget = int(effective_window * budget_ratio * _SOFT_TRIM_BUDGET_FRACTION)
    current_tokens = _estimate_messages_tokens(messages, token_multiplier)

    if current_tokens <= soft_budget:
        return messages

    protect_from = max(0, len(messages) - _SOFT_TRIM_RECENT_PROTECTED * 2)
    trim_count = 0

    # First pass: identify duplicate tool results and mark latest index
    seen: dict[str, int] = {}
    for i, m in enumerate(messages):
        if m.role == "tool" or (m.role == "user" and isinstance(m.content, str) and m.content.startswith("[Tool Result]")):
            if i > 0:
                prev = messages[i - 1]
                if prev.role == "assistant" and isinstance(prev.content, str):
                    content_str = m.content if isinstance(m.content, str) else ""
                    key = prev.content[:200] + "|" + content_str[:200]
                    seen[key] = i

    # Second pass: trim/prune
    result: list[LLMMessage] = []
    for i, m in enumerate(messages):
        if i >= protect_from:
            result.append(m)
            continue

        is_tool_result = (
            m.role == "tool"
            or (m.role == "user" and isinstance(m.content, str) and m.content.startswith("[Tool Result]"))
        )

        if is_tool_result and isinstance(m.content, str):
            # Prune image-only tool results from early turns
            trimmed_content = m.content.replace("[Tool Result]", "", 1).strip()
            if _IMAGE_DATA_RE.match(trimmed_content):
                result.append(LLMMessage(role=m.role, content="[Tool Result] [image data removed — stale]",
                                         tool_call_id=m.tool_call_id))
                trim_count += 1
                continue

            # Remove duplicate tool results (keep only the most recent)
            if i > 0:
                prev = messages[i - 1]
                if prev.role == "assistant" and isinstance(prev.content, str):
                    key = prev.content[:200] + "|" + m.content[:200]
                    latest_idx = seen.get(key)
                    if latest_idx is not None and latest_idx != i:
                        result.append(LLMMessage(role=m.role, content="[Tool Result] [duplicate — see later result]",
                                                 tool_call_id=m.tool_call_id))
                        trim_count += 1
                        continue

            # Trim large old tool results
            if len(m.content) > _SOFT_TRIM_RESULT_MAX_CHARS:
                half = _SOFT_TRIM_RESULT_KEEP_CHARS // 2
                trimmed = (
                    m.content[:half]
                    + f"\n...[soft-trimmed {len(m.content) - _SOFT_TRIM_RESULT_KEEP_CHARS} chars]...\n"
                    + m.content[-half:]
                )
                result.append(LLMMessage(role=m.role, content=trimmed, tool_call_id=m.tool_call_id))
                trim_count += 1
                continue

        result.append(m)

    if trim_count > 0:
        emit("context", {"message": f"soft-trim: trimmed {trim_count} old tool results"})
    return result


# ─── Context Window Guard with Auto-Compaction ────────────────────────────

_CONTEXT_BUDGET_RATIO = 0.75
_RECENT_MESSAGES_TO_KEEP = 20
_COMPACTION_CHUNK_TOKENS = 30_000
_COMPACTION_MAX_RETRIES = 3

_IDENTIFIER_PRESERVATION = """
CRITICAL: Preserve these verbatim (do not paraphrase or omit):
- File paths (e.g., src/utils/auth.ts)
- Function/variable/class names (e.g., handleAuth, userToken)
- Error messages and stack traces
- Command-line commands that were run
- Configuration values and URLs"""


def _find_safe_split_index(non_system: list[LLMMessage], desired_recent: int) -> int:
    """Find a split index that doesn't break tool_call/tool_result pairs.

    Walks backward from the desired split point until we find a boundary
    that doesn't land between an assistant tool_call and its tool result.
    """
    split = max(0, len(non_system) - desired_recent)
    while split < len(non_system) - 1:
        msg = non_system[split]
        if msg.role == "tool" and msg.tool_call_id:
            split += 1
            continue
        break
    return split


async def _summarize_chunk(
    llm: LLMProvider,
    chunk_text: str,
    task_context: str,
) -> str:
    """Summarize a single chunk with retry and exponential backoff."""
    prompt = (
        "You are summarizing a chunk of an AI agent's conversation history.\n\n"
        f"## Original Task\n{task_context}\n\n"
        f"## Conversation Chunk\n{chunk_text}\n\n"
        "## Instructions\n"
        "Write a structured summary preserving:\n"
        "- What tools were called and their key results (file paths, data, errors)\n"
        "- What has been accomplished\n"
        "- Any critical facts, variable values, or decisions made\n"
        + _IDENTIFIER_PRESERVATION + "\n"
        "Be concise but preserve all actionable information."
    )

    last_error: BaseException | None = None
    for attempt in range(_COMPACTION_MAX_RETRIES):
        try:
            resp = await llm.chat([LLMMessage(role="user", content=prompt)])
            if resp.content.strip():
                return resp.content.strip()
        except Exception as e:
            last_error = e
        if attempt < _COMPACTION_MAX_RETRIES - 1:
            await asyncio.sleep(1.0 * (2 ** attempt))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Summarization returned empty")


async def _compact_if_needed(
    messages: list[LLMMessage],
    context_window: int,
    llm: LLMProvider,
    emit: OnEvent,
    token_multiplier: float = 1.0,
    model_name: Optional[str] = None,
    run_context: Optional[RunContext] = None,
) -> list[LLMMessage]:
    messages = _truncate_old_tool_args(messages)

    effective_window, ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    budget = int(effective_window * ratio)
    current_tokens = _estimate_messages_tokens(messages, token_multiplier)

    if current_tokens <= budget:
        return messages

    emit("context", {"message": f"~{current_tokens} tokens exceeds budget {budget} — compacting"})
    emit("compact_progress", {
        "phase": "start",
        "message": "context budget exceeded; compacting older turns",
        "current_tokens": current_tokens,
        "budget": budget,
        "message_count": len(messages),
    })

    system_msgs: list[LLMMessage] = []
    non_system: list[LLMMessage] = []
    for m in messages:
        (system_msgs if m.role == "system" else non_system).append(m)

    if len(non_system) <= _RECENT_MESSAGES_TO_KEEP:
        return messages

    split_idx = _find_safe_split_index(non_system, _RECENT_MESSAGES_TO_KEEP)
    if split_idx <= 0:
        return messages

    older = non_system[:split_idx]
    recent = non_system[split_idx:]

    task_context = ""
    for m in non_system:
        if m.role == "user" and not (isinstance(m.content, str) and m.content.startswith("[Tool Result]")):
            task_context = m.content[:500] if isinstance(m.content, str) else ""
            break
    carryover = get_compaction_carryover(run_context, task_context=task_context)

    _archive_pre_compact_transcript(older, task_context)

    offload_path = _offload_history(older)
    if offload_path:
        emit("context", {"message": f"offloaded {len(older)} messages to {offload_path}"})

    text_parts: list[str] = []
    for m in older:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if m.role == "assistant" and m.tool_calls_meta:
            calls = ", ".join(tc["name"] for tc in m.tool_calls_meta)
            text_parts.append(f"[TOOL CALLS: {calls}] {content[:200]}")
        elif m.role == "tool":
            text_parts.append(f"[TOOL RESULT]: {content[:200]}")
        else:
            text_parts.append(f"[{m.role.upper()}]: {content[:500]}")

    total_tokens = _estimate_tokens("\n\n".join(text_parts), token_multiplier)

    try:
        emit("compact_progress", {
            "phase": "summarize",
            "message": "summarizing compacted turns",
            "older_messages": len(older),
            "recent_messages": len(recent),
            "carryover": carryover.to_dict(),
        })
        if total_tokens <= _COMPACTION_CHUNK_TOKENS:
            text_log = "\n\n".join(text_parts)
            summary_text = await _summarize_chunk(llm, text_log, task_context)
        else:
            chunks: list[str] = []
            current_chunk: list[str] = []
            current_chunk_tokens = 0

            for part in text_parts:
                part_tokens = _estimate_tokens(part, token_multiplier)
                if current_chunk_tokens + part_tokens > _COMPACTION_CHUNK_TOKENS and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_chunk_tokens = 0
                current_chunk.append(part)
                current_chunk_tokens += part_tokens
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            emit("context", {
                "message": f"splitting {len(text_parts)} parts into {len(chunks)} chunks for summarization",
            })
            emit("compact_progress", {
                "phase": "chunk",
                "message": "splitting older turns into summary chunks",
                "chunks": len(chunks),
                "older_messages": len(older),
                "recent_messages": len(recent),
            })

            chunk_summaries: list[str] = []
            for i, chunk in enumerate(chunks):
                chunk_summary = await _summarize_chunk(llm, chunk, task_context)
                chunk_summaries.append(f"### Chunk {i + 1}/{len(chunks)}\n{chunk_summary}")
            summary_text = "\n\n".join(chunk_summaries)

        if not summary_text.strip():
            emit("context", {"message": "compaction returned empty summary — dropping oldest"})
            emit("compact_progress", {
                "phase": "dropped",
                "message": "empty compaction summary; dropped older turns",
                "older_messages": len(older),
                "recent_messages": len(recent),
                "carryover": carryover.to_dict(),
            })
            return [*system_msgs, *recent]

        carryover_text = carryover.to_markdown()
        content = f"[System — Compacted History]\n{summary_text}"
        if carryover_text:
            content = f"[System — Compacted History]\n{carryover_text}\n\n## Conversation Summary\n{summary_text}"
        summary = LLMMessage(
            role="user",
            content=content,
        )
        emit("context", {"message": f"compacted {len(older)} messages into summary"})
        emit("compact_progress", {
            "phase": "end",
            "message": "compaction completed",
            "older_messages": len(older),
            "recent_messages": len(recent),
            "carryover": carryover.to_dict(),
        })
        return [*system_msgs, summary, *recent]
    except Exception:
        logger.debug("Compaction LLM call failed", exc_info=True)
        emit("context", {"message": "compaction failed — dropping oldest messages"})
        emit("compact_progress", {
            "phase": "failed",
            "message": "compaction failed; dropped older turns",
            "older_messages": len(older),
            "recent_messages": len(recent),
            "carryover": carryover.to_dict(),
        })
        return [*system_msgs, *recent]


# ─── History Offloading ───────────────────────────────────────────────────


def _get_history_dir() -> Path:
    return Path.cwd() / ".clawagents" / "history"


def _archive_pre_compact_transcript(older_messages: list[LLMMessage], task_context: str) -> None:
    """Archive full messages to a markdown file before compaction (feature-gated)."""
    from clawagents.config.features import is_enabled
    if not is_enabled("transcript_archival"):
        return

    try:
        transcript_dir = Path.cwd() / ".clawagents" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = transcript_dir / f"pre_compact_{ts}_{len(older_messages)}msgs.md"

        lines: list[str] = [
            "## Pre-Compact Transcript\n",
            f"\nTask: {task_context}\n",
            "\n### Messages\n\n",
        ]
        for m in older_messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"**{m.role}**: {content[:2000]}\n\n")

        path.write_text("".join(lines), "utf-8")
    except Exception:
        logger.debug("Pre-compact transcript archival failed", exc_info=True)


def _offload_history(messages: list[LLMMessage]) -> str | None:
    """Save older messages to a JSON file before compaction."""
    try:
        _get_history_dir().mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = _get_history_dir() / f"compacted_{ts}_{len(messages)}msgs.json"
        data = [{"role": m.role, "content": m.content} for m in messages]
        path.write_text(json.dumps(data, indent=2), "utf-8")
        return str(path)
    except Exception:
        logger.debug("History offload failed", exc_info=True)
        return None


# ─── Write-Ahead Log (learned from Claude Code) ──────────────────────────
# Persist the latest message before each LLM API call so that if the process
# crashes mid-call, the user's last message isn't lost.


def _wal_write(messages: list[LLMMessage]) -> None:
    """Append the latest message to the WAL file for crash recovery."""
    from clawagents.config.features import is_enabled
    if not is_enabled("wal"):
        return

    try:
        wal_path = Path.cwd() / ".clawagents" / "wal.jsonl"
        wal_path.parent.mkdir(parents=True, exist_ok=True)
        last_msg = messages[-1] if messages else None
        if not last_msg:
            return
        content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        entry = json.dumps({
            "role": last_msg.role,
            "content": content[:500],
            "ts": time.time(),
            "msg_count": len(messages),
        })
        with open(wal_path, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass  # WAL failure should never block the agent loop


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_buffer():
    buf: list[str] = []
    def on_chunk(chunk: str) -> None:
        buf.append(chunk)
    return buf, on_chunk


# ─── Truncated JSON Detection ─────────────────────────────────────────────

_TRUNCATED_JSON_RE = re.compile(r'\{\s*"tool"\s*:', re.DOTALL)


def _looks_like_truncated_json(text: str) -> bool:
    """Detect if text looks like a JSON tool call that was cut off mid-output."""
    stripped = text.strip()
    if not stripped:
        return False
    if not _TRUNCATED_JSON_RE.search(stripped):
        return False
    # Has what looks like a tool call but doesn't parse as valid JSON
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, (dict, list)):
            return False  # Valid JSON — not truncated
    except json.JSONDecodeError:
        pass
    # Check for fence-wrapped truncated JSON
    for m in re.finditer(r'```(?:json)?\s*\n?(.*?)(?:```|$)', stripped, re.DOTALL):
        inner = m.group(1).strip()
        if _TRUNCATED_JSON_RE.search(inner):
            try:
                json.loads(inner)
                return False
            except json.JSONDecodeError:
                return True
    return True


# ─── ReAct Loop ──────────────────────────────────────────────────────────

MAX_TOOL_ROUNDS = 1000


async def run_agent_graph(
    task: str,
    llm: LLMProvider,
    tools: Optional[ToolRegistry] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_TOOL_ROUNDS,
    streaming: bool = True,
    context_window: int = 1_000_000,
    on_event: Optional[OnEvent] = None,
    before_llm: Optional[BeforeLLMHook] = None,
    before_tool: Optional[BeforeToolHook] = None,
    after_tool: Optional[AfterToolHook] = None,
    use_native_tools: bool = True,
    trajectory: bool = False,
    rethink: bool = False,
    learn: bool = False,
    preview_chars: int = 120,
    response_chars: int = 500,
    timeout_s: float = 0,
    features: Optional[dict[str, bool]] = None,
    advisor_llm: Optional[LLMProvider] = None,
    advisor_max_calls: int = 3,
    # ── New, fully backward-compatible keyword-only parameters ──
    run_context: Optional[RunContext] = None,
    user_context: Any = None,
    hooks: Optional[RunHooks] = None,
    agent_hooks: Optional[AgentHooks] = None,
    input_guardrails: Optional[list[InputGuardrail]] = None,
    output_guardrails: Optional[list[OutputGuardrail]] = None,
    output_type: Optional[type] = None,
    on_stream_event: Optional[Callable[[StreamEvent], None]] = None,
    session: Optional[Any] = None,  # clawagents.session.Session protocol
    session_preload_limit: int | None = 200,
    handoffs: Optional[list[Handoff]] = None,
    agent_name: Optional[str] = None,
) -> AgentState:
    """Single ReAct loop: LLM → tools → LLM → tools → ... → final answer."""
    if features is not None:
        from clawagents.config.features import set_overrides
        set_overrides(features)

    registry = tools or ToolRegistry()
    native_schemas: list[NativeToolSchema] | None = (
        registry.to_native_schemas() if use_native_tools and tools else None
    )
    tool_desc = registry.describe_for_llm() if not use_native_tools else ""
    loop_tracker = _ToolCallTracker()
    emit = on_event or _default_on_event

    # ── Synthesise handoff tools (v6.4) ──
    # Each Handoff becomes a synthetic tool the LLM can call. We DO NOT add
    # these to the registry — they're dispatched directly by the loop so
    # they can switch the active agent rather than execute a tool. We also
    # build a name → Handoff map for fast lookup at dispatch time.
    handoff_list: list[Handoff] = list(handoffs) if handoffs else []
    handoff_map: dict[str, Handoff] = {h.name: h for h in handoff_list}
    if handoff_list:
        handoff_params = {
            "reason": {
                "type": "string",
                "description": "Free-text rationale for why the handoff is appropriate.",
                "required": False,
            }
        }
        if use_native_tools:
            if native_schemas is None:
                native_schemas = []
            for h in handoff_list:
                native_schemas.append(NativeToolSchema(
                    name=h.name,
                    description=h.description,
                    parameters=handoff_params,
                ))
        else:
            # Append handoff descriptions to the text-mode tool block so the
            # LLM still discovers them.
            extra_lines = ["", "## Handoffs"]
            for h in handoff_list:
                extra_lines.append(f"### {h.name}\n{h.description}")
                extra_lines.append("Parameters:")
                extra_lines.append("- `reason` (string): Free-text rationale.")
                extra_lines.append("")
            tool_desc = (tool_desc or "") + "\n" + "\n".join(extra_lines)

    # ── Typed run context + usage accumulator ──
    if run_context is None:
        run_context = RunContext(context=user_context)
    elif user_context is not None and run_context.context is None:
        run_context.context = user_context
    usage = run_context.usage

    # Per-agent iteration budget (Hermes parity). If the caller has not
    # already attached one (e.g., through a subagent-spawning path that
    # creates a fresh budget), build one sized to ``max_iterations`` so
    # the loop has a single source of truth for "are we out of turns?".
    # We size it to ``max_iterations`` directly; the existing ``for
    # round_idx in range(effective_max_rounds)`` loop still acts as a
    # belt-and-braces hard ceiling, but the budget is the user-visible
    # control surface.
    _budget_size = max_iterations if max_iterations > 0 else MAX_TOOL_ROUNDS
    await run_context.ensure_iteration_budget(_budget_size)

    def _emit_typed(kind: str, data: dict[str, Any] | None = None) -> None:
        """Dispatch a typed StreamEvent alongside the existing ``emit`` hook."""
        if on_stream_event is None:
            return
        try:
            on_stream_event(stream_event_from_kind(kind, data or {}))
        except Exception as err:
            emit("warn", {"message": f"on_stream_event error: {err}"})

    def _accumulate_usage(resp: LLMResponse) -> RequestUsage:
        prompt_t = int(getattr(resp, "prompt_tokens", 0) or 0)
        total_t = int(getattr(resp, "tokens_used", 0) or 0)
        output_t = int(getattr(resp, "completion_tokens", max(total_t - prompt_t, 0)) or 0)
        req = usage.add_response(
            model=getattr(resp, "model", None) or "",
            input_tokens=prompt_t,
            output_tokens=output_t,
            total_tokens=total_t,
            cached_input_tokens=int(getattr(resp, "cache_read_tokens", 0) or 0),
            cache_creation_tokens=int(getattr(resp, "cache_creation_tokens", 0) or 0),
        )
        _emit_typed("usage", {
            "input_tokens": req.input_tokens,
            "output_tokens": req.output_tokens,
            "total_tokens": req.total_tokens,
            "cached_input_tokens": req.cached_input_tokens,
            "cache_creation_tokens": req.cache_creation_tokens,
            "model": req.model,
        })
        return req

    # RunHooks / AgentHooks — combine into a single call list.
    active_hooks: list[RunHooks] = []
    if hooks is not None:
        active_hooks.append(hooks)
    if agent_hooks is not None and agent_hooks is not hooks:
        active_hooks.append(agent_hooks)

    async def _fire_hook(method_name: str, *args: Any) -> None:
        for h in active_hooks:
            fn = getattr(h, method_name, None)
            if fn is None:
                continue
            try:
                result = fn(run_context, *args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as hook_err:
                emit("warn", {"message": f"{method_name} hook error: {hook_err}"})

    # Feature C + F: detect task type for adaptive rethink threshold
    _task_type = "general"
    if rethink or learn:
        try:
            from clawagents.trajectory.verifier import detect_task_type, compute_adaptive_rethink_threshold
            _task_type = detect_task_type(task)
            adaptive_threshold = compute_adaptive_rethink_threshold(_task_type, 0, 0)
        except Exception:
            adaptive_threshold = _RETHINK_THRESHOLD
    else:
        adaptive_threshold = _RETHINK_THRESHOLD
    failure_tracker = _FailureTracker(threshold=adaptive_threshold) if rethink else None

    # Trajectory recorder (opt-in; learn implies trajectory)
    recorder = None
    if trajectory or learn:
        from clawagents.trajectory.recorder import TrajectoryRecorder
        recorder = TrajectoryRecorder(task=task, response_chars=response_chars)

    # Feature: Session Persistence — save session as append-only JSONL
    session_writer = None
    from clawagents.config.features import is_enabled as _feat_enabled
    if _feat_enabled("session_persistence"):
        from clawagents.session.persistence import SessionWriter
        session_writer = SessionWriter()
        emit("context", {"message": f"session: {session_writer.session_id} → {session_writer.path}"})

    # Feature: External Hooks — load shell hooks from .clawagents/hooks.json or env
    ext_hook_runner = None
    if _feat_enabled("external_hooks"):
        from clawagents.hooks.external import load_hooks_config, ExternalHookRunner
        hooks_cfg = load_hooks_config()
        if hooks_cfg:
            ext_hook_runner = ExternalHookRunner(hooks_cfg)
            emit("context", {"message": "external hooks: loaded"})

    token_multiplier = 1.0
    resolved_model_name: Optional[str] = None
    _cached_sys_tokens: int = 0  # Feature D: cache system prompt token count
    _last_memory_extraction_turn: int = 0  # Background memory extraction cursor

    # ── Advisor model: phone-a-friend for strategic guidance ────────
    _advisor_call_count = 0

    async def _consult_advisor(msgs: list[LLMMessage], trigger: str) -> None:
        nonlocal _advisor_call_count
        if not advisor_llm or _advisor_call_count >= advisor_max_calls:
            return
        _advisor_call_count += 1
        emit("context", {"message": f"advisor consultation #{_advisor_call_count} ({trigger})"})
        try:
            advisor_response = await advisor_llm.chat([
                LLMMessage(role="system", content="You are a senior advisor. Review the agent's full transcript and provide concise strategic guidance. Under 150 words. Use numbered steps, not explanations."),
                *msgs,
                LLMMessage(role="user", content=f"[Advisor Request — {trigger}] Review the conversation above and provide strategic guidance for the next steps."),
            ])
            if advisor_response.content:
                msgs.append(LLMMessage(role="user", content=f"[Advisor Guidance]\n{advisor_response.content}"))
                emit("context", {"message": f"advisor: {advisor_response.content[:120]}..."})
        except Exception as err:
            emit("warn", {"message": f"advisor consultation failed: {err}"})

    prompt_to_use = system_prompt or BASE_SYSTEM_PROMPT
    lesson_preamble = ""

    # PTRL Layer 1: Pre-run lesson injection (skipped for isolated subagents).
    if learn and not getattr(run_context, "skip_memory", False):
        from clawagents.trajectory.lessons import build_lesson_preamble
        preamble = build_lesson_preamble()
        if preamble:
            lesson_preamble = preamble
            emit("context", {"message": "PTRL: injected lessons from past runs"})

    # Insert __CACHE_BOUNDARY__ between static (instructions + tools) and dynamic content.
    # The Anthropic provider splits on this marker to enable prompt caching.
    system_content = build_system_prompt(
        base_prompt=prompt_to_use,
        tool_description=tool_desc,
        lesson_preamble=lesson_preamble,
    )
    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=system_content),
        LLMMessage(role="user", content=task),
    ]

    # Session: write initial state
    if session_writer:
        session_writer.write_system_prompt(system_content)

    # Pre-flight: ensure initial payload fits in context window
    messages, tool_desc, native_schemas = _preflight_context_check(
        messages, context_window, tool_desc, native_schemas, registry, emit,
    )

    # Feature D: cache system prompt tokens (static prefix never changes)
    if messages:
        _cached_sys_tokens = _estimate_tokens(messages[0].content)
        emit("context", {"message": f"system prompt: ~{_cached_sys_tokens} tokens (cached for budget calc)"})

    state = AgentState(
        messages=messages,
        current_task=task,
        status="running",
        result="",
        iterations=0,
        max_iterations=max_iterations,
        tool_calls=0,
        usage=usage,
        run_context=run_context,
    )

    # Session protocol — hydrate history before first LLM call (non-destructive).
    _session_preloaded_count = 0
    if session is not None:
        try:
            prior = await _session_get_items(session, limit=session_preload_limit)
            if prior:
                # Keep original system + user (task), append replayed history after.
                messages = messages + prior
                state.messages = messages
                _session_preloaded_count = len(prior)
        except Exception as err:
            emit("warn", {"message": f"session load failed: {err}"})
    # Snapshot of messages we brought in before the loop runs; anything
    # appended after this cursor is what we'll persist back.
    _session_start_cursor = len(messages)

    # RunHooks: on_run_start
    if active_hooks:
        await _fire_hook("on_run_start", task)
    _emit_typed("turn_started", {"iteration": 0, "task": task})

    # Input guardrails (short-circuit before the first LLM call).
    if input_guardrails:
        try:
            tripped = await _run_input_guardrails(
                input_guardrails, run_context, task,
            )
        except GuardrailTripwireTriggered as tripwire:
            state.status = "done"
            state.result = (
                tripwire.result.message
                or f"Input rejected by guardrail '{tripwire.guardrail_name}'"
            )
            state.guardrail_triggered = tripwire.guardrail_name
            _emit_typed("guardrail_tripped", {
                "guardrail_name": tripwire.guardrail_name,
                "where": "input",
                "behavior": tripwire.result.behavior.value,
                "message": state.result,
            })
            emit("warn", {"message": f"input guardrail tripped: {tripwire.guardrail_name}"})
            if active_hooks:
                await _fire_hook("on_run_end", state.result)
            return state
        if tripped:
            messages.append(LLMMessage(role="user", content=tripped))
            _emit_typed("guardrail_tripped", {
                "guardrail_name": "input",
                "where": "input",
                "behavior": "reject_content",
                "message": tripped,
                "stage": "input",
                "rewrite": True,
            })

    overflow_retries = 0
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        emit("warn", {"message": "interrupted"})
        cancel_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except (NotImplementedError, OSError, RuntimeError):
        pass

    effective_max_rounds = min(
        max_iterations if max_iterations > 0 else MAX_TOOL_ROUNDS,
        MAX_TOOL_ROUNDS,
    )

    t0 = time.monotonic()

    def _check_timeout():
        if timeout_s > 0 and (time.monotonic() - t0) > timeout_s:
            raise TimeoutError(f"Agent run exceeded {timeout_s}s global timeout")

    try:
        for round_idx in range(effective_max_rounds):
            if cancel_event.is_set():
                state.status = "done"
                state.result = state.result or "[cancelled]"
                break

            # Consume one unit of the iteration budget. When exhausted,
            # surface the same outcome as Hermes' "max_iterations
            # reached" so trajectory recorders can flag the run as
            # truncated rather than successful. Note: ``round_idx == 0``
            # always succeeds because the budget was sized to
            # ``max_iterations`` above.
            if not run_context.iteration_budget.consume():
                emit("warn", {
                    "message": (
                        f"iteration budget exhausted "
                        f"({run_context.iteration_budget.used}/"
                        f"{run_context.iteration_budget.max_total})"
                    ),
                })
                state.status = "max_iterations"
                state.result = state.result or "[iteration budget exhausted]"
                break

            # Session: mark turn start
            if session_writer:
                session_writer.write_turn_started(round_idx)

            try:
                _check_timeout()
            except TimeoutError as te:
                emit("warn", {"message": str(te)})
                state.status = "error"
                state.result = str(te)
                break

            # ── Advisor: consult after initial orientation (first tool results in transcript)
            if advisor_llm and round_idx == 1 and _advisor_call_count == 0:
                await _consult_advisor(messages, "planning")

            # Write-ahead log: persist last message before API call (Claude Code pattern)
            _wal_write(messages)

            # Patch dangling tool calls before sending to LLM
            messages = _patch_dangling_tool_calls(messages)
            messages = _micro_compact_tool_results(messages)  # Claude Code pattern: clear old tool results
            messages = _soft_trim_messages(messages, context_window, token_multiplier, emit, resolved_model_name)
            messages = await _compact_if_needed(
                messages, context_window, llm, emit, token_multiplier, resolved_model_name, run_context,
            )

            # External pre_llm hook (runs before programmatic hook)
            if ext_hook_runner:
                try:
                    extra_msgs = await ext_hook_runner.pre_llm(
                        [{"role": m.role, "content": m.content[:100] if isinstance(m.content, str) else ""} for m in messages[-3:]]
                    )
                    if extra_msgs:
                        from typing import Literal as _Literal, cast as _cast
                        _ALLOWED_ROLES = ("system", "user", "assistant", "tool")
                        _AllowedRole = _Literal["system", "user", "assistant", "tool"]
                        for em in extra_msgs:
                            raw_role = em.get("role", "user")
                            if raw_role not in _ALLOWED_ROLES:
                                emit("warn", {
                                    "message": (
                                        f"external pre_llm hook returned message with unknown role "
                                        f"{raw_role!r}; coercing to 'user'"
                                    )
                                })
                                role: _AllowedRole = "user"
                            else:
                                # raw_role is now provably one of the allowed literal strings,
                                # but mypy can't narrow ``str`` from a tuple membership test.
                                role = _cast(_AllowedRole, raw_role)
                            messages.append(LLMMessage(role=role, content=em.get("content", "")))
                except Exception as hook_err:
                    emit("warn", {"message": f"external pre_llm hook error: {hook_err}"})

            if before_llm:
                try:
                    hooked = before_llm(messages)
                    if isinstance(hooked, list) and len(hooked) > 0:
                        messages = hooked
                    else:
                        emit("warn", {"message": "before_llm returned invalid value — ignored"})
                except Exception as hook_err:
                    emit("warn", {"message": f"before_llm hook error: {hook_err}"})

            buf, on_chunk = _make_buffer()
            if active_hooks:
                await _fire_hook("on_llm_start", resolved_model_name or "", messages)
            try:
                response = await llm.chat(
                    messages,
                    on_chunk=on_chunk if streaming else None,
                    cancel_event=cancel_event,
                    tools=native_schemas,
                )
                if not resolved_model_name and response.model:
                    resolved_model_name = response.model
                _last_request_usage = _accumulate_usage(response)
                if active_hooks:
                    await _fire_hook(
                        "on_llm_end",
                        response.model or resolved_model_name or "",
                        response.content or "",
                        _last_request_usage,
                    )

                # Session: write usage
                if session_writer:
                    session_writer.write_usage(
                        response.tokens_used,
                        cache_read_tokens=response.cache_read_tokens,
                        cache_creation_tokens=response.cache_creation_tokens,
                    )

                # External post_llm hook (fire-and-forget)
                if ext_hook_runner:
                    try:
                        await ext_hook_runner.post_llm(
                            response.content[:500],
                            len(response.tool_calls or []),
                        )
                    except Exception:
                        pass

                # Prompt cache tracking (Claude Code pattern)
                from clawagents.config.features import is_enabled as _is_feat_enabled
                if _is_feat_enabled("cache_tracking") and response.prompt_tokens > 0:
                    cache_pct = (response.cache_read_tokens / response.prompt_tokens * 100) if response.prompt_tokens > 0 else 0
                    emit("context", {
                        "message": f"cache: {cache_pct:.0f}% hit ({response.cache_read_tokens}/{response.prompt_tokens} prompt tokens, {response.cache_creation_tokens} created)"
                    })
            except Exception as err:
                # Feature: Error Taxonomy — classify and apply recovery recipe
                from clawagents.errors.taxonomy import classify_error, ErrorClass
                descriptor = classify_error(err)
                emit("error", {
                    "phase": "llm_call",
                    "message": str(err),
                    "error_class": descriptor.error_class.value,
                    "retryable": descriptor.retryable,
                    "recovery_hint": descriptor.recovery_hint,
                })

                if descriptor.error_class == ErrorClass.CONTEXT_WINDOW:
                    overflow_retries += 1
                    if overflow_retries > _MAX_OVERFLOW_RETRIES:
                        emit("error", {
                            "phase": "llm_call",
                            "message": (
                                f"context overflow persists after {_MAX_OVERFLOW_RETRIES} retries. "
                                "Increase CONTEXT_WINDOW, reduce tools, or shorten your instruction."
                            ),
                        })
                        state.status = "error"
                        state.result = str(err)
                        break
                    observed_ratio = context_window / max(
                        _estimate_messages_tokens(messages, 1.0), 1,
                    )
                    token_multiplier = min(observed_ratio * 1.1, 3.0)
                    emit("context", {
                        "message": f"token overflow — calibrated multiplier to {token_multiplier:.2f} (retry {overflow_retries}/{_MAX_OVERFLOW_RETRIES})",
                    })
                    messages = _soft_trim_messages(messages, context_window, token_multiplier, emit, resolved_model_name)
                    messages = await _compact_if_needed(
                        messages, context_window, llm, emit, token_multiplier, resolved_model_name, run_context,
                    )
                    continue

                logger.exception("LLM call failed at round %d: [%s] %s", round_idx, descriptor.error_class.value, err)
                state.status = "error"
                state.result = f"[{descriptor.error_class.value}] {descriptor.recovery_hint}"
                break

            if response.partial and not response.content.strip():
                emit("warn", {"message": "interrupted — no content received"})
                state.status = "done"
                state.result = state.result or "[interrupted]"
                break

            # Feature H: extract and preserve thinking tokens (<think>...</think>)
            _thinking_content: str | None = None
            if response.content and "<think>" in response.content:
                clean_content, _thinking_content = strip_thinking_tokens(response.content)
                response = LLMResponse(
                    content=clean_content,
                    model=response.model,
                    tokens_used=response.tokens_used,
                    partial=response.partial,
                    tool_calls=response.tool_calls,
                    gemini_parts=response.gemini_parts,
                )

            # Use exclusively native or text-based tool calls based on user-provided mode
            native_tool_call_objects: list[NativeToolCall] | None = None
            if use_native_tools:
                native_tool_call_objects = response.tool_calls or []
                tool_calls = [
                    ParsedToolCall(tool_name=tc.tool_name, args=tc.args)
                    for tc in native_tool_call_objects
                ]
            else:
                tool_calls = registry.parse_tool_calls(response.content)

            if not tool_calls:
                # Check if the response is a truncated JSON tool call (hit max_tokens)
                if not use_native_tools and _looks_like_truncated_json(response.content):
                    emit("warn", {"message": "truncated JSON tool call detected — asking LLM to retry"})
                    messages.append(LLMMessage(role="assistant", content=response.content, thinking=_thinking_content))
                    messages.append(LLMMessage(
                        role="user",
                        content=(
                            "Your previous response was cut off mid-JSON. "
                            "Please resend the complete tool call as valid JSON."
                        ),
                    ))
                    continue

                # ── Advisor: final check before declaring done ──
                if advisor_llm and _advisor_call_count > 0 and _advisor_call_count < advisor_max_calls and state.tool_calls > 0:
                    messages.append(LLMMessage(role="assistant", content=response.content, thinking=_thinking_content))
                    await _consult_advisor(messages, "final-check")
                    # If advisor injected guidance, let the LLM process it
                    last_msg = messages[-1] if messages else None
                    if last_msg and isinstance(last_msg.content, str) and last_msg.content.startswith("[Advisor Guidance]"):
                        continue

                if recorder:
                    recorder.record_turn(
                        response_text=response.content or "",
                        model=response.model,
                        tokens_used=response.tokens_used,
                        thinking=_thinking_content,
                    )
                state.result = _sanitize_assistant_text(response.content)
                state.status = "done"
                state.iterations += 1
                emit("final_content", {"content": state.result})
                messages.append(LLMMessage(role="assistant", content=response.content, thinking=_thinking_content))
                break

            # ── Handoff dispatch (v6.4) ──────────────────────────────
            # If the LLM called a synthetic handoff tool, transfer control
            # to the target agent and return its terminal state. We honour
            # only the first handoff call in a batch — multiple handoffs
            # in one turn don't make sense (a transfer is exclusive).
            if handoff_map:
                _handoff_call: ParsedToolCall | None = None
                _handoff_native_tc: NativeToolCall | None = None
                for _i, _tc in enumerate(tool_calls):
                    if _tc.tool_name in handoff_map:
                        _handoff_call = _tc
                        if native_tool_call_objects and _i < len(native_tool_call_objects):
                            _handoff_native_tc = native_tool_call_objects[_i]
                        break
                if _handoff_call is not None:
                    h_obj = handoff_map[_handoff_call.tool_name]
                    reason_text = str(_handoff_call.args.get("reason", "")) if isinstance(_handoff_call.args, dict) else ""
                    # Materialise the target agent now so a Handoff(factory=)
                    # constructed before the import cycle was broken can
                    # still resolve.
                    try:
                        target_agent = h_obj.resolve_target()
                    except Exception as resolve_err:
                        emit("warn", {"message": f"handoff target resolution failed: {resolve_err}"})
                        messages.append(LLMMessage(
                            role="user",
                            content=f"[Handoff Error] Could not resolve target agent: {resolve_err}",
                        ))
                        state.iterations += 1
                        continue

                    target_name = getattr(target_agent, "name", None) or _handoff_call.tool_name
                    from_name = agent_name or "ClawAgent"

                    # Stamp the assistant message that triggered the handoff
                    # so the input filter sees a complete transcript.
                    if use_native_tools and _handoff_native_tc and _handoff_native_tc.tool_call_id:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=response.content or "",
                            tool_calls_meta=[{
                                "id": _handoff_native_tc.tool_call_id,
                                "name": _handoff_call.tool_name,
                                "args": _handoff_call.args,
                            }],
                            thinking=_thinking_content,
                        ))
                        # Synthesize a tool-result message acknowledging the
                        # transfer, since most providers reject orphan tool
                        # calls (the rule that drives _patch_dangling_tool_calls).
                        messages.append(LLMMessage(
                            role="tool",
                            content=f"[Handoff] transferred to {target_name}",
                            tool_call_id=_handoff_native_tc.tool_call_id,
                        ))
                    else:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=f'{{"tool": "{_handoff_call.tool_name}", "args": {json.dumps(_handoff_call.args)}}}',
                            thinking=_thinking_content,
                        ))
                        messages.append(LLMMessage(
                            role="user",
                            content=f"[Handoff] transferred to {target_name}",
                        ))

                    # Build the input filter payload from the messages
                    # accumulated so far (input_history). The pre/new split
                    # is approximate: we treat anything past the original
                    # user task as new_items.
                    handoff_payload = HandoffInputData(
                        input_history=list(messages),
                        pre_handoff_items=list(messages[:_session_start_cursor]),
                        new_items=list(messages[_session_start_cursor:]),
                        run_context=run_context,
                    )
                    if h_obj.input_filter is not None:
                        try:
                            handoff_payload = h_obj.input_filter(handoff_payload)
                        except Exception as filter_err:
                            emit("warn", {"message": f"handoff input_filter raised: {filter_err}"})
                    filtered_messages = list(handoff_payload.input_history)

                    # Fire the on_handoff side-effect (per-Handoff) before
                    # firing class-based RunHooks.on_handoff. Both are
                    # observation-only — exceptions are logged.
                    if h_obj.on_handoff is not None:
                        try:
                            await h_obj.on_handoff(run_context)
                        except Exception as hk_err:
                            emit("warn", {"message": f"handoff on_handoff raised: {hk_err}"})
                    if active_hooks:
                        await _fire_hook("on_handoff", from_name, target_name)

                    # Emit the typed event + warn line so callers tracking
                    # `on_event` see the transfer too.
                    emit("warn", {"message": f"handoff: {from_name} → {target_name}"})
                    _emit_typed("handoff_occurred", {
                        "from_agent": from_name,
                        "to_agent": target_name,
                        "tool_name": _handoff_call.tool_name,
                        "reason": reason_text,
                    })

                    # Re-enter the loop on the target agent inside a
                    # ``handoff_span`` so traces capture the transfer.
                    with handoff_span(_handoff_call.tool_name, from_agent=from_name, to_agent=target_name):
                        # Build a fresh task string. We forward the most
                        # recent user message from the filtered history if
                        # one exists; otherwise fall back to the original.
                        last_user = next(
                            (m for m in reversed(filtered_messages)
                             if m.role == "user" and isinstance(m.content, str)),
                            None,
                        )
                        if last_user is not None and isinstance(last_user.content, str):
                            forward_task = last_user.content
                        else:
                            forward_task = task

                        # Drop the system message — the target agent has
                        # its own. Pass remaining filtered history (if any)
                        # via session-protocol-style preload by appending
                        # to messages after the loop's own system prompt
                        # is constructed; the simplest path is to re-call
                        # invoke() with the filtered task + pass any prior
                        # non-system messages via a transient session.
                        non_system = [
                            m for m in filtered_messages if m.role != "system"
                        ]

                        class _TransientSession:
                            def __init__(self, items: list[LLMMessage]):
                                self._items = items

                            async def get_items(self) -> list[LLMMessage]:
                                return list(self._items)

                            async def add_items(self, _new: list[LLMMessage]) -> None:
                                return None

                        # Don't preload the user message we're about to
                        # send as ``task`` — drop the trailing user msg.
                        preload = list(non_system)
                        if preload and preload[-1].role == "user" and isinstance(preload[-1].content, str) and preload[-1].content == forward_task:
                            preload = preload[:-1]

                        try:
                            child_state = await target_agent.invoke(
                                forward_task,
                                run_context=run_context,
                                session=_TransientSession(preload) if preload else None,
                                on_stream_event=on_stream_event,
                            )
                        except Exception as run_err:
                            emit("warn", {"message": f"handoff target raised: {run_err}"})
                            messages.append(LLMMessage(
                                role="user",
                                content=f"[Handoff Error] Target agent failed: {run_err}",
                            ))
                            state.iterations += 1
                            continue

                    state.result = child_state.result
                    state.status = child_state.status if child_state.status != "running" else "done"
                    state.final_output = (
                        child_state.final_output
                        if child_state.final_output is not None
                        else child_state.result
                    )
                    state.tool_calls += child_state.tool_calls
                    state.iterations += 1
                    state.messages = messages + child_state.messages
                    break

            if loop_tracker.is_circuit_broken():
                emit("warn", {"message": f"circuit breaker tripped ({loop_tracker._no_progress_count} no-progress calls) — breaking"})
                state.status = "done"
                state.result = "Circuit breaker: too many tool calls with no progress. Stopping."
                state.iterations += 1
                break

            if loop_tracker.is_hard_looping_batch(tool_calls):
                names = ", ".join(c.tool_name for c in tool_calls)
                emit("warn", {"message": f"tool loop detected ({names}) — breaking"})
                state.status = "done"
                state.result = f"Tool loop detected ({names}). Stopping."
                state.iterations += 1
                break

            if loop_tracker.is_ping_ponging():
                recent_unique = list(set(loop_tracker._history[-6:]))
                emit("warn", {"message": f"ping-pong oscillation detected ({' ↔ '.join(recent_unique)}) — breaking"})
                state.status = "done"
                state.result = "Ping-pong loop detected between tools. Stopping."
                state.iterations += 1
                break

            if loop_tracker.is_soft_looping_batch(tool_calls):
                loop_tracker.record_batch(tool_calls)
                n = loop_tracker.bump_soft_warning()
                repeated_calls = [
                    c for c in tool_calls
                    if loop_tracker.is_soft_looping(c.tool_name, c.args)
                ]
                repeated_names = ", ".join(c.tool_name for c in repeated_calls)
                has_repeated_execute = any(c.tool_name == "execute" for c in repeated_calls)
                emit("warn", {"message": f"repeated tool call warning #{n}: {repeated_names}"})
                if has_repeated_execute:
                    hint = (
                        "[System] You are re-calling the same execute command with the same arguments. "
                        "The command already ran; if the previous result has success=false or a nonzero "
                        "exit_code, treat stdout/stderr as diagnostic feedback, not as a tool failure. "
                        "Read the prior output, then edit code or inspect new evidence before trying again. "
                        "Do not rerun this command until something relevant changed. "
                        "If you believe the task is complete, provide your final answer now."
                    )
                else:
                    hint = (
                        f"[System] You are re-calling {repeated_names} with the same arguments. "
                        "You already have the result in the conversation above. "
                        "Use the existing data instead of re-reading. "
                        "If you believe the task is complete, provide your final answer now."
                    )
                messages.append(LLMMessage(
                    role="user",
                    content=hint,
                ))
                state.iterations += 1
                continue

            # Session: write assistant message with tool calls
            if session_writer:
                tc_meta = []
                if native_tool_call_objects:
                    tc_meta = [{"id": tc.tool_call_id, "name": tc.tool_name, "args": tc.args} for tc in native_tool_call_objects]
                session_writer.write_assistant_message(
                    response.content or "",
                    tool_calls=tc_meta or None,
                    thinking=_thinking_content,
                )

            if len(tool_calls) == 1:
                call = tool_calls[0]
                native_tc = native_tool_call_objects[0] if native_tool_call_objects else None
                emit("tool_call", {"name": call.tool_name})

                # External pre_tool_use hook
                if ext_hook_runner:
                    try:
                        ext_allowed, ext_args = await ext_hook_runner.pre_tool_use(call.tool_name, call.args)
                        if not ext_allowed:
                            emit("tool_skipped", {"name": call.tool_name, "reason": "blocked by external hook"})
                            messages.append(LLMMessage(role="user", content=f"[Tool Skipped] {call.tool_name} was blocked by external hook."))
                            continue
                        call = ParsedToolCall(tool_name=call.tool_name, args=ext_args)
                    except Exception as hook_err:
                        emit("warn", {"message": f"external pre_tool_use hook error: {hook_err}"})

                if before_tool:
                    hook_approved = True
                    hook_reason = "rejected by before_tool hook"
                    try:
                        hook_raw = before_tool(call.tool_name, call.args)
                        if isinstance(hook_raw, HookResult):
                            hook_approved = hook_raw.allowed
                            if hook_raw.reason:
                                hook_reason = hook_raw.reason
                            if hook_raw.allowed and hook_raw.updated_args is not None:
                                call = ParsedToolCall(tool_name=call.tool_name, args=hook_raw.updated_args)
                            if hook_raw.messages:
                                messages.extend(hook_raw.messages)
                        else:
                            hook_approved = bool(hook_raw)
                    except Exception as hook_err:
                        emit("warn", {"message": f"before_tool hook error: {hook_err}"})
                        hook_approved = False
                    if not hook_approved:
                        emit("tool_skipped", {"name": call.tool_name, "reason": hook_reason})
                        messages.append(LLMMessage(role="user", content=f"[Tool Skipped] {call.tool_name} was not approved: {hook_reason}"))
                        continue

                loop_tracker.record(call.tool_name, call.args)

                # HITL tool approval (via RunContext). ``None`` means undecided,
                # which we treat as approve-by-default for backward compatibility.
                native_tc_id = native_tc.tool_call_id if native_tc else call.tool_name
                approval_state = run_context.is_tool_approved(
                    native_tc_id, tool_name=call.tool_name,
                )
                if approval_state is False:
                    rec = run_context.get_approval(native_tc_id, tool_name=call.tool_name)
                    reason = (rec.reason if rec else None) or "rejected via RunContext"
                    emit("tool_skipped", {"name": call.tool_name, "reason": reason})
                    messages.append(LLMMessage(
                        role="user",
                        content=f"[Tool Skipped] {call.tool_name} was rejected: {reason}",
                    ))
                    continue
                if approval_state is None:
                    emit("approval_required", {"name": call.tool_name, "id": native_tc_id})
                    _emit_typed("approval_required", {
                        "tool_name": call.tool_name,
                        "call_id": native_tc_id,
                        "args": call.args,
                    })

                _emit_typed("tool_started", {
                    "tool_name": call.tool_name,
                    "call_id": native_tc_id,
                    "args": call.args,
                })
                if active_hooks:
                    await _fire_hook(
                        "on_tool_start", call.tool_name, native_tc_id, call.args,
                    )
                # ── Activity heartbeats (Hermes parity) ─────────────────
                # Long-running tools (slow web fetches, deep bash runs)
                # would otherwise produce zero events between start and
                # finish; upstream proxies and chat-platform gateways
                # interpret that as "idle" and kill the connection.
                # Emit a periodic ``tool_heartbeat`` while the call is
                # in flight so listeners can keep the channel alive and
                # surface progress.
                tool_result = await run_with_heartbeat(
                    registry.execute_tool(
                        call.tool_name, call.args, run_context=run_context,
                    ),
                    on_event=on_event,
                    kind="tool_heartbeat",
                    payload={
                        "tool_name": call.tool_name,
                        "call_id": native_tc_id,
                    },
                    interval=DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S,
                )
                state.tool_calls += 1
                if active_hooks:
                    await _fire_hook(
                        "on_tool_end",
                        call.tool_name,
                        native_tc_id,
                        tool_result.success,
                        str(tool_result.output)[:2000] if tool_result.output else "",
                        tool_result.error if not tool_result.success else None,
                    )

                # External post_tool_use hook
                if ext_hook_runner:
                    try:
                        ext_result = await ext_hook_runner.post_tool_use(
                            call.tool_name, call.args,
                            {"success": tool_result.success, "output": str(tool_result.output)[:1000]},
                        )
                        if "success" in ext_result and "output" in ext_result:
                            tool_result = ToolResult(
                                success=ext_result["success"],
                                output=ext_result["output"],
                                error=ext_result.get("error"),
                            )
                    except Exception as hook_err:
                        emit("warn", {"message": f"external post_tool_use hook error: {hook_err}"})

                if after_tool:
                    try:
                        hooked_result = after_tool(call.tool_name, call.args, tool_result)
                        if hasattr(hooked_result, "success") and hasattr(hooked_result, "output"):
                            tool_result = hooked_result
                        else:
                            emit("warn", {"message": "after_tool returned invalid ToolResult — ignored"})
                    except Exception as hook_err:
                        emit("warn", {"message": f"after_tool hook error: {hook_err}"})

                raw_output: str | list[dict[str, Any]] = _tool_observation(tool_result)
                tool_output: str | list[dict[str, Any]]
                if isinstance(raw_output, list):
                    tool_output = raw_output
                    preview: str = "[Multimodal Array Content]"
                else:
                    tool_output = _evict_large_tool_result(call.tool_name, raw_output)
                    preview = tool_output[:preview_chars]

                emit("tool_result", {
                    "name": call.tool_name,
                    "success": tool_result.success,
                    "preview": preview,
                })
                _emit_typed("tool_result", {
                    "tool_name": call.tool_name,
                    "call_id": native_tc_id,
                    "success": tool_result.success,
                    "output": preview if isinstance(preview, str) else "[multimodal]",
                    "error": tool_result.error if not tool_result.success else None,
                })

                # Session: write tool result
                if session_writer:
                    tc_id = native_tc.tool_call_id if native_tc else ""
                    session_writer.write_tool_result(
                        tc_id, call.tool_name, tool_result.success,
                        str(tool_result.output)[:2000],
                        error=tool_result.error if not tool_result.success else None,
                    )

                # Record result hash for no-progress / circuit breaker detection
                if isinstance(tool_output, str):
                    loop_tracker.record_result(call.tool_name, call.args, tool_output)

                # ── Failure tracking + trajectory ──
                if failure_tracker:
                    failure_tracker.record(tool_result.success, call.tool_name)
                if recorder:
                    from clawagents.trajectory.recorder import ToolCallRecord
                    # Feature 4: capture observation context (last tool result the agent saw)
                    obs_ctx = ""
                    for m in reversed(messages):
                        if m.role in ("user", "tool") and m.content and isinstance(m.content, str) and m.content.startswith("[Tool Result]"):
                            obs_ctx = m.content[:300]
                            break
                    recorder.record_turn(
                        response_text=response.content or "",
                        model=response.model,
                        tokens_used=response.tokens_used,
                        tool_calls=[ToolCallRecord(
                            tool_name=call.tool_name,
                            args=call.args,
                            success=tool_result.success,
                            output_preview=preview if isinstance(preview, str) else "[multimodal]",
                            error=tool_result.error if not tool_result.success else None,
                        )],
                        observation_context=obs_ctx,
                        thinking=_thinking_content,
                    )

                # Use proper tool role messages when native tools are enabled
                if use_native_tools and native_tc and native_tc.tool_call_id:
                    messages.append(LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls_meta=[{"id": native_tc.tool_call_id, "name": call.tool_name, "args": call.args}],
                        gemini_parts=getattr(response, "gemini_parts", None),
                        thinking=_thinking_content,
                    ))
                    tool_content = f"{tool_output}" if isinstance(tool_output, str) else json.dumps(tool_output)
                    messages.append(LLMMessage(
                        role="tool",
                        content=tool_content,
                        tool_call_id=native_tc.tool_call_id,
                    ))
                else:
                    messages.append(
                        LLMMessage(role="assistant", content=f'{{"tool": "{call.tool_name}", "args": {json.dumps(call.args)}}}', thinking=_thinking_content)
                    )
                    user_content: str | list[dict[str, Any]]
                    if isinstance(tool_output, str):
                        user_content = f"[Tool Result] {tool_output}"
                    else:
                        user_content = tool_output
                    messages.append(
                        LLMMessage(role="user", content=user_content)
                    )

                # ── Rethink injection on consecutive failures ──
                if failure_tracker:
                    # Feature F: update threshold dynamically based on progress
                    try:
                        from clawagents.trajectory.verifier import compute_adaptive_rethink_threshold
                        failure_tracker._threshold = compute_adaptive_rethink_threshold(
                            _task_type, round_idx, state.tool_calls
                        )
                    except Exception:
                        pass
                    if failure_tracker.should_rethink():
                        # ── Advisor: consult when stuck ──
                        await _consult_advisor(messages, "stuck")
                        n = failure_tracker.consecutive_failures
                        rethink_num = failure_tracker.bump_rethink()
                        emit("warn", {"message": f"rethink #{rethink_num}: {n} consecutive failures (threshold={failure_tracker._threshold})"})
                        rethink_msg = _RETHINK_MESSAGE.format(n=n)
                        if learn:
                            from clawagents.trajectory.lessons import build_rethink_with_lessons
                            fmt_count = sum(1 for t in (recorder.turns if recorder else []) for tc in t.tool_calls if not tc.success and tc.failure_type == "format")
                            logic_count = sum(1 for t in (recorder.turns if recorder else []) for tc in t.tool_calls if not tc.success and tc.failure_type == "logic")
                            rethink_msg = build_rethink_with_lessons(rethink_msg, fmt_count, logic_count)
                        messages.append(LLMMessage(
                            role="user",
                            content=rethink_msg,
                        ))

            else:
                # ── before_tool hook (parallel) — filter out rejected calls ──
                # Track original tool_calls index alongside each approved call so
                # native_tool_call_objects[orig_idx] stays correct even when the hook
                # rejects calls (skipping reduces approved length) or modifies args
                # (which produces a new ParsedToolCall instance, breaking identity checks).
                approved_calls: list[ParsedToolCall] = []
                _approved_orig_indices: list[int] = []
                if before_tool:
                    def _apply_hook(c):
                        """Return (approved_call_or_None, reason) after running the hook."""
                        try:
                            hook_raw = before_tool(c.tool_name, c.args)
                            if isinstance(hook_raw, HookResult):
                                if not hook_raw.allowed:
                                    return None, hook_raw.reason or "rejected by before_tool hook"
                                if hook_raw.messages:
                                    messages.extend(hook_raw.messages)
                                if hook_raw.updated_args is not None:
                                    c = ParsedToolCall(tool_name=c.tool_name, args=hook_raw.updated_args)
                                return c, ""
                            else:
                                if not bool(hook_raw):
                                    return None, "rejected by before_tool hook"
                                return c, ""
                        except Exception as hook_err:
                            emit("warn", {"message": f"before_tool hook error: {hook_err}"})
                            return None, "hook error"

                    for _orig_i, c in enumerate(tool_calls):
                        result_call, reason = _apply_hook(c)
                        if result_call is None:
                            emit("tool_skipped", {"name": c.tool_name, "reason": reason})
                        else:
                            approved_calls.append(result_call)
                            _approved_orig_indices.append(_orig_i)
                    if not approved_calls:
                        messages.append(LLMMessage(role="user", content="[Tool Skipped] All tool calls were not approved."))
                        continue
                else:
                    approved_calls = list(tool_calls)
                    _approved_orig_indices = list(range(len(tool_calls)))

                # Resolve a stable call_id per approved call (prefer native tc id).
                # Index native_tool_call_objects by ORIGINAL tool_calls index, not
                # approved_calls index — those diverge when before_tool rejects a call.
                _approved_call_ids: list[str] = []
                for _idx, _c in enumerate(approved_calls):
                    _orig_idx = _approved_orig_indices[_idx]
                    _ntc = native_tool_call_objects[_orig_idx] if (
                        native_tool_call_objects
                        and _orig_idx < len(native_tool_call_objects)
                    ) else None
                    _approved_call_ids.append(
                        (_ntc.tool_call_id if _ntc else None) or _c.tool_name
                    )

                _runnable_calls: list[ParsedToolCall] = []
                _runnable_call_ids: list[str] = []
                for _c, _cid in zip(approved_calls, _approved_call_ids):
                    approval_state = run_context.is_tool_approved(_cid, tool_name=_c.tool_name)
                    if approval_state is False:
                        rec = run_context.get_approval(_cid, tool_name=_c.tool_name)
                        reason = (rec.reason if rec else None) or "rejected via RunContext"
                        emit("tool_skipped", {"name": _c.tool_name, "reason": reason})
                        messages.append(LLMMessage(
                            role="user",
                            content=f"[Tool Skipped] {_c.tool_name} was rejected: {reason}",
                        ))
                        continue
                    if approval_state is None:
                        emit("approval_required", {"name": _c.tool_name, "id": _cid})
                        _emit_typed("approval_required", {
                            "tool_name": _c.tool_name,
                            "call_id": _cid,
                            "args": _c.args,
                        })
                    _runnable_calls.append(_c)
                    _runnable_call_ids.append(_cid)
                approved_calls = _runnable_calls
                _approved_call_ids = _runnable_call_ids
                if not approved_calls:
                    continue

                for call in approved_calls:
                    emit("tool_call", {"name": call.tool_name})
                loop_tracker.record_batch(approved_calls)

                for _c, _cid in zip(approved_calls, _approved_call_ids):
                    _emit_typed("tool_started", {
                        "tool_name": _c.tool_name,
                        "call_id": _cid,
                        "args": _c.args,
                    })
                    if active_hooks:
                        await _fire_hook(
                            "on_tool_start", _c.tool_name, _cid, _c.args,
                        )
                # Heartbeat while the parallel batch runs; the call_ids
                # field lets listeners disambiguate which group of tools
                # is in flight.
                results = await run_with_heartbeat(
                    registry.execute_tools_parallel(
                        approved_calls, run_context=run_context,
                    ),
                    on_event=on_event,
                    kind="tool_heartbeat",
                    payload={
                        "parallel": True,
                        "tool_names": [_c.tool_name for _c in approved_calls],
                        "call_ids": list(_approved_call_ids),
                    },
                    interval=DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S,
                )
                state.tool_calls += len(approved_calls)
                if active_hooks:
                    for _c, _cid, _r in zip(
                        approved_calls, _approved_call_ids, results
                    ):
                        await _fire_hook(
                            "on_tool_end",
                            _c.tool_name,
                            _cid,
                            _r.success,
                            str(_r.output)[:2000] if _r.output else "",
                            _r.error if not _r.success else None,
                        )

                if after_tool:
                    safe_results: list[ToolResult] = []
                    for c, r in zip(approved_calls, results):
                        try:
                            hooked_parallel = after_tool(c.tool_name, c.args, r)
                            if hasattr(hooked_parallel, "success") and hasattr(hooked_parallel, "output"):
                                safe_results.append(hooked_parallel)
                            else:
                                emit("warn", {"message": "after_tool returned invalid ToolResult — ignored"})
                                safe_results.append(r)
                        except Exception as hook_err:
                            emit("warn", {"message": f"after_tool hook error: {hook_err}"})
                            safe_results.append(r)
                    results = safe_results

                # Build a map from approved-list index to NativeToolCall for ID lookup.
                # Use the orig-index list captured during approval — identity-checking
                # `tc is approved_calls[i]` fails when before_tool returns updated_args
                # (which constructs a new ParsedToolCall).
                native_tc_map: dict[int, NativeToolCall] = {}
                if native_tool_call_objects:
                    for _idx, _orig_idx in enumerate(_approved_orig_indices):
                        if _orig_idx < len(native_tool_call_objects):
                            native_tc_map[_idx] = native_tool_call_objects[_orig_idx]

                call_summaries: list[str] = []
                tool_outputs: list[str] = []
                for _idx2, (call, result) in enumerate(zip(approved_calls, results)):
                    raw_out: str | list[dict[str, Any]] = _tool_observation(result)
                    output: str | list[dict[str, Any]]
                    if isinstance(raw_out, list):
                        output = raw_out
                        preview = "[Multimodal Array Content]"
                    else:
                        output = _evict_large_tool_result(call.tool_name, raw_out)
                        preview = output[:preview_chars]

                    _call_id = (
                        _approved_call_ids[_idx2]
                        if _idx2 < len(_approved_call_ids)
                        else call.tool_name
                    )
                    emit("tool_result", {
                        "name": call.tool_name,
                        "success": result.success,
                        "preview": preview,
                    })
                    _emit_typed("tool_result", {
                        "tool_name": call.tool_name,
                        "call_id": _call_id,
                        "success": result.success,
                        "output": preview if isinstance(preview, str) else "[multimodal]",
                        "error": result.error if not result.success else None,
                    })
                    
                    if isinstance(output, str):
                        call_summaries.append(f"{call.tool_name}({json.dumps(call.args)}) => {output}")
                        tool_outputs.append(output)
                    else:
                        call_summaries.append(f"{call.tool_name}({json.dumps(call.args)}) => [Multimodal Output Length: {len(output)}]")
                        call_summaries.append(json.dumps(output))
                        tool_outputs.append(json.dumps(output))

                # Record result hashes for no-progress / circuit breaker detection
                for c_rec, out_rec in zip(approved_calls, tool_outputs):
                    if isinstance(out_rec, str):
                        loop_tracker.record_result(c_rec.tool_name, c_rec.args, out_rec)

                # ── Failure tracking + trajectory (parallel) ──
                if failure_tracker:
                    failure_tracker.record_batch([
                        (r.success, c.tool_name) for c, r in zip(approved_calls, results)
                    ])
                if recorder:
                    from clawagents.trajectory.recorder import ToolCallRecord
                    tc_records = []
                    for call, result in zip(approved_calls, results):
                        if not result.success:
                            raw_p = (result.error or "")[:preview_chars]
                        elif isinstance(result.output, str):
                            raw_p = result.output[:preview_chars]
                        else:
                            # Multimodal output (list of content blocks) — store a marker
                            # because ToolCallRecord.output_preview expects a str.
                            raw_p = "[multimodal]"
                        tc_records.append(ToolCallRecord(
                            tool_name=call.tool_name,
                            args=call.args,
                            success=result.success,
                            output_preview=raw_p,
                            error=result.error if not result.success else None,
                        ))
                    # Feature 4: capture observation context
                    obs_ctx = ""
                    for m in reversed(messages):
                        if m.role in ("user", "tool") and m.content and isinstance(m.content, str) and m.content.startswith("[Tool Result"):
                            obs_ctx = m.content[:300]
                            break
                    recorder.record_turn(
                        response_text=response.content or "",
                        model=response.model,
                        tokens_used=response.tokens_used,
                        tool_calls=tc_records,
                        observation_context=obs_ctx,
                        thinking=_thinking_content,
                    )

                # Use proper tool role messages when native tools are enabled
                if use_native_tools and native_tc_map:
                    tc_meta = []
                    for idx, call in enumerate(approved_calls):
                        ntc = native_tc_map.get(idx)
                        tc_id = ntc.tool_call_id if ntc else f"fallback_{idx}"
                        tc_meta.append({"id": tc_id, "name": call.tool_name, "args": call.args})
                    
                    messages.append(LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls_meta=tc_meta,
                        gemini_parts=getattr(response, "gemini_parts", None),
                        thinking=_thinking_content,
                    ))
                    for idx, (call, output_str) in enumerate(zip(approved_calls, tool_outputs)):
                        ntc = native_tc_map.get(idx)
                        tc_id = ntc.tool_call_id if ntc else f"fallback_{idx}"
                        messages.append(LLMMessage(
                            role="tool",
                            content=output_str,
                            tool_call_id=tc_id,
                        ))
                else:
                    tool_call_str = json.dumps([
                        {"tool": c.tool_name, "args": c.args} for c in approved_calls
                    ])
                    messages.append(
                        LLMMessage(role="assistant", content=tool_call_str, thinking=_thinking_content)
                    )
                    messages.append(
                        LLMMessage(
                            role="user",
                            content="[Tool Results]\n" + "\n".join(call_summaries),
                        )
                    )

                # ── Rethink injection on consecutive failures (parallel) ──
                if failure_tracker:
                    # Feature F: update threshold dynamically
                    try:
                        from clawagents.trajectory.verifier import compute_adaptive_rethink_threshold
                        failure_tracker._threshold = compute_adaptive_rethink_threshold(
                            _task_type, round_idx, state.tool_calls
                        )
                    except Exception:
                        pass
                    if failure_tracker.should_rethink():
                        # ── Advisor: consult when stuck ──
                        await _consult_advisor(messages, "stuck")
                        n = failure_tracker.consecutive_failures
                        rethink_num = failure_tracker.bump_rethink()
                        emit("warn", {"message": f"rethink #{rethink_num}: {n} consecutive failures (threshold={failure_tracker._threshold})"})
                        rethink_msg = _RETHINK_MESSAGE.format(n=n)
                        if learn:
                            from clawagents.trajectory.lessons import build_rethink_with_lessons
                            fmt_count = sum(1 for t in (recorder.turns if recorder else []) for tc in t.tool_calls if not tc.success and tc.failure_type == "format")
                            logic_count = sum(1 for t in (recorder.turns if recorder else []) for tc in t.tool_calls if not tc.success and tc.failure_type == "logic")
                            rethink_msg = build_rethink_with_lessons(rethink_msg, fmt_count, logic_count)
                        messages.append(LLMMessage(
                            role="user",
                            content=rethink_msg,
                        ))

                # ── Continuous background memory extraction (Claude Code pattern) ──
                # Skipped for isolated subagents (skip_memory=True): they do
                # not write back to the parent's memory store.
                if (
                    learn
                    and recorder
                    and not getattr(run_context, "skip_memory", False)
                ):
                    try:
                        from clawagents.trajectory.background_memory import maybe_extract_memories
                        _last_memory_extraction_turn = await maybe_extract_memories(
                            llm, messages, round_idx, _last_memory_extraction_turn,
                        )
                    except Exception:
                        pass  # Background extraction failure should never block the loop

        else:
            emit("warn", {"message": f"reached max {effective_max_rounds} tool rounds"})
            state.status = "done"
            state.result = state.result or f"Reached maximum of {effective_max_rounds} tool rounds."
            state.iterations += 1

    except KeyboardInterrupt:
        emit("warn", {"message": "interrupted"})
        state.status = "done"
        state.result = state.result or "[interrupted]"
    except asyncio.CancelledError:
        emit("warn", {"message": "cancelled"})
        state.status = "done"
        state.result = state.result or "[cancelled]"
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, OSError):
            pass

    elapsed = time.monotonic() - t0
    state.messages = messages

    # Session: write final turn_completed
    if session_writer:
        session_writer.write_turn_completed(
            state.iterations, state.tool_calls, state.status,
        )
        state.session_file = str(session_writer.path)

    # ── Finalize trajectory ──
    run_summary = None
    if recorder:
        outcome = state.status if state.status != "running" else "success"
        run_summary = recorder.finalize(outcome)
        state.trajectory_file = run_summary.trajectory_file
        emit("context", {"message": f"trajectory saved to {run_summary.trajectory_file}"})

    # ── Feature G: LLM-as-Judge verification ──
    if learn and recorder and run_summary:
        try:
            from dataclasses import asdict
            from clawagents.trajectory.judge import judge_run
            summary_dict = asdict(run_summary)
            turn_dicts = [asdict(t) for t in recorder.turns]
            judge_result = await judge_run(
                llm, task, summary_dict, state.result, turn_dicts,
            )
            run_summary.judge_score = judge_result.get("judge_score")
            run_summary.judge_justification = judge_result.get("judge_justification", "")
            emit("context", {
                "message": f"LLM Judge: score={run_summary.judge_score}/3 — {run_summary.judge_justification[:80]}"
            })
        except Exception:
            logger.debug("LLM-as-Judge failed", exc_info=True)

    # ── PTRL Layer 3: Post-run self-analysis (with quality gate) ──
    # Skipped for isolated subagents (skip_memory=True): subagents must not
    # write lessons back into the parent's lesson store.
    if (
        learn
        and recorder
        and run_summary
        and not getattr(run_context, "skip_memory", False)
    ):
        try:
            from dataclasses import asdict
            from clawagents.trajectory.lessons import extract_lessons, save_lessons, should_extract_lessons
            summary_dict = asdict(run_summary)

            # Feature 1: Quality gate — only extract lessons from informative runs
            if should_extract_lessons(summary_dict):
                turn_dicts = [asdict(t) for t in recorder.turns]
                lessons_text = await extract_lessons(llm, summary_dict, turn_dicts)
                if lessons_text:
                    save_lessons(
                        lessons_text, run_summary.task, run_summary.outcome,
                        model=run_summary.model,
                    )
                    emit("context", {"message": "PTRL: extracted and saved lessons from this run"})
            else:
                emit("context", {
                    "message": f"PTRL: skipped lesson extraction (quality={run_summary.quality}, "
                    f"mixed={run_summary.has_mixed_outcomes}, score={run_summary.run_score})"
                })
        except Exception:
            logger.debug("PTRL: post-run self-analysis failed", exc_info=True)

    # ── Output guardrails + structured output coercion ──
    if output_guardrails and state.result:
        try:
            rewritten, tripped = await _run_output_guardrails(
                output_guardrails, run_context, state.result,
            )
            if tripped:
                state.guardrail_triggered = tripped
                state.result = str(rewritten)
                _emit_typed("guardrail_tripped", {
                    "guardrail_name": tripped,
                    "where": "output",
                    "behavior": GuardrailBehavior.REJECT_CONTENT.value,
                    "message": state.result,
                })
                emit("warn", {"message": f"output guardrail tripped: {tripped}"})
        except GuardrailTripwireTriggered as tripwire:
            state.guardrail_triggered = tripwire.guardrail_name
            _emit_typed("guardrail_tripped", {
                "guardrail_name": tripwire.guardrail_name,
                "where": "output",
                "behavior": tripwire.result.behavior.value,
                "message": tripwire.result.message or "",
            })
            emit("warn", {"message": f"output guardrail raised: {tripwire.guardrail_name}"})

    if output_type is not None and state.status == "done" and state.result:
        try:
            state.final_output = _coerce_output_type(state.result, output_type)
        except Exception as err:
            emit("warn", {"message": f"output_type coercion failed: {err}"})
            state.final_output = state.result
    elif state.status == "done":
        state.final_output = state.result

    # Persist only the messages newly added in this run to the Session backend.
    if session is not None:
        try:
            new_msgs = messages[_session_start_cursor:]
            if new_msgs:
                await _session_add_items(session, new_msgs)
        except Exception as err:
            emit("warn", {"message": f"session save failed: {err}"})

    _emit_typed("final_output", {
        "output": (
            state.final_output
            if state.final_output is not None
            else state.result
        ),
        "raw": state.result if isinstance(state.result, str) else "",
        "usage": usage.to_dict(),
    })
    if active_hooks:
        await _fire_hook("on_run_end", state.result)

    emit("agent_done", {
        "tool_calls": state.tool_calls,
        "iterations": state.iterations,
        "elapsed": elapsed,
        "usage": usage.to_dict(),
    })
    return state
