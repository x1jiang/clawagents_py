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
import hashlib
import inspect
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

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
from clawagents.prompts import append_model_identity, build_system_prompt
from clawagents.tokenizer import (
    count_messages_tokens as _count_messages_tokens,
    count_tokens_content,
)
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
# It also drops orphan role="tool" messages whose tool_call_id was never declared
# by a preceding assistant tool_calls_meta (common after session preload limit
# cuts mid-pair → OpenAI 400: "messages with role 'tool' must be a response to
# a preceding message with 'tool_calls'").

def _patch_dangling_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    if not messages:
        return messages

    # Ids declared by assistant messages in this transcript.
    declared_ids: set[str] = set()
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls_meta:
            for tc in msg.tool_calls_meta:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    declared_ids.add(str(tc_id))

    # Drop orphan tool results first (no matching assistant tool_calls).
    filtered: list[LLMMessage] = []
    for msg in messages:
        if msg.role == "tool":
            tc_id = str(msg.tool_call_id) if msg.tool_call_id else ""
            if not tc_id or tc_id not in declared_ids:
                continue
        filtered.append(msg)

    # Build set of all tool_call_ids that have a matching role="tool" response
    responded_ids: set[str] = set()
    for msg in filtered:
        if msg.role == "tool" and msg.tool_call_id:
            responded_ids.add(str(msg.tool_call_id))

    patched: list[LLMMessage] = []
    for i, msg in enumerate(filtered):
        patched.append(msg)

        # Text-mode: look for assistant messages with JSON tool calls without a following [Tool Result]
        if msg.role == "assistant" and isinstance(msg.content, str) and msg.content.startswith('{"tool":'):
            _next_msg = filtered[i + 1] if i + 1 < len(filtered) else None
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
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id and str(tc_id) not in responded_ids:
                    patched.append(LLMMessage(
                        role="tool",
                        content="Tool call was cancelled — the agent was interrupted before it could complete.",
                        tool_call_id=str(tc_id),
                    ))
                    responded_ids.add(str(tc_id))

    return patched


def _drop_leading_orphan_tools(messages: list[LLMMessage]) -> list[LLMMessage]:
    """If a limited session preload starts mid tool-pair, drop leading orphans."""
    if not messages:
        return messages
    i = 0
    while i < len(messages) and messages[i].role == "tool":
        i += 1
    return messages[i:] if i else messages


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

    head = "\n".join(
        f"{i + 1}: {line}" for i, line in enumerate(lines[:head_lines])
    )
    total = len(lines)
    tail = "\n".join(
        f"{total - tail_lines + i + 1}: {line}"
        for i, line in enumerate(lines[-tail_lines:])
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
    """Observation text for the model — prefers full ``raw_output`` when present."""
    payload = getattr(result, "raw_output", None)
    if payload is None:
        payload = result.output
    if result.success:
        return payload
    error = f"Error: {result.error}" if result.error else "Error: Tool failed"
    if isinstance(payload, list):
        return [{"type": "text", "text": error}, *payload]
    output = str(payload or "").strip()
    return f"{error}\nOutput:\n{output}" if output else error


def _run_context_workspace(run_context: Any) -> str | None:
    """Workspace root from run_context metadata (never trust bare cwd alone)."""
    meta = getattr(run_context, "_metadata", None) if run_context is not None else None
    if isinstance(meta, dict):
        ws = meta.get("workspace")
        if isinstance(ws, str) and ws.strip():
            return ws.strip()
    return None


from clawagents.graph.model_profiles import resolve_context_budget as _resolve_context_budget


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

# Keep _CHARS_PER_TOKEN for the Tier-3 preflight char-budget calculation only
_CHARS_PER_TOKEN = 4


def _estimate_tokens(content: str | list[dict], multiplier: float = 1.0, model: str | None = None) -> int:
    return count_tokens_content(content, model=model, multiplier=multiplier)


def _estimate_messages_tokens(
    messages: list[LLMMessage],
    multiplier: float = 1.0,
    model: str | None = None,
    *,
    cached_system_tokens: int | None = None,
) -> int:
    return _count_messages_tokens(
        messages,
        model=model,
        multiplier=multiplier,
        cached_system_tokens=cached_system_tokens,
    )


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


if TYPE_CHECKING:
    from clawagents.loop_detection import LoopDetectionConfig


class _ToolCallTracker:
    def __init__(
        self,
        window_size: int = 30,
        soft_limit: int = 3,
        hard_limit: int = 6,
        circuit_breaker_limit: int = 30,
        loop_config: "LoopDetectionConfig | None" = None,
    ):
        from clawagents.loop_detection import resolve_loop_detection_config

        self._history: list[str] = []
        self._poll_history: list[tuple[str, str, str | None]] = []
        self._window_size = window_size
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._circuit_breaker_limit = circuit_breaker_limit
        self._loop_config = resolve_loop_detection_config(loop_config)
        self._result_hashes: dict[str, str] = {}
        self._no_progress_count = 0
        self._soft_warnings = 0
        self._poll_warnings: set[str] = set()

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
        from clawagents.loop_detection import hash_tool_call

        key = self._key(tool_name, args)
        result_hash = self._hash_result(output)
        prev_hash = self._result_hashes.get(key)
        if prev_hash == result_hash:
            self._no_progress_count += 1
        else:
            self._no_progress_count = max(0, self._no_progress_count - 1)
        self._result_hashes[key] = result_hash
        call_hash = hash_tool_call(tool_name, args)
        self._poll_history.append((tool_name, call_hash, result_hash))
        if len(self._poll_history) > self._window_size:
            self._poll_history.pop(0)

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

    def check_known_poll_no_progress(self, tool_name: str, args: dict):
        from clawagents.loop_detection import detect_known_poll_no_progress

        result = detect_known_poll_no_progress(
            tool_name=tool_name,
            params=args,
            history=self._poll_history,
            config=self._loop_config,
        )
        if result and result.stuck and result.warning_key in self._poll_warnings:
            if result.level == "warning":
                return None
        if result and result.stuck and result.warning_key:
            self._poll_warnings.add(result.warning_key)
        return result


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
# Only micro-compact once the transcript actually uses a meaningful share of
# the context window. Running it unconditionally blanked all but the last 3
# read/grep/exec results every round, degrading multi-file tasks at low usage.
_MICRO_COMPACT_MIN_USAGE_RATIO = 0.4
_ARTIFACT_ID_RE = re.compile(
    r"(?:Artifact id:\s*|id=)([A-Za-z0-9._-]{4,80})",
    re.IGNORECASE,
)


def _extract_artifact_id(content: str) -> str | None:
    if not content:
        return None
    m = _ARTIFACT_ID_RE.search(content)
    return m.group(1) if m else None


async def _wait_for_tool_approval(
    run_context: RunContext,
    call_id: str,
    tool_name: str,
    args: dict[str, Any],
    *,
    approval_handler: Any,
    emit: OnEvent,
    timeout_s: float = 300.0,
) -> bool:
    """Block until RunContext has an approval decision, or handler returns one.

    ``approval_handler`` may be:
      - ``"event"``: poll ``is_tool_approved`` until decided (host must call
        ``approve_tool`` / ``reject_tool``)
      - callable ``(tool_name, args, call_id) -> bool | Awaitable[bool]``
    """
    if callable(approval_handler):
        try:
            result = approval_handler(tool_name, args, call_id)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[misc]
            return bool(result)
        except Exception as exc:
            emit("warn", {"message": f"approval_handler error: {exc}"})
            return False

    # "event" (or any other truthy non-callable): poll RunContext
    deadline = asyncio.get_event_loop().time() + max(1.0, timeout_s)
    while asyncio.get_event_loop().time() < deadline:
        state = run_context.is_tool_approved(call_id, tool_name=tool_name)
        if state is not None:
            return bool(state)
        await asyncio.sleep(0.05)
    emit("warn", {"message": f"approval timed out for {tool_name}"})
    return False


def _post_tool_side_effects(
    tool_name: str,
    args: dict[str, Any],
    success: bool,
    tool_output: str | list[dict[str, Any]],
    *,
    emit: OnEvent,
    run_context: RunContext | None = None,
) -> str | list[dict[str, Any]]:
    """Ledger / shadow-checkpoint / auto-verify after a tool completes."""
    from clawagents.config.features import is_enabled

    out = tool_output
    try:
        if success:
            path = None
            if isinstance(args, dict):
                path = args.get("path") or args.get("file_path") or args.get("file")
            if path:
                from clawagents.skills.strategy import note_touched_path

                note_touched_path(run_context, str(path))
                store = None
                if run_context is not None and isinstance(run_context._metadata, dict):
                    store = run_context._metadata.get("skill_store")
                if store is not None and hasattr(store, "note_touched_path"):
                    store.note_touched_path(str(path))
    except Exception:
        logger.debug("skill path touch tracking failed", exc_info=True)

    try:
        if success and is_enabled("context_ledger"):
            from clawagents.memory.context_ledger import maybe_record_from_tool_result

            text = out if isinstance(out, str) else str(out)
            entry = maybe_record_from_tool_result(tool_name, args, text)
            if entry is not None:
                emit("context", {"message": f"context ledger recorded {entry.sha[:12]}"})
                emit("checkpoint", {"kind": "ledger", "sha": entry.sha})
    except Exception:
        logger.debug("ledger record failed", exc_info=True)

    try:
        if success and is_enabled("shadow_checkpoints"):
            from clawagents.permissions.mode import is_write_class_tool
            from clawagents.memory.shadow_checkpoint import create_checkpoint

            # Post-success for execute/git_commit (pre-mutation already covers writes).
            if tool_name in {"execute", "git_commit"} or (
                is_write_class_tool(tool_name)
                and tool_name not in {
                    "write_file", "edit_file", "apply_patch", "create_file",
                    "replace_in_file", "insert_in_file", "insert_lines", "patch_file",
                }
            ):
                info = create_checkpoint(label=tool_name, tool=tool_name, phase="post")
                if info.get("ok") and info.get("sha"):
                    emit(
                        "checkpoint",
                        {
                            "kind": "shadow",
                            "sha": info["sha"],
                            "tool": tool_name,
                            "phase": "post",
                            "label": tool_name,
                            "ts": info.get("ts"),
                        },
                    )
                    if isinstance(out, str):
                        out = out + f"\n[checkpoint {info['sha'][:12]}]"
            elif is_write_class_tool(tool_name) or tool_name in {
                "write_file", "edit_file", "apply_patch",
            }:
                # Pre-mutation checkpoint already created in registry; emit latest HEAD for UI.
                from clawagents.memory.shadow_checkpoint import list_checkpoints

                rows = list_checkpoints(limit=1)
                if rows:
                    row = rows[0]
                    emit(
                        "checkpoint",
                        {
                            "kind": "shadow",
                            "sha": row.get("sha"),
                            "tool": tool_name,
                            "phase": "pre",
                            "label": row.get("label") or tool_name,
                            "ts": row.get("ts"),
                        },
                    )
                    if isinstance(out, str) and row.get("sha"):
                        out = out + f"\n[checkpoint {str(row['sha'])[:12]}]"
    except Exception:
        logger.debug("shadow checkpoint failed", exc_info=True)

    try:
        if is_enabled("auto_verify") and isinstance(out, str):
            from clawagents.tools.auto_verify import maybe_verify_after_edit

            extra = maybe_verify_after_edit(tool_name, success)
            if extra:
                out = out + "\n\n" + extra
    except Exception:
        logger.debug("auto_verify failed", exc_info=True)

    return out


def _micro_compact_stub(content: str, *, tool_call_id: str | None = None) -> str:
    """Replace old tool bodies with a stub that still points at a recoverable artifact."""
    aid = _extract_artifact_id(content)
    if aid is None and isinstance(content, str) and len(content) > 500:
        try:
            from clawagents.tool_output_artifacts import store_tool_artifact

            aid, _ = store_tool_artifact(
                tool_name="micro_compact",
                tool_use_id=tool_call_id or f"micro-{abs(hash(content[:200])) % 10_000_000}",
                output=content,
                kind="prose",
                extra_meta={"source": "micro_compact"},
            )
        except Exception:
            logger.debug("micro-compact artifact store failed", exc_info=True)
            aid = None
    if aid:
        return (
            f"[Old tool result cleared to save context — artifact id={aid}. "
            f"Call retrieve_tool_result(id=\"{aid}\") to restore.]"
        )
    return "[Old tool result cleared to save context]"


def _micro_compact_tool_results(
    messages: list[LLMMessage],
    keep_recent: int = _MICRO_COMPACT_KEEP_RECENT,
) -> list[LLMMessage]:
    """Clear old tool result content for compactable tools (keep last N).

    The model still sees the tool_use → tool_result pairs, just not the raw
    50KB grep/file output. This preserves the agent's sense of *what* it did
    while freeing massive amounts of context. Stubs retain artifact ids when
    available so content remains recoverable via retrieve_tool_result.
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
                body = msg.content if isinstance(msg.content, str) else str(msg.content)
                result.append(LLMMessage(
                    role="tool",
                    content=_micro_compact_stub(body, tool_call_id=msg.tool_call_id),
                    tool_call_id=msg.tool_call_id,
                ))
                cleared += 1
                continue
        # Text-based tool results (user message following assistant tool call)
        elif msg.role == "user" and isinstance(msg.content, str) and msg.content.startswith("[Tool Result]"):
            if i > 0 and (i - 1) in compactable_text_indices and (i - 1) not in keep_text_indices:
                stub = _micro_compact_stub(msg.content)
                result.append(LLMMessage(
                    role="user",
                    content=f"[Tool Result] {stub}",
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
    # Bound is < len(non_system), NOT len - 1: with the tighter bound a tail
    # run of ≥N tool messages left the last orphan tool result in `recent`
    # while its paired assistant tool_call got summarized away → provider 400.
    while split < len(non_system):
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


def _content_key_text(content: Any) -> str:
    """Stable text stand-in for message content (compaction input + reuse keys).

    Multimodal list content must not go through ``str()`` — that dumps the
    full base64 data URL (megabytes) into the summarizer prompt. Join the
    real text parts and replace each image with a short digest placeholder;
    the digest keeps reuse keys distinct per distinct image so compaction's
    original-message reuse can't swap two same-text messages.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            parts.append(str(p.get("text", "") or ""))
        elif p.get("type") in ("image_url", "image"):
            digest = hashlib.sha1(
                json.dumps(p, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:8]
            parts.append(f"[image attachment #{digest}]")
        elif p.get("type") in ("file", "document"):
            digest = hashlib.sha1(
                json.dumps(p, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:8]
            parts.append(f"[file attachment #{digest}]")
    return "\n".join(parts)


def _is_compactable_user(msg: LLMMessage) -> bool:
    if msg.role != "user":
        return False
    content = msg.content if isinstance(msg.content, str) else ""
    if content.startswith("[Tool Result]"):
        return False
    if "Compacted History" in content:
        return False
    if content.startswith("This session is being continued"):
        return False
    return True


_FILE_PATH_RE = re.compile(
    r"""(?:write_file|edit_file|apply_patch|read_file)[^\"']*[\"']([^\"']+)[\"']"""
    r"""|path[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']""",
    re.I,
)


def _extract_recent_files(messages: list[LLMMessage], *, limit: int = 12) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in messages:
        blob = ""
        if isinstance(m.content, str):
            blob = m.content
        if getattr(m, "tool_calls_meta", None):
            try:
                blob += " " + json.dumps(m.tool_calls_meta, default=str)
            except TypeError:
                pass
        for match in _FILE_PATH_RE.finditer(blob):
            path = next((g for g in match.groups() if g), None)
            if not path or path in seen:
                continue
            if any(ch in path for ch in ("\n", " ")):
                continue
            seen.add(path)
            found.append(path)
            if len(found) >= limit:
                return found
    return found


def _message_reuse_key(m: LLMMessage) -> tuple[Any, ...]:
    """Disambiguate reuse keys so empty-content tool calls cannot swap meta.

    Matching only on ``(role, content)`` swapped ``tool_calls_meta`` between
    unrelated assistants that both had ``content=None`` / ``""``.
    """
    role = m.role
    content = _content_key_text(m.content)
    if role == "tool":
        return (role, content, str(m.tool_call_id or ""), ())
    meta = getattr(m, "tool_calls_meta", None) or []
    if role == "assistant" and meta:
        ids = tuple(str(tc.get("id") or "") for tc in meta if isinstance(tc, dict))
        names = tuple(str(tc.get("name") or "") for tc in meta if isinstance(tc, dict))
        return (role, content, ids, names)
    return (role, content, "", ())


def _reuse_messages_where_possible(
    originals: list[LLMMessage],
    rebuilt: list[LLMMessage],
) -> list[LLMMessage]:
    """Prefer original LLMMessage object identity when reuse keys match.

    Required for session-persistence trackers that key off identity.
    """
    buckets: dict[tuple[Any, ...], list[LLMMessage]] = {}
    for om in originals:
        buckets.setdefault(_message_reuse_key(om), []).append(om)
    out: list[LLMMessage] = []
    for m in rebuilt:
        bucket = buckets.get(_message_reuse_key(m))
        if bucket:
            out.append(bucket.pop())
        else:
            out.append(m)
    return out


def _goal_llm_complete(run_context: Any, llm: LLMProvider):
    """Bind a prompt→text callable for goal planner/verifier/strategist."""

    async def _complete(prompt: str) -> str:
        meta = getattr(run_context, "_metadata", None) if run_context else None
        if isinstance(meta, dict) and callable(meta.get("goal_llm_complete")):
            return await meta["goal_llm_complete"](prompt)
        resp = await llm.chat([LLMMessage(role="user", content=prompt)])
        return str(getattr(resp, "content", "") or "")

    return _complete


def _drain_interject(run_context: Any) -> str | None:
    """Legacy single-string drain — prefer :func:`drain_interject_messages`."""
    from clawagents.interjection import drain_interjects

    parts = drain_interjects(run_context)
    if not parts:
        return None
    # Compat: join only if caller expects one blob (prefer multi-message path).
    return parts[0] if len(parts) == 1 else "\n\n".join(parts)


def _drain_interject_messages(run_context: Any) -> list[LLMMessage]:
    """Each pending interject → one standalone synthetic user turn (Grok parity)."""
    from clawagents.interjection import drain_interjects

    return [LLMMessage(role="user", content=text) for text in drain_interjects(run_context)]


_GOAL_REMINDER_START = "\n\n<!--claw:goal-reminder-->\n"
_GOAL_REMINDER_END = "\n<!--/claw:goal-reminder-->"


def _strip_goal_reminder(system_content: str) -> str:
    if not isinstance(system_content, str):
        return system_content
    start = system_content.find("<!--claw:goal-reminder-->")
    if start < 0:
        # Legacy unwrapped block from first-turn injection
        marker = "\n## Active Goal\n"
        idx = system_content.find(marker)
        if idx < 0:
            return system_content
        return system_content[:idx].rstrip()
    # Include any blank lines immediately before the marker
    while start > 0 and system_content[start - 1] == "\n":
        start -= 1
        if start > 0 and system_content[start - 1] == "\n":
            break
    end = system_content.find("<!--/claw:goal-reminder-->", start)
    if end < 0:
        return system_content[:start].rstrip()
    end += len("<!--/claw:goal-reminder-->")
    return (system_content[:start] + system_content[end:]).rstrip()


def _sync_goal_reminder_into_system(
    messages: list[LLMMessage],
    run_context: Any,
) -> None:
    """Keep Active Goal standing reminder fresh when start_goal runs mid-loop."""
    if not messages or getattr(messages[0], "role", None) != "system":
        return
    content = messages[0].content
    if not isinstance(content, str):
        return
    try:
        from clawagents.config.features import is_enabled as _feat_goal_sys
        from clawagents.goal import get_goal_tracker, goal_system_reminder

        meta = getattr(run_context, "_metadata", None)
        if not (isinstance(meta, dict) and meta.get("goal_mode")):
            return
        if not _feat_goal_sys("goal_autopilot"):
            return
        tracker = get_goal_tracker(run_context)
        rem = goal_system_reminder(tracker.state if tracker else None)
    except Exception:
        return
    base = _strip_goal_reminder(content)
    if rem:
        messages[0].content = base + _GOAL_REMINDER_START + rem + _GOAL_REMINDER_END
    else:
        messages[0].content = base


async def _compact_if_needed(
    messages: list[LLMMessage],
    context_window: int,
    llm: LLMProvider,
    emit: OnEvent,
    token_multiplier: float = 1.0,
    model_name: Optional[str] = None,
    run_context: Optional[RunContext] = None,
    fire_hook: Optional[Callable[..., Any]] = None,
    savings_history: list[float] | None = None,
    taxonomy_dispatcher: Any | None = None,
) -> list[LLMMessage]:
    messages = _truncate_old_tool_args(messages)

    # Soft-cap verbose assistant/user turns before heavier compaction.
    try:
        from clawagents.memory.output_trim import trim_verbose_messages

        messages, trimmed_n = trim_verbose_messages(messages)
        if trimmed_n:
            emit("context", {"message": f"trimmed {trimmed_n} verbose turn(s)"})
    except Exception:
        logger.debug("output trim failed", exc_info=True)

    # If recent compressions are thrashing, prefer artifact eviction only.
    if savings_history:
        try:
            from clawagents.memory.compaction import is_compression_thrashing

            if is_compression_thrashing(savings_history):
                emit("context", {
                    "message": "compaction thrashing detected — skipping LLM summarize; soft-trim only",
                })
                return messages
        except Exception:
            logger.debug("thrash check failed", exc_info=True)

    effective_window, ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    budget = int(effective_window * ratio)
    from clawagents.memory.compact_tool_results import compact_tool_results
    from clawagents.harness_profiles import resolve_harness_profile

    profile = resolve_harness_profile(model_name)
    headroom = (
        float(profile.compaction_headroom_ratio)
        if profile and profile.compaction_headroom_ratio is not None
        else 0.7
    )

    messages, compacted = compact_tool_results(
        messages,
        max_input_tokens=budget,
        token_multiplier=token_multiplier,
        headroom_ratio=headroom,
    )
    if compacted:
        emit("context", {"message": "compacted oversized tool results before summarization"})
    current_tokens = _estimate_messages_tokens(messages, token_multiplier)

    # Pre-compaction memory flush (Grok memory_flush)
    try:
        from clawagents.config.features import is_enabled as _feat_flush
        from clawagents.memory.memory_flush import should_flush, run_memory_flush

        cycle = 0
        if run_context is not None and isinstance(run_context._metadata, dict):
            cycle = int(run_context._metadata.get("compaction_cycle") or 0)
        ws = None
        if run_context is not None and isinstance(run_context._metadata, dict):
            ws = run_context._metadata.get("workspace")
        if _feat_flush("memory_flush") and should_flush(
            current_tokens, budget, compaction_cycle=cycle, workspace=ws
        ):
            async def _flush_llm(prompt: str) -> str:
                resp = await llm.chat([LLMMessage(role="user", content=prompt)])
                return str(getattr(resp, "content", "") or "")

            flush_out = await run_memory_flush(
                messages, _flush_llm, workspace=ws, compaction_cycle=cycle
            )
            emit(
                "context",
                {
                    "message": (
                        f"memory flush: {flush_out.status}"
                        + (f" ({flush_out.detail})" if flush_out.detail else "")
                    )
                },
            )
            if run_context is not None and isinstance(run_context._metadata, dict):
                run_context._metadata["compaction_cycle"] = cycle + 1
    except Exception:
        logger.debug("memory flush failed", exc_info=True)

    # Prefire / two-pass: summarize before the hard cliff (Grok two_pass).
    try:
        from clawagents.config.features import is_enabled as _feat_prefire

        prefire_ratio = 0.85
        if (
            _feat_prefire("prefire_compaction")
            and current_tokens > int(budget * prefire_ratio)
            and current_tokens <= budget
        ):
            emit(
                "context",
                {
                    "message": (
                        f"prefire compaction ~{current_tokens}/{budget} "
                        f"(>{int(prefire_ratio * 100)}% headroom)"
                    )
                },
            )
            # Force into the compaction path below by pretending we're over budget
            # only for the summarize stage — callers still see a successful shrink.
            current_tokens = budget + 1
    except Exception:
        logger.debug("prefire compaction probe failed", exc_info=True)

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

    if fire_hook is not None:
        try:
            await fire_hook("on_pre_compact", len(messages), current_tokens)
        except Exception:
            logger.debug("on_pre_compact hook failed", exc_info=True)

    if taxonomy_dispatcher is not None:
        try:
            from clawagents.hooks.external import dispatch_taxonomy_hook
            from clawagents.hooks.taxonomy import HookEvent

            await dispatch_taxonomy_hook(
                taxonomy_dispatcher,
                HookEvent.PRE_COMPACT,
                {
                    "message_count": len(messages),
                    "current_tokens": current_tokens,
                    "budget": budget,
                },
                blocking=False,
            )
        except Exception:
            logger.debug("taxonomy pre_compact hook failed", exc_info=True)

    system_msgs: list[LLMMessage] = []
    non_system: list[LLMMessage] = []
    for m in messages:
        (system_msgs if m.role == "system" else non_system).append(m)

    if len(non_system) <= _RECENT_MESSAGES_TO_KEEP:
        return messages

    # ── Grok-style full-replace (preferred when enabled) ───────────────
    try:
        from clawagents.config.features import is_enabled as _feat

        if _feat("full_replace_compaction"):
            from clawagents.memory.full_replace_compaction import (
                apply_full_replace_compaction,
                build_state_reminder,
            )
            from clawagents.context.carryover import (
                get_compaction_carryover,
                set_compaction_carryover,
            )

            workspace = None
            if run_context is not None:
                ws = run_context._metadata.get("workspace")
                if isinstance(ws, str):
                    workspace = ws
            if not workspace:
                workspace = os.getcwd()

            # Auto-enrich carryover from transcript signals when host didn't set it
            try:
                task_focus = ""
                for m in non_system:
                    if m.role == "user" and isinstance(m.content, str) and _is_compactable_user(m):
                        task_focus = m.content[:500]
                        break
                recent_files = _extract_recent_files(non_system)
                existing = get_compaction_carryover(run_context, task_context=task_focus)
                active = list(getattr(run_context, "active_skills", {}) or {})
                invoked = list(existing.invoked_skills) or active
                for name in active:
                    if name not in invoked:
                        invoked.append(name)
                if run_context is not None and (
                    not existing.recent_files
                    or not existing.task_focus
                    or (active and not existing.invoked_skills)
                ):
                    set_compaction_carryover(
                        run_context,
                        task_focus=existing.task_focus or task_focus or None,
                        recent_files=existing.recent_files or recent_files,
                        recent_work_log=existing.recent_work_log,
                        invoked_skills=invoked,
                        active_workers=existing.active_workers,
                        channel_log=existing.channel_log,
                        plan_reminder=existing.plan_reminder,
                        metadata=existing.metadata,
                    )
                carryover = get_compaction_carryover(run_context, task_context=task_focus)
                carryover_md = carryover.to_markdown()
                reminder = build_state_reminder(
                    recent_files=carryover.recent_files,
                    plan_text=carryover.plan_reminder,
                    invoked_skills=carryover.invoked_skills,
                    active_workers=carryover.active_workers,
                )
            except Exception:
                logger.debug("full-replace carryover enrich failed", exc_info=True)
                carryover_md = ""
                reminder = None

            fr = await apply_full_replace_compaction(
                messages,
                llm,
                workspace=workspace,
                carryover_markdown=carryover_md or None,
                system_reminder=reminder,
                history_then_steps=_feat("history_then_steps"),
            )
            if fr is not None:
                # Prefer identity reuse for system + recent tails that survived
                fr = _reuse_messages_where_possible(messages, fr)
                fr_tokens = _estimate_messages_tokens(fr, token_multiplier)
                # Input ladder: if still over budget, retry lossy summarizer input
                if fr_tokens > budget:
                    fr_lossy = await apply_full_replace_compaction(
                        messages,
                        llm,
                        workspace=workspace,
                        carryover_markdown=carryover_md or None,
                        system_reminder=reminder,
                        lossy=True,
                        history_then_steps=_feat("history_then_steps"),
                    )
                    if fr_lossy is not None:
                        fr = _reuse_messages_where_possible(messages, fr_lossy)
                        fr_tokens = _estimate_messages_tokens(fr, token_multiplier)
                if fr_tokens <= budget or fr_tokens < current_tokens:
                    if savings_history is not None and current_tokens > 0:
                        saved = max(0, current_tokens - fr_tokens)
                        savings_history.append(saved / current_tokens * 100.0)
                    emit("context", {
                        "message": (
                            f"full-replace compaction rebuilt history "
                            f"(~{current_tokens} → ~{fr_tokens} tokens)"
                        ),
                    })
                    emit("compact_progress", {
                        "phase": "end",
                        "message": "compaction completed via full_replace",
                        "mode": "full_replace",
                        "before_tokens": current_tokens,
                        "after_tokens": fr_tokens,
                    })
                    if fire_hook is not None:
                        try:
                            summary_snip = next(
                                (
                                    m.content
                                    for m in fr
                                    if isinstance(m.content, str)
                                    and "being continued" in m.content
                                ),
                                None,
                            )
                            await fire_hook("on_post_compact", len(fr), summary_snip)
                        except Exception:
                            logger.debug("on_post_compact hook failed", exc_info=True)
                    if taxonomy_dispatcher is not None:
                        try:
                            from clawagents.hooks.external import dispatch_taxonomy_hook
                            from clawagents.hooks.taxonomy import HookEvent

                            await dispatch_taxonomy_hook(
                                taxonomy_dispatcher,
                                HookEvent.POST_COMPACT,
                                {
                                    "message_count": len(fr),
                                    "before_tokens": current_tokens,
                                    "after_tokens": fr_tokens,
                                    "mode": "full_replace",
                                },
                                blocking=False,
                            )
                        except Exception:
                            logger.debug("taxonomy post_compact hook failed", exc_info=True)
                    # Greppable compaction segment archive
                    try:
                        from clawagents.config.features import is_enabled as _feat_seg
                        from clawagents.memory.compaction_segments import (
                            write_segment,
                            segment_recovery_hint,
                        )

                        if _feat_seg("compaction_segments") and workspace:
                            archive = "\n".join(
                                f"[{m.role}] {str(m.content)[:500]}"
                                for m in messages
                                if getattr(m, "role", None) != "system"
                            )[:12000]
                            write_segment(
                                archive,
                                workspace=workspace,
                                turns=max(1, len(messages) - 1),
                            )
                            emit("context", {"message": segment_recovery_hint()})
                    except Exception:
                        logger.debug("compaction segment write failed", exc_info=True)
                    return fr
    except Exception:
        logger.debug("full_replace_compaction path failed; falling back", exc_info=True)

    # Prefer hardened compress_messages_safe when it yields meaningful savings.
    try:
        from clawagents.memory.compaction import AgentMessage, compress_messages_safe

        agent_msgs = [
            AgentMessage(
                role=m.role,
                content=_content_key_text(m.content),
            )
            for m in ([*system_msgs, *non_system])
        ]
        safe = await compress_messages_safe(
            llm,
            agent_msgs,
            context_window=effective_window,
            protect_first_n=max(1, len(system_msgs)),
            protect_last_n=_RECENT_MESSAGES_TO_KEEP,
        )
        if safe.get("effective"):
            # Rebuilding from AgentMessage would mint new objects for every
            # turn — breaking the identity-based session-persistence tracker
            # (unpersisted turns silently vanish) and stripping tool-call
            # metadata (tool_calls_meta / tool_call_id) that providers need
            # for transcript linkage. Reuse the original LLMMessage object
            # whenever (role, content) survived compression unchanged.
            # AgentMessage views only carry role+content — empty assistant/tool
            # bodies are ambiguous and must not reuse originals (that was the
            # tool_calls_meta swap). Non-empty content still reuses by text.
            _originals_by_key: dict[tuple[str, str], list[LLMMessage]] = {}
            for _om in (*system_msgs, *non_system):
                _text = _content_key_text(_om.content)
                if _om.role in ("assistant", "tool") and not str(_text or "").strip():
                    continue  # never offer empty bodies for reuse
                _originals_by_key.setdefault((_om.role, _text), []).append(_om)

            def _reuse_original(role: str, content: str) -> LLMMessage:
                text = content or ""
                if role in ("assistant", "tool") and not text.strip():
                    return LLMMessage(role=role, content=text)
                bucket = _originals_by_key.get((role, _content_key_text(text)))
                if bucket:
                    return bucket.pop()
                return LLMMessage(role=role, content=text)

            compact_out = [
                _reuse_original(m.role, m.content or "")
                for m in safe["messages"]
            ]
            summary_text = str(safe.get("summary") or "")
            if savings_history is not None:
                savings_history.append(float(safe.get("compression_savings_pct") or 0.0))
            # Preserve carryover, then normalize to user+assistant compaction pair.
            try:
                task_context = ""
                for m in non_system:
                    if m.role == "user" and not (
                        isinstance(m.content, str) and m.content.startswith("[Tool Result]")
                    ):
                        task_context = m.content[:500] if isinstance(m.content, str) else ""
                        break
                carryover = get_compaction_carryover(run_context, task_context=task_context)
                carryover_text = carryover.to_markdown()
            except Exception:
                logger.debug("carryover enrich after compress_messages_safe failed", exc_info=True)
                carryover_text = ""

            handoff = f"[System — Compacted History]\n{summary_text}"
            if carryover_text and summary_text:
                handoff = (
                    f"[System — Compacted History]\n{carryover_text}\n\n"
                    f"## Conversation Summary\n{summary_text}"
                )
            replaced = False
            for i, m in enumerate(compact_out):
                if m.role != "system" and (m.content or "") == summary_text:
                    compact_out[i] = LLMMessage(role="user", content=handoff)
                    replaced = True
                    break
            if not replaced and summary_text:
                insert_at = len([m for m in compact_out if m.role == "system"])
                compact_out.insert(insert_at, LLMMessage(role="user", content=handoff))
            # Assistant ack keeps providers that expect alternating roles happy.
            if summary_text and not any(
                m.role == "assistant"
                and isinstance(m.content, str)
                and "compacted handoff" in m.content.lower()
                for m in compact_out
            ):
                # Insert immediately after the handoff user message.
                for i, m in enumerate(compact_out):
                    if m.role == "user" and isinstance(m.content, str) and "Compacted History" in m.content:
                        compact_out.insert(
                            i + 1,
                            LLMMessage(
                                role="assistant",
                                content="Understood — continuing from the compacted handoff summary.",
                            ),
                        )
                        break
            compacted_tokens = _estimate_messages_tokens(compact_out, token_multiplier)
            if compacted_tokens <= budget:
                emit("context", {
                    "message": (
                        f"compress_messages_safe saved "
                        f"{safe.get('compression_savings_pct', 0):.1f}%"
                    ),
                })
                emit("compact_progress", {
                    "phase": "end",
                    "message": "compaction completed via compress_messages_safe",
                    "older_messages": len(safe.get("dropped_messages_list") or []),
                    "recent_messages": _RECENT_MESSAGES_TO_KEEP,
                })
                if fire_hook is not None:
                    try:
                        await fire_hook("on_post_compact", len(compact_out), summary_text or None)
                    except Exception:
                        logger.debug("on_post_compact hook failed", exc_info=True)
                return compact_out
            # "Effective" savings alone are not enough: the transcript is
            # still over budget, and returning here would hand the next LLM
            # call an oversized context. Keep the lossless savings and
            # escalate to the summarization tier below.
            emit("context", {
                "message": (
                    f"compress_messages_safe saved "
                    f"{safe.get('compression_savings_pct', 0):.1f}% but "
                    f"~{compacted_tokens} tokens still exceeds budget {budget} "
                    "— escalating to summarization"
                ),
            })
            messages = compact_out
            system_msgs = [m for m in compact_out if m.role == "system"]
            non_system = [m for m in compact_out if m.role != "system"]
    except Exception:
        logger.debug("compress_messages_safe path failed; falling back", exc_info=True)

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
            out = [*system_msgs, *recent]
            if fire_hook is not None:
                try:
                    await fire_hook("on_post_compact", len(out), None)
                except Exception:
                    logger.debug("on_post_compact hook failed", exc_info=True)
            return out

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
        out = [*system_msgs, summary, *recent]
        if fire_hook is not None:
            try:
                await fire_hook("on_post_compact", len(out), summary_text)
            except Exception:
                logger.debug("on_post_compact hook failed", exc_info=True)
        return out
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
        out = [*system_msgs, *recent]
        if fire_hook is not None:
            try:
                await fire_hook("on_post_compact", len(out), None)
            except Exception:
                logger.debug("on_post_compact hook failed", exc_info=True)
        return out


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
            content = _content_key_text(m.content)
            lines.append(f"**{m.role}**: {content[:2000]}\n\n")

        path.write_text("".join(lines), "utf-8")
    except Exception:
        logger.debug("Pre-compact transcript archival failed", exc_info=True)


def _offload_history(messages: list[LLMMessage]) -> str | None:
    """Save older messages to a JSON file before compaction.

    Content is passed through :func:`redact_obj` first — the offload file
    is a plain-text artifact on disk, so secrets the agent saw mid-run
    (bearer tokens, ``.env`` contents, …) must not be persisted verbatim,
    matching the redaction applied by every other persistence surface.
    """
    try:
        from clawagents.redact import redact_obj

        _get_history_dir().mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = _get_history_dir() / f"compacted_{ts}_{len(messages)}msgs.json"
        data = redact_obj([{"role": m.role, "content": m.content} for m in messages])
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
    max_iterations: int = 200,
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
    atlas: bool = False,  # deprecated no-op (ATLAS removed)
    atlas_config: Optional[Any] = None,  # deprecated no-op
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
    action_mode: str = "tools",
    approval_handler: Any = None,
    require_approval_tools: Optional[list[str]] = None,
    image_blocks: Optional[list[dict]] = None,
    file_blocks: Optional[list[dict]] = None,
    session_end_tail: bool = True,
) -> AgentState:
    """Single ReAct loop: LLM → tools → LLM → tools → ... → final answer."""
    if features is not None:
        from clawagents.config.features import temporary_overrides

        with temporary_overrides(features):
            return await _run_agent_graph_core(
                task=task,
                llm=llm,
                tools=tools,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                streaming=streaming,
                context_window=context_window,
                on_event=on_event,
                before_llm=before_llm,
                before_tool=before_tool,
                after_tool=after_tool,
                use_native_tools=use_native_tools,
                trajectory=trajectory,
                rethink=rethink,
                learn=learn,
                atlas=atlas,
                atlas_config=atlas_config,
                preview_chars=preview_chars,
                response_chars=response_chars,
                timeout_s=timeout_s,
                advisor_llm=advisor_llm,
                advisor_max_calls=advisor_max_calls,
                run_context=run_context,
                user_context=user_context,
                hooks=hooks,
                agent_hooks=agent_hooks,
                input_guardrails=input_guardrails,
                output_guardrails=output_guardrails,
                output_type=output_type,
                on_stream_event=on_stream_event,
                session=session,
                session_preload_limit=session_preload_limit,
                handoffs=handoffs,
                agent_name=agent_name,
                action_mode=action_mode,
                approval_handler=approval_handler,
                require_approval_tools=require_approval_tools,
                image_blocks=image_blocks,
                file_blocks=file_blocks,
                session_end_tail=session_end_tail,
            )
    return await _run_agent_graph_core(
        task=task,
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        streaming=streaming,
        context_window=context_window,
        on_event=on_event,
        before_llm=before_llm,
        before_tool=before_tool,
        after_tool=after_tool,
        use_native_tools=use_native_tools,
        trajectory=trajectory,
        rethink=rethink,
        learn=learn,
        atlas=atlas,
        atlas_config=atlas_config,
        preview_chars=preview_chars,
        response_chars=response_chars,
        timeout_s=timeout_s,
        advisor_llm=advisor_llm,
        advisor_max_calls=advisor_max_calls,
        run_context=run_context,
        user_context=user_context,
        hooks=hooks,
        agent_hooks=agent_hooks,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        output_type=output_type,
        on_stream_event=on_stream_event,
        session=session,
        session_preload_limit=session_preload_limit,
        handoffs=handoffs,
        agent_name=agent_name,
        action_mode=action_mode,
        approval_handler=approval_handler,
        require_approval_tools=require_approval_tools,
        image_blocks=image_blocks,
        file_blocks=file_blocks,
        session_end_tail=session_end_tail,
    )


async def _run_agent_graph_core(
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
    atlas: bool = False,  # deprecated no-op (ATLAS removed)
    atlas_config: Optional[Any] = None,  # deprecated no-op
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
    action_mode: str = "tools",
    approval_handler: Any = None,
    require_approval_tools: Optional[list[str]] = None,
    image_blocks: Optional[list[dict]] = None,
    file_blocks: Optional[list[dict]] = None,
    session_end_tail: bool = True,
) -> AgentState:
    """Internal ReAct loop body (feature overrides applied by :func:`run_agent_graph`)."""
    registry = tools or ToolRegistry()
    action_mode_norm = action_mode if action_mode in ("tools", "code") else "tools"
    require_approval_set = {
        n for n in (require_approval_tools or []) if n
    }
    # When approval_handler is set, write-class tools require approval by default.
    if approval_handler is not None:
        from clawagents.permissions.mode import WRITE_CLASS_TOOLS

        require_approval_set |= set(WRITE_CLASS_TOOLS)
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
    # Tools (execute streaming, skills) read callbacks/metadata from run_context.
    run_context.on_event = emit
    # Ephemeral id for ${SESSION_ID} skill substitutions when persistence is off.
    if not getattr(run_context, "session_id", None):
        import uuid as _uuid

        _ephemeral_sid = f"run-{_uuid.uuid4().hex[:12]}"
        run_context.session_id = _ephemeral_sid
        run_context._metadata["session_id"] = _ephemeral_sid
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
    # Expose hooks to nested tools (e.g. task → on_subagent_start/end).
    run_context._metadata["hooks"] = active_hooks
    run_context._metadata["agent_name"] = agent_name or "ClawAgent"

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
    _compaction_savings: list[float] = []

    # Trajectory recorder (opt-in; learn implies trajectory)
    recorder = None
    if trajectory or learn:
        from clawagents.trajectory.recorder import TrajectoryRecorder
        recorder = TrajectoryRecorder(task=task, response_chars=response_chars)

    # Bind workspace + goal LLM for tools / final gate (parent runs).
    if run_context is not None:
        meta = run_context._metadata
        if not isinstance(meta.get("workspace"), str):
            meta["workspace"] = os.getcwd()
        if getattr(registry, "_permission_engine", None) is not None:
            meta.setdefault("permission_engine", registry._permission_engine)
        if before_tool is not None:
            meta["before_tool"] = before_tool
        if approval_handler is not None:
            meta["approval_handler"] = approval_handler

        async def _bound_goal_llm(prompt: str) -> str:
            resp = await llm.chat([LLMMessage(role="user", content=prompt)])
            return str(getattr(resp, "content", "") or "")

        meta["goal_llm_complete"] = _bound_goal_llm
        try:
            from clawagents.config.features import is_enabled as _feat_goal_bind
            from clawagents.goal import (
                GoalTracker,
                attach_goal_to_run_context,
                get_goal_tracker,
            )

            # Only bind the disk-backed goal tracker in Goal mode. Act/Plan must
            # not inherit an active `.clawagents/goal/state.json` from a prior run.
            _want_goal = bool(meta.get("goal_mode"))
            if (
                _want_goal
                and _feat_goal_bind("goal_autopilot")
                and get_goal_tracker(run_context) is None
            ):
                attach_goal_to_run_context(
                    run_context, GoalTracker(meta["workspace"])
                )
        except Exception:
            logger.debug("goal tracker bind failed", exc_info=True)


    # Feature: Session Persistence — save session as append-only JSONL
    session_writer = None
    from clawagents.config.features import is_enabled as _feat_enabled
    if _feat_enabled("session_persistence"):
        from clawagents.session.persistence import SessionWriter
        session_writer = SessionWriter()
        run_context.session_id = session_writer.session_id
        run_context._metadata["session_id"] = session_writer.session_id
        emit("context", {"message": f"session: {session_writer.session_id} → {session_writer.path}"})

    # Feature: External Hooks — load shell hooks from .clawagents/hooks.json or env
    ext_hook_runner = None
    hooks_cfg = None
    if _feat_enabled("external_hooks"):
        from clawagents.hooks.external import load_hooks_config, ExternalHookRunner
        hooks_cfg = load_hooks_config()
        if hooks_cfg:
            ext_hook_runner = ExternalHookRunner(hooks_cfg)
            emit("context", {"message": "external hooks: loaded"})

    taxonomy_dispatcher = None
    try:
        from clawagents.hooks.external import build_taxonomy_dispatcher

        taxonomy_dispatcher = build_taxonomy_dispatcher(hooks_cfg)
        if taxonomy_dispatcher is not None:
            emit("context", {"message": "hook taxonomy: loaded"})
    except Exception:
        logger.debug("hook taxonomy load failed", exc_info=True)

    if taxonomy_dispatcher is not None and isinstance(
        getattr(run_context, "_metadata", None), dict
    ):
        run_context._metadata["taxonomy_dispatcher"] = taxonomy_dispatcher

    _base_emit = emit

    async def _fire_taxonomy(
        event: Any,
        payload: dict[str, Any] | None = None,
        *,
        blocking: bool = False,
    ) -> None:
        if taxonomy_dispatcher is None:
            return
        try:
            from clawagents.hooks.external import dispatch_taxonomy_hook

            await dispatch_taxonomy_hook(
                taxonomy_dispatcher,
                event,
                payload or {},
                blocking=blocking,
            )
        except Exception:
            pass

    async def _tax_permission_denied(
        tool: str, reason: str, *, source: str,
    ) -> None:
        from clawagents.hooks.taxonomy import HookEvent

        await _fire_taxonomy(
            HookEvent.PERMISSION_DENIED,
            {"tool": tool, "reason": reason, "source": source},
        )

    def emit(kind: EventKind, data: dict[str, Any] | None = None) -> None:
        payload = data or {}
        _base_emit(kind, payload)
        if kind == "warn" and taxonomy_dispatcher is not None:
            from clawagents.hooks.taxonomy import HookEvent

            msg = str(payload.get("message") or payload)
            try:
                asyncio.get_running_loop().create_task(
                    _fire_taxonomy(
                        HookEvent.NOTIFICATION,
                        {"message": msg, "kind": "warn"},
                    )
                )
            except RuntimeError:
                pass

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

    prompt_to_use = append_model_identity(
        system_prompt or BASE_SYSTEM_PROMPT,
        getattr(llm, "name", None),
        getattr(llm, "model", None),
    )
    lesson_preamble = ""
    dynamic_parts: list[str] = []

    # PTRL Layer 1: Pre-run lesson injection (skipped for isolated subagents).
    if learn and not getattr(run_context, "skip_memory", False):
        from clawagents.trajectory.lessons import build_lesson_preamble
        preamble = build_lesson_preamble()
        if preamble:
            dynamic_parts.append(preamble)
            emit("context", {"message": "PTRL: injected lessons from past runs"})

    # Goal autopilot standing reminder (preferred long-horizon gate).
    # Wrapped in markers so mid-run start_goal can refresh it each turn.
    try:
        from clawagents.config.features import is_enabled as _feat_goal_sys
        from clawagents.goal import get_goal_tracker, goal_system_reminder

        _goal_mode_on = bool(
            isinstance(run_context._metadata, dict)
            and run_context._metadata.get("goal_mode")
        )
        if _goal_mode_on and _feat_goal_sys("goal_autopilot"):
            _gt_sys = get_goal_tracker(run_context)
            _rem = goal_system_reminder(_gt_sys.state if _gt_sys else None)
            if _rem:
                dynamic_parts.append(
                    "<!--claw:goal-reminder-->\n"
                    + _rem
                    + "\n<!--/claw:goal-reminder-->"
                )
                emit("context", {"message": "goal: injected active goal reminder"})
    except Exception:
        logger.debug("goal system reminder failed", exc_info=True)

    # Dynamic context packs
    # Dynamic context packs (after cache boundary) — local only.
    if not getattr(run_context, "skip_memory", False):
        from clawagents.config.features import is_enabled
        try:
            if is_enabled("core_memory"):
                from clawagents.memory.core_memory import load_core_memory
                cm = load_core_memory()
                if cm:
                    dynamic_parts.append(cm)
            if is_enabled("context_ledger"):
                from clawagents.memory.context_ledger import load_ledger_preamble
                led = load_ledger_preamble()
                if led:
                    dynamic_parts.append(led)
            if is_enabled("memory_bank"):
                from clawagents.memory.core_memory import (
                    ensure_memory_bank_stubs,
                    load_memory_bank_preamble,
                )
                ensure_memory_bank_stubs()
                mb = load_memory_bank_preamble()
                if mb:
                    dynamic_parts.append(mb)
            if is_enabled("fact_store"):
                from clawagents.memory.facts import live_facts_preamble
                facts = live_facts_preamble()
                if facts:
                    dynamic_parts.append(facts)
            from clawagents.tools.context_tools import load_plan_preamble
            plan = load_plan_preamble()
            if plan:
                dynamic_parts.append(plan)
            if is_enabled("repo_map_inject"):
                from clawagents.memory.repo_map import build_repo_map
                rm = build_repo_map(max_chars=3_500)
                if rm:
                    dynamic_parts.append(rm)
                    emit("context", {"message": "injected ranked repo map"})
            # Workspace facts models need before inventing git /tmp paths.
            try:
                import tempfile
                from pathlib import Path as _P

                from clawagents.tools.git_tools import is_git_work_tree

                ws = str(getattr(run_context, "workspace", None) or _P.cwd())
                git_ok = is_git_work_tree(ws)
                scratch = tempfile.gettempdir()
                meta = getattr(run_context, "_metadata", None)
                sb_name = "workspace"
                if isinstance(meta, dict):
                    sb_name = str(meta.get("sandbox_profile") or sb_name)
                dynamic_parts.append(
                    "## Workspace env\n"
                    f"- workspace: `{ws}`\n"
                    f"- is_git_repo: {'true' if git_ok else 'false'}\n"
                    f"- sandbox: `{sb_name}`\n"
                    f"- scratch_dir: `{scratch}` (also /tmp when sandbox allows)\n"
                    + (
                        "- Prefer `snapshot_diff` to review edits (no git).\n"
                        if not git_ok
                        else "- Prefer `git_status` / `git_diff` to review edits.\n"
                    )
                    + "- Do not chain `&& git …` after syntax checks when is_git_repo is false.\n"
                    + (
                        "- OS sandbox is off — home config CLIs (gcloud/aws/docker) may run.\n"
                        if sb_name == "off"
                        else ""
                    )
                )
            except Exception:
                logger.debug("workspace env preamble failed", exc_info=True)
        except Exception:
            logger.debug("dynamic context pack failed", exc_info=True)

    if dynamic_parts:
        lesson_preamble = "\n\n".join(dynamic_parts)

    # Insert __CACHE_BOUNDARY__ between static (instructions + tools) and dynamic content.
    # The Anthropic provider splits on this marker to enable prompt caching.
    system_content = build_system_prompt(
        base_prompt=prompt_to_use,
        tool_description=tool_desc,
        lesson_preamble=lesson_preamble,
    )
    # Attach images/files (if any) to the first user message as content
    # blocks so the model sees pixels/documents. ``current_task`` stays the
    # plain string, so compaction/events/session paths that expect text are
    # unaffected.
    if image_blocks or file_blocks:
        first_user_content: Any = (
            ([{"type": "text", "text": task}] if task else [])
            + list(image_blocks or [])
            + list(file_blocks or [])
        )
    else:
        first_user_content = task
    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=system_content),
        LLMMessage(role="user", content=first_user_content),
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

    def _budget_tokens(msgs: list[LLMMessage], mult: float | None = None) -> int:
        return _estimate_messages_tokens(
            msgs,
            mult if mult is not None else token_multiplier,
            resolved_model_name,
            cached_system_tokens=_cached_sys_tokens or None,
        )

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

    if taxonomy_dispatcher is not None:
        try:
            from clawagents.hooks.external import dispatch_taxonomy_hook
            from clawagents.hooks.taxonomy import HookEvent

            await dispatch_taxonomy_hook(
                taxonomy_dispatcher,
                HookEvent.SESSION_START,
                {"task": task[:500] if task else ""},
                blocking=False,
            )
            await dispatch_taxonomy_hook(
                taxonomy_dispatcher,
                HookEvent.USER_PROMPT_SUBMIT,
                {"prompt": task[:2000] if task else ""},
                blocking=False,
            )
        except Exception:
            logger.debug("taxonomy session_start hook failed", exc_info=True)

    # Session rewind: snapshot workspace-touched files at prompt boundary
    try:
        from clawagents.config.features import is_enabled as _feat_rw

        if _feat_rw("session_rewind") or _feat_rw("hunk_watcher"):
            from clawagents.memory.hunk_watcher import get_watcher

            _ws_rw = None
            if run_context is not None and isinstance(run_context._metadata, dict):
                _ws_rw = run_context._metadata.get("workspace")
            w = get_watcher(_ws_rw)
            meta_rw = (
                run_context._metadata
                if run_context is not None and isinstance(run_context._metadata, dict)
                else None
            )
            idx = int((meta_rw or {}).get("prompt_index") or 0) + 1
            # RunContext is recreated every VS Code turn, so metadata alone always
            # yields idx=1 and overwrites prompt_0001.json. Prefer the watcher.
            idx = max(idx, int(getattr(w, "_prompt_index", 0) or 0) + 1)
            if meta_rw is not None:
                meta_rw["prompt_index"] = idx
            _conv_marker: list[dict[str, str]] = []
            for _m in messages[-6:]:
                if _m.role in ("user", "assistant"):
                    _preview = (
                        _m.content
                        if isinstance(_m.content, str)
                        else str(_m.content)
                    )
                    _conv_marker.append(
                        {"role": _m.role, "preview": _preview[:120]}
                    )
            w.snapshot_turn(
                idx,
                user_text=(task or "")[:2000],
                message_count=len(messages),
                conversation_marker=_conv_marker,
            )
    except Exception:
        logger.debug("rewind snapshot failed", exc_info=True)

    # Session protocol — hydrate history before first LLM call (non-destructive).
    _session_preloaded_count = 0
    # The current task message is part of the durable conversation and must
    # be persisted alongside run-appended messages (assistant/tool turns).
    _session_task_msg = next((m for m in messages if m.role == "user"), None)
    if session is not None:
        try:
            prior = await _session_get_items(session, limit=session_preload_limit)
            if prior:
                # Limited preload can start mid tool-pair (tool result without
                # its assistant tool_calls) → provider 400. Drop leading orphans
                # here; full pair sanitization still runs before each LLM call.
                prior = _drop_leading_orphan_tools(prior)
                prior = _patch_dangling_tool_calls(prior)
                # Replay history between the system prompt and the current
                # task so the transcript reads in true conversational order.
                insert_at = next(
                    (i for i, m in enumerate(messages) if m.role == "user"),
                    len(messages),
                )
                messages = [*messages[:insert_at], *prior, *messages[insert_at:]]
                state.messages = messages
                _session_preloaded_count = len(prior)
        except Exception as err:
            emit("warn", {"message": f"session load failed: {err}"})
    # Identity-based tracking of preloaded vs. run-appended messages.
    # A numeric cursor breaks when compaction rebuilds ``messages`` or
    # dangling-tool-call patching inserts items mid-list, silently losing
    # (or duplicating) turns at persist time. Instead we track message
    # *objects*: anything unseen at the top of a round was appended by the
    # run and gets persisted; anything synthesized by the pre-LLM transform
    # pipeline (compaction summaries, patch inserts, prompt injections) is
    # marked seen without being persisted. The dict keeps strong refs so
    # CPython can't recycle an id() for a new message.
    _session_initial_ids = frozenset(id(m) for m in messages)
    _session_seen: dict[int, LLMMessage] = {id(m): m for m in messages}
    _session_new_msgs: list[LLMMessage] = []
    # Persist the task itself: without it, replayed history contains answers
    # with no questions and multi-turn recall silently degrades.
    if session is not None and _session_task_msg is not None:
        _session_new_msgs.append(_session_task_msg)

    def _session_note_messages(track: bool) -> None:
        for _m in messages:
            _mid = id(_m)
            if _mid in _session_seen:
                continue
            _session_seen[_mid] = _m
            if track:
                _session_new_msgs.append(_m)

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
    # Set when a handoff installs the combined parent+child transcript on
    # ``state.messages`` — the post-loop assignment must not overwrite it.
    _handoff_transcript_set = False
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        emit("warn", {"message": "interrupted"})
        cancel_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except (NotImplementedError, OSError, RuntimeError, ValueError):
        # ValueError: uvloop (and RuntimeError: vanilla asyncio) refuse signal
        # handlers off the main thread — e.g. when embedded in a server that
        # runs agent turns in worker threads. Ctrl-C handling is best-effort.
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

            # One increment per loop round, unconditionally — previously only
            # a handful of exit paths bumped this, so normal multi-round runs
            # reported "1 iteration" in events and the session writer.
            state.iterations += 1

            # Mid-turn user redirect — each entry is its own synthetic user turn.
            try:
                from clawagents.config.features import is_enabled as _feat_ij

                if _feat_ij("mid_turn_interject"):
                    _ij_msgs = _drain_interject_messages(run_context)
                    if _ij_msgs:
                        messages.extend(_ij_msgs)
                        emit(
                            "context",
                            {
                                "message": (
                                    f"mid-turn interjection applied ({len(_ij_msgs)} turn(s))"
                                )
                            },
                        )
            except Exception:
                logger.debug("mid-turn interject drain failed", exc_info=True)

            # Refresh Goal standing reminder if start_goal fired mid-run.
            try:
                _sync_goal_reminder_into_system(messages, run_context)
            except Exception:
                logger.debug("goal reminder sync failed", exc_info=True)

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

            # Capture messages appended by the previous round *before* the
            # transform pipeline below can compact/rebuild the list.
            _session_note_messages(track=True)

            # Patch dangling tool calls before sending to LLM
            messages = _patch_dangling_tool_calls(messages)
            # Claude Code pattern: clear old tool results — but only when the
            # transcript is actually filling up (see _MICRO_COMPACT_MIN_USAGE_RATIO).
            from clawagents.harness_profiles import resolve_harness_profile as _rhp
            _mc_profile = _rhp(resolved_model_name)
            _mc_keep = (
                int(_mc_profile.clear_tool_keep)
                if _mc_profile and _mc_profile.clear_tool_keep is not None
                else _MICRO_COMPACT_KEEP_RECENT
            )
            _mc_ratio = (
                float(_mc_profile.clear_tool_trigger_ratio)
                if _mc_profile and _mc_profile.clear_tool_trigger_ratio is not None
                else _MICRO_COMPACT_MIN_USAGE_RATIO
            )
            if (
                _budget_tokens(messages)
                > context_window * _mc_ratio
            ):
                messages = _micro_compact_tool_results(messages, keep_recent=_mc_keep)
            messages = _soft_trim_messages(messages, context_window, token_multiplier, emit, resolved_model_name)
            messages = await _compact_if_needed(
                messages, context_window, llm, emit, token_multiplier, resolved_model_name, run_context,
                fire_hook=_fire_hook,
                savings_history=_compaction_savings,
                taxonomy_dispatcher=taxonomy_dispatcher,
            )
            # Compaction / trim can still leave pairs inconsistent — sanitize again.
            messages = _patch_dangling_tool_calls(messages)

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

            # Framework-synthesized messages (compaction summaries, dangling
            # tool-call patches, hook/prompt injections) are regenerated per
            # run — mark them seen so they are never persisted to the session.
            _session_note_messages(track=False)

            buf, _buffer_chunk = _make_buffer()

            def on_chunk(chunk: str) -> None:
                _buffer_chunk(chunk)
                _emit_typed("assistant_delta", {"delta": chunk})

            if active_hooks:
                await _fire_hook("on_llm_start", resolved_model_name or "", messages)
            try:
                # Native structured output (json_schema → provider wire formats)
                try:
                    from clawagents.config.features import is_enabled as _feat_so
                    from clawagents.structured_output import schema_from_output_type

                    if _feat_so("structured_output") and output_type is not None:
                        _schema = schema_from_output_type(output_type)
                        setattr(llm, "_structured_json_schema", _schema)
                    else:
                        setattr(llm, "_structured_json_schema", None)
                except Exception:
                    pass
                # Doom-loop recovery: force a non-thinking response channel.
                chat_messages = messages
                if (
                    run_context is not None
                    and isinstance(run_context._metadata, dict)
                    and run_context._metadata.get("doom_force_response")
                ):
                    chat_messages = list(messages) + [
                        LLMMessage(
                            role="user",
                            content=(
                                "CRITICAL recovery instruction: Do NOT emit any "
                                "<think>...</think> blocks or private chain-of-thought. "
                                "Respond with the next tool call or final answer only."
                            ),
                        )
                    ]
                # When a skill has activated an allowed-tools boundary, only
                # advertise those tools (+ control plane) so the model stops
                # reaching for tools the registry will refuse.
                turn_tools = native_schemas
                if turn_tools and run_context is not None:
                    allowed = getattr(run_context, "active_skill_allowed_tools", None)
                    if allowed is not None:
                        control = {
                            "use_skill",
                            "list_skills",
                            "retrieve_tool_result",
                        }
                        turn_tools = [
                            s
                            for s in turn_tools
                            if s.name in allowed or s.name in control
                        ]
                response = await llm.chat(
                    chat_messages,
                    on_chunk=on_chunk if streaming else None,
                    cancel_event=cancel_event,
                    tools=turn_tools,
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
                        _budget_tokens(messages, 1.0), 1,
                    )
                    token_multiplier = min(observed_ratio * 1.1, 3.0)
                    # Also shrink the effective window: the multiplier is
                    # capped at 3.0, so with a wildly overstated
                    # CONTEXT_WINDOW (e.g. 1M configured on a 128K model)
                    # compaction below would never fire and every retry
                    # would overflow again.
                    context_window = max(int(context_window * 0.5), 16_000)
                    emit("context", {
                        "message": (
                            f"token overflow — calibrated multiplier to {token_multiplier:.2f}, "
                            f"shrunk effective window to {context_window} "
                            f"(retry {overflow_retries}/{_MAX_OVERFLOW_RETRIES})"
                        ),
                    })
                    messages = _soft_trim_messages(messages, context_window, token_multiplier, emit, resolved_model_name)
                    messages = await _compact_if_needed(
                        messages, context_window, llm, emit, token_multiplier, resolved_model_name, run_context,
                        fire_hook=_fire_hook,
                        savings_history=_compaction_savings,
                        taxonomy_dispatcher=taxonomy_dispatcher,
                    )
                    # Don't persist recovery-compaction artifacts to the session.
                    _session_note_messages(track=False)
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
            # Provider-native thinking field (Anthropic/Gemini) when no <think> tags.
            if not _thinking_content and getattr(response, "thinking", None):
                _thinking_content = str(response.thinking)

            # Doom-loop: thinking OR response tail-repetition → resample with temp bump
            try:
                from clawagents.config.features import is_enabled as _feat_doom
                from clawagents.doom_loop import (
                    DoomLoopRecoveryPolicy,
                    DoomLoopState,
                    detect_tail_repetition,
                    note_trigger,
                    should_resample,
                )

                if _feat_doom("doom_loop"):
                    sig = None
                    if _thinking_content:
                        sig = detect_tail_repetition(
                            _thinking_content, channel="thinking"
                        )
                    if sig is None and response.content:
                        sig = detect_tail_repetition(
                            str(response.content), channel="response"
                        )
                    if sig is not None:
                        meta = (
                            run_context._metadata
                            if run_context is not None
                            and isinstance(run_context._metadata, dict)
                            else {}
                        )
                        doom_state = meta.get("doom_loop_state")
                        if not isinstance(doom_state, DoomLoopState):
                            doom_state = DoomLoopState()
                            if meta is not None:
                                meta["doom_loop_state"] = doom_state
                        note_trigger(doom_state, sig)
                        pol = DoomLoopRecoveryPolicy()
                        if should_resample(sig, doom_state, pol):
                            doom_state.retry_count += 1
                            # Bump temperature so resample is not a deterministic repeat.
                            try:
                                cur_t = float(getattr(llm, "_temperature", 0.0) or 0.0)
                                setattr(llm, "_temperature", min(1.0, max(0.4, cur_t + 0.4)))
                            except Exception:
                                pass
                            # Force next attempt onto the response channel (no think tags).
                            if meta is not None:
                                meta["doom_force_response"] = True
                            emit(
                                "warn",
                                {
                                    "message": (
                                        f"doom-loop {sig.label} — resampling "
                                        f"({doom_state.retry_count}/{pol.max_retries}, "
                                        "force response channel)"
                                    )
                                },
                            )
                            continue
                        if (
                            meta is not None
                            and meta.get("doom_force_response")
                            and sig.channel == "thinking"
                        ):
                            # Forced-response attempt still thought — keep forcing.
                            meta["doom_force_response"] = True
            except Exception:
                logger.debug("doom-loop check failed", exc_info=True)

            # Clear force-response once we got a usable non-doom turn.
            if (
                run_context is not None
                and isinstance(run_context._metadata, dict)
                and run_context._metadata.pop("doom_force_response", None)
            ):
                pass

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

                # ── CodeAct: execute Python action when no native tool calls ──
                if action_mode_norm == "code":
                    from clawagents.graph.codeact import extract_code_action, run_code_action

                    code = extract_code_action(response.content or "")
                    if code:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=response.content,
                            thinking=_thinking_content,
                        ))
                        emit("tool_call", {"name": "codeact", "args": {"code": code[:500]}})

                        def _run_async(coro: Any) -> Any:
                            try:
                                asyncio.get_running_loop()
                            except RuntimeError:
                                return asyncio.run(coro)
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                                return pool.submit(asyncio.run, coro).result()

                        result = run_code_action(
                            code,
                            registry,
                            before_tool=before_tool,
                            run_context=run_context,
                            run_async=_run_async,
                        )
                        state.tool_calls += len(result.get("tool_calls") or []) or 1
                        obs = str(result.get("observation") or "")
                        emit("tool_result", {
                            "name": "codeact",
                            "success": not result.get("error"),
                            "output": obs[:2000],
                        })
                        if result.get("done"):
                            state.result = obs
                            state.status = "done"
                            emit("final_content", {"content": state.result})
                            _emit_typed("assistant_message", {
                                "content": state.result,
                                "thinking": _thinking_content,
                            })
                            break
                        messages.append(LLMMessage(
                            role="user",
                            content=f"[CodeAct Observation]\n{obs}",
                        ))
                        continue

                # ── Advisor: final check before declaring done ──
                # Track whether the final-check path already appended the
                # assistant message: when the advisor errored or returned
                # nothing, the fall-through below appended it a second time.
                _final_assistant_appended = False
                if advisor_llm and _advisor_call_count > 0 and _advisor_call_count < advisor_max_calls and state.tool_calls > 0:
                    messages.append(LLMMessage(role="assistant", content=response.content, thinking=_thinking_content))
                    _final_assistant_appended = True
                    await _consult_advisor(messages, "final-check")
                    # If advisor injected guidance, let the LLM process it
                    last_msg = messages[-1] if messages else None
                    if last_msg and isinstance(last_msg.content, str) and last_msg.content.startswith("[Advisor Guidance]"):
                        continue


                # ── Goal autopilot final gate ──
                try:
                    from clawagents.config.features import is_enabled as _feat_goal
                    from clawagents.goal import GoalOrchestrator, get_goal_tracker

                    _goal_mode_on = bool(
                        isinstance(run_context._metadata, dict)
                        and run_context._metadata.get("goal_mode")
                    )
                    _goal_tracker = get_goal_tracker(run_context) if _goal_mode_on else None
                    if (
                        _goal_mode_on
                        and _feat_goal("goal_autopilot")
                        and _goal_tracker is not None
                        and _goal_tracker.is_active()
                        and _goal_tracker.state is not None
                        and _goal_tracker.state.status.value
                        not in ("done", "failed", "paused")
                    ):
                        if not _final_assistant_appended:
                            messages.append(LLMMessage(
                                role="assistant",
                                content=response.content,
                                thinking=_thinking_content,
                            ))
                            _final_assistant_appended = True
                        evidence = (response.content or "")[:6000]
                        orch = GoalOrchestrator(
                            _goal_tracker,
                            _goal_llm_complete(run_context, llm),
                        )
                        ok, gst = await orch.verify(evidence)
                        if not ok:
                            inject = (
                                "[Goal Verifier] Completion rejected. Continue the plan.\n"
                                f"Consecutive misses: {gst.consecutive_not_achieved}.\n"
                            )
                            if gst.strategy_text:
                                inject += f"Strategy note:\n{gst.strategy_text[:2000]}\n"
                            messages.append(LLMMessage(role="user", content=inject))
                            emit("context", {"message": "goal verifier rejected completion"})
                            continue
                        emit("context", {"message": "goal verifier accepted — DONE"})
                except Exception:
                    logger.debug("goal final gate failed", exc_info=True)

                if recorder:
                    recorder.record_turn(
                        response_text=response.content or "",
                        model=response.model,
                        tokens_used=response.tokens_used,
                        thinking=_thinking_content,
                    )
                state.result = _sanitize_assistant_text(response.content)
                state.status = "done"
                emit("final_content", {"content": state.result})
                _emit_typed("assistant_message", {
                    "content": state.result,
                    "thinking": _thinking_content,
                })
                if not _final_assistant_appended:
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
                        pre_handoff_items=[
                            m for m in messages if id(m) in _session_initial_ids
                        ],
                        new_items=[
                            m for m in messages if id(m) not in _session_initial_ids
                        ],
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
                                session_end_tail=False,
                            )
                        except Exception as run_err:
                            emit("warn", {"message": f"handoff target raised: {run_err}"})
                            messages.append(LLMMessage(
                                role="user",
                                content=f"[Handoff Error] Target agent failed: {run_err}",
                            ))
                            continue

                    state.result = child_state.result
                    state.status = child_state.status if child_state.status != "running" else "done"
                    state.final_output = (
                        child_state.final_output
                        if child_state.final_output is not None
                        else child_state.result
                    )
                    state.tool_calls += child_state.tool_calls
                    state.messages = messages + child_state.messages
                    _handoff_transcript_set = True
                    break

            if loop_tracker.is_circuit_broken():
                emit("warn", {"message": f"circuit breaker tripped ({loop_tracker._no_progress_count} no-progress calls) — breaking"})
                state.status = "done"
                state.result = "Circuit breaker: too many tool calls with no progress. Stopping."
                break

            poll_hit = None
            for call in tool_calls:
                poll_hit = loop_tracker.check_known_poll_no_progress(call.tool_name, call.args)
                if poll_hit and poll_hit.level == "critical":
                    break
            if poll_hit and poll_hit.stuck and poll_hit.level == "critical":
                emit("warn", {"message": poll_hit.message})
                state.status = "done"
                state.result = poll_hit.message
                break

            if loop_tracker.is_hard_looping_batch(tool_calls):
                names = ", ".join(c.tool_name for c in tool_calls)
                emit("warn", {"message": f"tool loop detected ({names}) — breaking"})
                state.status = "done"
                state.result = f"Tool loop detected ({names}). Stopping."
                break

            if loop_tracker.is_ping_ponging():
                recent_unique = list(set(loop_tracker._history[-6:]))
                emit("warn", {"message": f"ping-pong oscillation detected ({' ↔ '.join(recent_unique)}) — breaking"})
                state.status = "done"
                state.result = "Ping-pong loop detected between tools. Stopping."
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
                continue

            # Surface intermediate assistant commentary (text alongside tool
            # calls) on the typed stream. Skipped in text-tool mode where
            # ``response.content`` is the raw JSON tool call itself.
            if use_native_tools and response.content and response.content.strip():
                _emit_typed("assistant_message", {
                    "content": response.content,
                    "thinking": _thinking_content,
                })

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

                # External / taxonomy pre_tool_use hook
                if ext_hook_runner and taxonomy_dispatcher is None:
                    try:
                        ext_allowed, ext_args = await ext_hook_runner.pre_tool_use(call.tool_name, call.args)
                        if not ext_allowed:
                            emit("tool_skipped", {"name": call.tool_name, "reason": "blocked by external hook"})
                            await _tax_permission_denied(
                                call.tool_name,
                                "blocked by external hook",
                                source="external_hook",
                            )
                            if use_native_tools and native_tc and native_tc.tool_call_id:
                                messages.append(LLMMessage(
                                    role="assistant",
                                    content=response.content or "",
                                    tool_calls_meta=[{
                                        "id": native_tc.tool_call_id,
                                        "name": call.tool_name,
                                        "args": call.args,
                                    }],
                                    gemini_parts=getattr(response, "gemini_parts", None),
                                    thinking=_thinking_content,
                                ))
                                messages.append(LLMMessage(
                                    role="tool",
                                    content=f"[Tool Skipped] {call.tool_name} was blocked by external hook.",
                                    tool_call_id=native_tc.tool_call_id,
                                ))
                            else:
                                messages.append(LLMMessage(
                                    role="user",
                                    content=f"[Tool Skipped] {call.tool_name} was blocked by external hook.",
                                ))
                            continue
                        call = ParsedToolCall(tool_name=call.tool_name, args=ext_args)
                    except Exception as hook_err:
                        emit("warn", {"message": f"external pre_tool_use hook error: {hook_err}"})

                if taxonomy_dispatcher is not None:
                    try:
                        from clawagents.hooks.external import dispatch_taxonomy_hook
                        from clawagents.hooks.taxonomy import HookEvent

                        tax_allowed, tax_reason = await dispatch_taxonomy_hook(
                            taxonomy_dispatcher,
                            HookEvent.PRE_TOOL_USE,
                            {"tool": call.tool_name, "args": call.args},
                            blocking=True,
                        )
                        if not tax_allowed:
                            reason = tax_reason or "blocked by taxonomy hook"
                            emit("tool_skipped", {"name": call.tool_name, "reason": reason})
                            await _tax_permission_denied(
                                call.tool_name, reason, source="taxonomy",
                            )
                            if use_native_tools and native_tc and native_tc.tool_call_id:
                                messages.append(LLMMessage(
                                    role="assistant",
                                    content=response.content or "",
                                    tool_calls_meta=[{
                                        "id": native_tc.tool_call_id,
                                        "name": call.tool_name,
                                        "args": call.args,
                                    }],
                                    gemini_parts=getattr(response, "gemini_parts", None),
                                    thinking=_thinking_content,
                                ))
                                messages.append(LLMMessage(
                                    role="tool",
                                    content=f"[Tool Skipped] {call.tool_name} was not approved: {reason}",
                                    tool_call_id=native_tc.tool_call_id,
                                ))
                            else:
                                messages.append(LLMMessage(
                                    role="user",
                                    content=f"[Tool Skipped] {call.tool_name} was not approved: {reason}",
                                ))
                            continue
                    except Exception as hook_err:
                        emit("warn", {"message": f"taxonomy pre_tool_use hook error: {hook_err}"})

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
                        await _tax_permission_denied(
                            call.tool_name, hook_reason, source="before_tool",
                        )
                        # Close the native tool pair so Gemini sees
                        # model(function_call) → user(function_response), not a bare
                        # "[Tool Skipped]" user turn that breaks turn alternation.
                        if use_native_tools and native_tc and native_tc.tool_call_id:
                            messages.append(LLMMessage(
                                role="assistant",
                                content=response.content or "",
                                tool_calls_meta=[{
                                    "id": native_tc.tool_call_id,
                                    "name": call.tool_name,
                                    "args": call.args,
                                }],
                                gemini_parts=getattr(response, "gemini_parts", None),
                                thinking=_thinking_content,
                            ))
                            messages.append(LLMMessage(
                                role="tool",
                                content=f"[Tool Skipped] {call.tool_name} was not approved: {hook_reason}",
                                tool_call_id=native_tc.tool_call_id,
                            ))
                        else:
                            messages.append(LLMMessage(
                                role="user",
                                content=f"[Tool Skipped] {call.tool_name} was not approved: {hook_reason}",
                            ))
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
                    if use_native_tools and native_tc and native_tc.tool_call_id:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=response.content or "",
                            tool_calls_meta=[{
                                "id": native_tc.tool_call_id,
                                "name": call.tool_name,
                                "args": call.args,
                            }],
                            gemini_parts=getattr(response, "gemini_parts", None),
                            thinking=_thinking_content,
                        ))
                        messages.append(LLMMessage(
                            role="tool",
                            content=f"[Tool Skipped] {call.tool_name} was rejected: {reason}",
                            tool_call_id=native_tc.tool_call_id,
                        ))
                    else:
                        messages.append(LLMMessage(
                            role="user",
                            content=f"[Tool Skipped] {call.tool_name} was rejected: {reason}",
                        ))
                    continue
                if approval_state is None:
                    tool_obj = registry.tools.get(call.tool_name)
                    needs_approval = (
                        call.tool_name in require_approval_set
                        or bool(getattr(tool_obj, "require_approval", False))
                    )
                    emit("approval_required", {"name": call.tool_name, "id": native_tc_id})
                    _emit_typed("approval_required", {
                        "tool_name": call.tool_name,
                        "call_id": native_tc_id,
                        "args": call.args,
                    })
                    if needs_approval and approval_handler is not None:
                        approved = await _wait_for_tool_approval(
                            run_context,
                            native_tc_id,
                            call.tool_name,
                            call.args if isinstance(call.args, dict) else {},
                            approval_handler=approval_handler,
                            emit=emit,
                        )
                        if not approved:
                            emit("tool_skipped", {
                                "name": call.tool_name,
                                "reason": "approval denied or timed out",
                            })
                            if use_native_tools and native_tc and native_tc.tool_call_id:
                                messages.append(LLMMessage(
                                    role="assistant",
                                    content=response.content or "",
                                    tool_calls_meta=[{
                                        "id": native_tc.tool_call_id,
                                        "name": call.tool_name,
                                        "args": call.args,
                                    }],
                                    gemini_parts=getattr(response, "gemini_parts", None),
                                    thinking=_thinking_content,
                                ))
                                messages.append(LLMMessage(
                                    role="tool",
                                    content=f"[Tool Skipped] {call.tool_name} was not approved",
                                    tool_call_id=native_tc.tool_call_id,
                                ))
                            else:
                                messages.append(LLMMessage(
                                    role="user",
                                    content=f"[Tool Skipped] {call.tool_name} was not approved",
                                ))
                            continue
                        run_context.approve_tool(native_tc_id, tool_name=call.tool_name)

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
                    if not tool_result.success:
                        await _fire_hook(
                            "on_tool_failure",
                            call.tool_name,
                            native_tc_id,
                            tool_result.error or str(tool_result.output)[:500],
                        )


                # External / taxonomy post_tool_use hook
                if ext_hook_runner and taxonomy_dispatcher is None:
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

                if taxonomy_dispatcher is not None:
                    try:
                        from clawagents.hooks.external import dispatch_taxonomy_hook
                        from clawagents.hooks.taxonomy import HookEvent

                        await dispatch_taxonomy_hook(
                            taxonomy_dispatcher,
                            HookEvent.POST_TOOL_USE,
                            {
                                "tool": call.tool_name,
                                "args": call.args,
                                "success": tool_result.success,
                                "output": str(tool_result.output)[:1000],
                            },
                            blocking=False,
                        )
                        if not tool_result.success:
                            await dispatch_taxonomy_hook(
                                taxonomy_dispatcher,
                                HookEvent.POST_TOOL_USE_FAILURE,
                                {
                                    "tool": call.tool_name,
                                    "args": call.args,
                                    "error": tool_result.error or str(tool_result.output)[:500],
                                },
                                blocking=False,
                            )
                    except Exception as hook_err:
                        emit("warn", {"message": f"taxonomy post_tool_use hook error: {hook_err}"})

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
                    try:
                        from clawagents.media.images import sanitize_tool_output

                        tool_output = sanitize_tool_output(raw_output)  # type: ignore[assignment]
                    except Exception:
                        logger.debug("sanitize_tool_output failed", exc_info=True)
                        tool_output = raw_output
                    preview: str = "[Multimodal Array Content]"
                else:
                    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

                    tool_output, artifact_id = prepare_tool_output_for_context(
                        tool_name=call.tool_name,
                        tool_use_id=native_tc.tool_call_id if native_tc else call.tool_name,
                        output=raw_output,
                        workspace=_run_context_workspace(run_context),
                        success=bool(tool_result.success),
                    )
                    if artifact_id is not None:
                        emit("context", {"message": f"tool output crushed/stored id={artifact_id}"})
                    preview = tool_output[:preview_chars]

                tool_output = _post_tool_side_effects(
                    call.tool_name,
                    call.args if isinstance(call.args, dict) else {},
                    tool_result.success,
                    tool_output,
                    emit=emit,
                    run_context=run_context,
                )
                if isinstance(tool_output, str):
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
                # ── External pre_tool_use hook (parallel) ──
                # Mirror the single-call path: an external policy gate must not
                # be bypassable by batching a forbidden call with another one.
                _candidate_pairs: list[tuple[int, ParsedToolCall]] = list(enumerate(tool_calls))
                if ext_hook_runner and taxonomy_dispatcher is None:
                    _ext_pairs: list[tuple[int, ParsedToolCall]] = []
                    for _orig_i, _c in _candidate_pairs:
                        try:
                            ext_allowed, ext_args = await ext_hook_runner.pre_tool_use(_c.tool_name, _c.args)
                            if not ext_allowed:
                                emit("tool_skipped", {"name": _c.tool_name, "reason": "blocked by external hook"})
                                await _tax_permission_denied(
                                    _c.tool_name,
                                    "blocked by external hook",
                                    source="external_hook",
                                )
                                messages.append(LLMMessage(role="user", content=f"[Tool Skipped] {_c.tool_name} was blocked by external hook."))
                                continue
                            _c = ParsedToolCall(tool_name=_c.tool_name, args=ext_args)
                        except Exception as hook_err:
                            emit("warn", {"message": f"external pre_tool_use hook error: {hook_err}"})
                        _ext_pairs.append((_orig_i, _c))
                    _candidate_pairs = _ext_pairs
                    if not _candidate_pairs:
                        continue

                if taxonomy_dispatcher is not None:
                    from clawagents.hooks.external import dispatch_taxonomy_hook
                    from clawagents.hooks.taxonomy import HookEvent

                    _tax_pairs: list[tuple[int, ParsedToolCall]] = []
                    for _orig_i, _c in _candidate_pairs:
                        try:
                            tax_allowed, tax_reason = await dispatch_taxonomy_hook(
                                taxonomy_dispatcher,
                                HookEvent.PRE_TOOL_USE,
                                {"tool": _c.tool_name, "args": _c.args},
                                blocking=True,
                            )
                            if not tax_allowed:
                                reason = tax_reason or "blocked by taxonomy hook"
                                emit("tool_skipped", {"name": _c.tool_name, "reason": reason})
                                await _tax_permission_denied(
                                    _c.tool_name, reason, source="taxonomy",
                                )
                                messages.append(
                                    LLMMessage(
                                        role="user",
                                        content=f"[Tool Skipped] {_c.tool_name} was not approved: {reason}",
                                    )
                                )
                                continue
                        except Exception as hook_err:
                            emit("warn", {"message": f"taxonomy pre_tool_use hook error: {hook_err}"})
                        _tax_pairs.append((_orig_i, _c))
                    _candidate_pairs = _tax_pairs
                    if not _candidate_pairs:
                        continue

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

                    for _orig_i, c in _candidate_pairs:
                        result_call, reason = _apply_hook(c)
                        if result_call is None:
                            emit("tool_skipped", {"name": c.tool_name, "reason": reason})
                            await _tax_permission_denied(
                                c.tool_name, reason, source="before_tool",
                            )
                        else:
                            approved_calls.append(result_call)
                            _approved_orig_indices.append(_orig_i)
                    if not approved_calls:
                        messages.append(LLMMessage(role="user", content="[Tool Skipped] All tool calls were not approved."))
                        continue
                else:
                    approved_calls = [c for _, c in _candidate_pairs]
                    _approved_orig_indices = [i for i, _ in _candidate_pairs]

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
                        if not _r.success:
                            await _fire_hook(
                                "on_tool_failure",
                                _c.tool_name,
                                _cid,
                                _r.error or str(_r.output)[:500],
                            )


                # External / taxonomy post_tool_use hook (parallel)
                if ext_hook_runner and taxonomy_dispatcher is None:
                    _ext_results: list[ToolResult] = []
                    for _c, _r in zip(approved_calls, results):
                        try:
                            ext_result = await ext_hook_runner.post_tool_use(
                                _c.tool_name, _c.args,
                                {"success": _r.success, "output": str(_r.output)[:1000]},
                            )
                            if "success" in ext_result and "output" in ext_result:
                                _r = ToolResult(
                                    success=ext_result["success"],
                                    output=ext_result["output"],
                                    error=ext_result.get("error"),
                                )
                        except Exception as hook_err:
                            emit("warn", {"message": f"external post_tool_use hook error: {hook_err}"})
                        _ext_results.append(_r)
                    results = _ext_results

                if taxonomy_dispatcher is not None:
                    from clawagents.hooks.external import dispatch_taxonomy_hook
                    from clawagents.hooks.taxonomy import HookEvent

                    for _c, _r in zip(approved_calls, results):
                        try:
                            await dispatch_taxonomy_hook(
                                taxonomy_dispatcher,
                                HookEvent.POST_TOOL_USE,
                                {
                                    "tool": _c.tool_name,
                                    "args": _c.args,
                                    "success": _r.success,
                                    "output": str(_r.output)[:1000],
                                },
                                blocking=False,
                            )
                            if not _r.success:
                                await dispatch_taxonomy_hook(
                                    taxonomy_dispatcher,
                                    HookEvent.POST_TOOL_USE_FAILURE,
                                    {
                                        "tool": _c.tool_name,
                                        "args": _c.args,
                                        "error": _r.error or str(_r.output)[:500],
                                    },
                                    blocking=False,
                                )
                        except Exception as hook_err:
                            emit("warn", {"message": f"taxonomy post_tool_use hook error: {hook_err}"})

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
                    _call_id = (
                        _approved_call_ids[_idx2]
                        if _idx2 < len(_approved_call_ids)
                        else call.tool_name
                    )
                    if isinstance(raw_out, list):
                        try:
                            from clawagents.media.images import sanitize_tool_output

                            output = sanitize_tool_output(raw_out)  # type: ignore[assignment]
                        except Exception:
                            logger.debug("sanitize_tool_output failed", exc_info=True)
                            output = raw_out
                        preview = "[Multimodal Array Content]"
                    else:
                        from clawagents.tool_output_artifacts import prepare_tool_output_for_context

                        output, artifact_id = prepare_tool_output_for_context(
                            tool_name=call.tool_name,
                            tool_use_id=_call_id,
                            output=raw_out,
                            workspace=_run_context_workspace(run_context),
                            success=bool(result.success),
                        )
                        if artifact_id is not None:
                            emit("context", {"message": f"tool output crushed/stored id={artifact_id}"})
                        preview = output[:preview_chars]

                    output = _post_tool_side_effects(
                        call.tool_name,
                        call.args if isinstance(call.args, dict) else {},
                        result.success,
                        output,
                        emit=emit,
                        run_context=run_context,
                    )
                    if isinstance(output, str):
                        preview = output[:preview_chars]

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

                    # Session: write tool result (parity with single-call path)
                    if session_writer:
                        session_writer.write_tool_result(
                            _call_id, call.tool_name, result.success,
                            str(result.output)[:2000],
                            error=result.error if not result.success else None,
                        )
                    
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
        except (NotImplementedError, OSError, RuntimeError, ValueError):
            # Mirrors add_signal_handler: uvloop raises ValueError and vanilla
            # asyncio RuntimeError when running off the main thread.
            pass

    elapsed = time.monotonic() - t0
    # Don't clobber the combined parent+child transcript a handoff installed.
    if not _handoff_transcript_set:
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
            # Count the judge's own LLM call into the run's usage totals so
            # PTRL/trajectory spend is never invisible to callers.
            judge_resp = judge_result.pop("_llm_response", None)
            if judge_resp is not None:
                try:
                    _accumulate_usage(judge_resp)
                except Exception:  # noqa: BLE001
                    pass
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
                    try:
                        from clawagents.trajectory.failure_learn import (
                            append_failure_lessons_to_agents_md,
                        )

                        promoted = append_failure_lessons_to_agents_md(lessons_text)
                        if promoted:
                            emit("context", {
                                "message": f"PTRL: appended {len(promoted)} failure lesson(s) to AGENTS.md",
                            })
                    except Exception:
                        logger.debug("PTRL: AGENTS.md failure-learn append failed", exc_info=True)
                    try:
                        from clawagents.config.features import is_enabled
                        if is_enabled("fact_store"):
                            from clawagents.memory.facts import promote_lesson_bullets_to_facts

                            facts = promote_lesson_bullets_to_facts(lessons_text)
                            if facts:
                                emit("context", {
                                    "message": f"PTRL: promoted {len(facts)} live fact(s)",
                                })
                    except Exception:
                        logger.debug("PTRL: fact promotion failed", exc_info=True)
                    try:
                        from clawagents.trajectory.lesson_promotion import maybe_promote_recurring_lessons

                        promoted = maybe_promote_recurring_lessons(
                            lessons_text,
                            task=run_summary.task,
                        )
                        if promoted:
                            emit("context", {
                                "message": f"PTRL: promoted {len(promoted)} recurring lesson(s) to skill_workshop",
                            })
                    except Exception:
                        logger.debug("PTRL: lesson promotion failed", exc_info=True)
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
            _session_note_messages(track=True)
            if _session_new_msgs:
                await _session_add_items(session, _session_new_msgs)
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

    # Dream consolidation + session-end taxonomy (non-blocking with timeout)
    try:
        from clawagents.config.features import is_enabled as _feat_dream

        _ws = None
        if run_context is not None and isinstance(run_context._metadata, dict):
            _ws = run_context._metadata.get("workspace")
        if _ws is None:
            import os as _os

            _ws = _os.getcwd()

        # Nested runs (handoff children, subagents, forks) are not session
        # ends: they must not append session logs or trigger dream — a dream
        # here burns an extra LLM call per child and rewrites MEMORY.md from
        # subagent context mid-parent-run.
        if session_end_tail and (_feat_dream("memory_dream") or _feat_dream("smart_memory")):
            from clawagents.memory.dream import (
                append_session_log,
                check_dream_gates,
                run_dream,
            )

            _stem = None
            if session_writer is not None:
                _stem = getattr(session_writer, "session_id", None)
            _log_body = f"## Task\n{(task or '')[:4000]}\n\n## Outcome\n{state.status}\n\n## Result\n{(state.result or '')[:8000]}"
            append_session_log(_log_body, workspace=_ws, stem=_stem)

        if session_end_tail and _feat_dream("memory_dream"):
            _gate = check_dream_gates(_ws)
            if not isinstance(_gate, str):

                async def _dream_llm(prompt: str) -> str:
                    resp = await llm.chat(
                        [LLMMessage(role="user", content=prompt)],
                    )
                    return str(getattr(resp, "content", "") or "")

                try:
                    dream_out = await asyncio.wait_for(
                        run_dream(_dream_llm, workspace=_ws),
                        timeout=90.0,
                    )
                    if dream_out.ok:
                        emit("context", {"message": f"dream: {dream_out.reason}"})
                    else:
                        emit("context", {"message": f"dream skipped: {dream_out.reason}"})
                except asyncio.TimeoutError:
                    # Do not orphan a second task while the cancelled coroutine
                    # still holds dream.lock — run_dream's finally releases it.
                    emit("context", {"message": "dream: timed out (lock released)"})
    except Exception:
        logger.debug("dream scheduling failed", exc_info=True)

    if taxonomy_dispatcher is not None:
        try:
            from clawagents.hooks.external import dispatch_taxonomy_hook
            from clawagents.hooks.taxonomy import HookEvent

            await dispatch_taxonomy_hook(
                taxonomy_dispatcher,
                HookEvent.SESSION_END,
                {
                    "status": state.status,
                    "result_preview": (state.result or "")[:500],
                    "tool_calls": state.tool_calls,
                },
                blocking=False,
            )
            _result_text = state.result or ""
            _stop_failed = (
                state.status in ("error", "max_iterations")
                or _result_text.startswith("[cancelled]")
                or _result_text.startswith("[interrupted]")
            )
            if _stop_failed:
                await dispatch_taxonomy_hook(
                    taxonomy_dispatcher,
                    HookEvent.STOP_FAILURE,
                    {
                        "status": state.status,
                        "message": _result_text or state.status,
                    },
                    blocking=False,
                )
                await dispatch_taxonomy_hook(
                    taxonomy_dispatcher,
                    HookEvent.NOTIFICATION,
                    {
                        "message": _result_text or state.status,
                        "kind": "stop_failure",
                    },
                    blocking=False,
                )
            await dispatch_taxonomy_hook(
                taxonomy_dispatcher,
                HookEvent.STOP,
                {"status": state.status},
                blocking=False,
            )
        except Exception:
            logger.debug("taxonomy session_end hook failed", exc_info=True)

    emit("agent_done", {
        "tool_calls": state.tool_calls,
        "iterations": state.iterations,
        "elapsed": elapsed,
        "usage": usage.to_dict(),
    })

    # Stranded interjects (arrived after last drain / on cancel) → host queues them.
    try:
        from clawagents.interjection import take_stranded_interjects

        stranded = take_stranded_interjects(run_context)
        if stranded:
            if run_context is not None and isinstance(getattr(run_context, "_metadata", None), dict):
                run_context._metadata["stranded_interjects"] = list(stranded)
            emit("stranded_interject", {"prompts": stranded, "count": len(stranded)})
    except Exception:
        logger.debug("stranded interject flush failed", exc_info=True)

    return state
