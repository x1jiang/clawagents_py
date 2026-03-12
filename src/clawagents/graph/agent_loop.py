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
import json
import logging
import math
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from clawagents.providers.llm import LLMProvider, LLMMessage, LLMResponse, NativeToolSchema, NativeToolCall, strip_thinking_tokens
from clawagents.tools.registry import ToolRegistry, ParsedToolCall, ToolResult

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

    patched: list[LLMMessage] = []
    for i, msg in enumerate(messages):
        patched.append(msg)

        # Look for assistant messages with JSON tool calls without a following [Tool Result]
        if msg.role == "assistant" and msg.content.startswith('{"tool":'):
            has_result = (
                i + 1 < len(messages)
                and messages[i + 1].role == "user"
                and isinstance(messages[i + 1].content, str)
                and messages[i + 1].content.startswith("[Tool Result]")
            )
            if not has_result:
                patched.append(LLMMessage(
                    role="user",
                    content="[Tool Result] Tool call was cancelled — the agent was interrupted before it could complete.",
                ))
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


# ─── Model-Aware Context Budget (learned from deepagents) ─────────────────

_MODEL_PROFILES: dict[str, dict[str, int | float]] = {
    # OpenAI
    "gpt-5": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gpt-5-mini": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gpt-5-nano": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gpt-4o": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    "gpt-4o-mini": {"max_input_tokens": 128_000, "budget_ratio": 0.80},
    # Gemini
    "gemini-3-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-3-flash-preview": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-2.5-flash": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    "gemini-2.5-pro": {"max_input_tokens": 1_000_000, "budget_ratio": 0.90},
    # Claude
    "claude-sonnet-4-5": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
    "claude-3-5-sonnet": {"max_input_tokens": 200_000, "budget_ratio": 0.85},
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

AgentStatus = Literal["running", "done", "error"]

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
]

OnEvent = Callable[[EventKind, dict[str, Any]], None]

# Hook types for extensibility without middleware overhead
BeforeLLMHook = Callable[[list["LLMMessage"]], list["LLMMessage"]]
BeforeToolHook = Callable[[str, dict[str, Any]], bool]
AfterToolHook = Callable[[str, dict[str, Any], "ToolResult"], "ToolResult"]


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
    sys.stderr.flush()


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
- Prefer fewer, well-targeted tool calls over many exploratory ones."""


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
        messages = [
            LLMMessage(role="system", content=sys_msg.content.replace(tool_desc, short_desc)),
            *messages[1:],
        ]
        tool_desc = short_desc
        emit("context", {"message": f"tier-1: shortened tool descriptions -> ~{_payload_tokens()} tokens"})

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    # ── Tier 2: Drop text tool descriptions if native schemas exist ───
    if tool_desc and native_schemas:
        sys_msg = messages[0]
        messages = [
            LLMMessage(role="system", content=sys_msg.content.replace(tool_desc, "").strip()),
            *messages[1:],
        ]
        tool_desc = ""
        emit("context", {"message": f"tier-2: removed text tool descriptions -> ~{_payload_tokens()} tokens"})

    if _payload_tokens() <= budget:
        return messages, tool_desc, native_schemas

    # ── Tier 3: Truncate system prompt, preserving core behavior ──────
    sys_content = messages[0].content
    max_sys_chars = int((budget - native_schema_tokens - _estimate_tokens(messages[1].content if len(messages) > 1 else "")) * _CHARS_PER_TOKEN * 0.8)
    if max_sys_chars > 200 and len(sys_content) > max_sys_chars:
        truncated = sys_content[:max_sys_chars] + "\n\n...(system prompt truncated to fit context window)"
        messages = [LLMMessage(role="system", content=truncated), *messages[1:]]
        emit("context", {"message": f"tier-3: truncated system prompt -> ~{_payload_tokens()} tokens"})

    if _payload_tokens() > budget:
        emit("warn", {
            "message": (
                f"pre-flight: payload still ~{_payload_tokens()} tokens after all shedding "
                f"(budget {budget}). Consider increasing CONTEXT_WINDOW or reducing tools/instruction."
            )
        })

    return messages, tool_desc, native_schemas


# ─── Soft-Trim: prune stale/low-value content before compaction ───────────

_SOFT_TRIM_RATIO = 0.60
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
    effective_window, _ratio = (
        _resolve_context_budget(model_name, context_window)
        if model_name
        else (context_window, _CONTEXT_BUDGET_RATIO)
    )
    soft_budget = int(effective_window * _SOFT_TRIM_RATIO)
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

    offload_path = _offload_history(older)
    if offload_path:
        emit("context", {"message": f"offloaded {len(older)} messages to {offload_path}"})

    task_context = ""
    for m in non_system:
        if m.role == "user" and not (isinstance(m.content, str) and m.content.startswith("[Tool Result]")):
            task_context = m.content[:500] if isinstance(m.content, str) else ""
            break

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

            chunk_summaries: list[str] = []
            for i, chunk in enumerate(chunks):
                chunk_summary = await _summarize_chunk(llm, chunk, task_context)
                chunk_summaries.append(f"### Chunk {i + 1}/{len(chunks)}\n{chunk_summary}")
            summary_text = "\n\n".join(chunk_summaries)

        if not summary_text.strip():
            emit("context", {"message": "compaction returned empty summary — dropping oldest"})
            return [*system_msgs, *recent]

        summary = LLMMessage(
            role="user",
            content=f"[System — Compacted History]\n{summary_text}",
        )
        emit("context", {"message": f"compacted {len(older)} messages into summary"})
        return [*system_msgs, summary, *recent]
    except Exception:
        logger.debug("Compaction LLM call failed", exc_info=True)
        emit("context", {"message": "compaction failed — dropping oldest messages"})
        return [*system_msgs, *recent]


# ─── History Offloading ───────────────────────────────────────────────────



def _get_history_dir() -> Path:
    return Path.cwd() / ".clawagents" / "history"


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
) -> AgentState:
    """Single ReAct loop: LLM → tools → LLM → tools → ... → final answer."""
    registry = tools or ToolRegistry()
    native_schemas: list[NativeToolSchema] | None = (
        registry.to_native_schemas() if use_native_tools and tools else None
    )
    tool_desc = registry.describe_for_llm() if not use_native_tools else ""
    loop_tracker = _ToolCallTracker()
    emit = on_event or _default_on_event

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

    token_multiplier = 1.0
    resolved_model_name: Optional[str] = None
    _cached_sys_tokens: int = 0  # Feature D: cache system prompt token count

    prompt_to_use = system_prompt or BASE_SYSTEM_PROMPT

    # PTRL Layer 1: Pre-run lesson injection
    if learn:
        from clawagents.trajectory.lessons import build_lesson_preamble
        preamble = build_lesson_preamble()
        if preamble:
            prompt_to_use = prompt_to_use + preamble
            emit("context", {"message": "PTRL: injected lessons from past runs"})

    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=f"{prompt_to_use}\n\n{tool_desc}"),
        LLMMessage(role="user", content=task),
    ]
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
    )

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

            try:
                _check_timeout()
            except TimeoutError as te:
                emit("warn", {"message": str(te)})
                state.status = "error"
                state.result = str(te)
                break

            # Patch dangling tool calls before sending to LLM
            messages = _patch_dangling_tool_calls(messages)
            messages = _soft_trim_messages(messages, context_window, token_multiplier, emit, resolved_model_name)
            messages = await _compact_if_needed(
                messages, context_window, llm, emit, token_multiplier, resolved_model_name,
            )

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
            try:
                response = await llm.chat(
                    messages,
                    on_chunk=on_chunk if streaming else None,
                    cancel_event=cancel_event,
                    tools=native_schemas,
                )
                if not resolved_model_name and response.model:
                    resolved_model_name = response.model
            except Exception as err:
                err_msg = str(err).lower()
                if "context" in err_msg or "token" in err_msg:
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
                        messages, context_window, llm, emit, token_multiplier, resolved_model_name,
                    )
                    continue

                logger.exception("LLM call failed at round %d", round_idx)
                emit("error", {"phase": "llm_call", "message": str(err)})
                state.status = "error"
                state.result = str(err)
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
                repeated_names = ", ".join(
                    c.tool_name for c in tool_calls
                    if loop_tracker.is_soft_looping(c.tool_name, c.args)
                )
                emit("warn", {"message": f"repeated tool call warning #{n}: {repeated_names}"})
                messages.append(LLMMessage(
                    role="user",
                    content=(
                        f"[System] You are re-calling {repeated_names} with the same arguments. "
                        "You already have the result in the conversation above. "
                        "Use the existing data instead of re-reading. "
                        "If you believe the task is complete, provide your final answer now."
                    ),
                ))
                state.iterations += 1
                continue

            if len(tool_calls) == 1:
                call = tool_calls[0]
                native_tc = native_tool_call_objects[0] if native_tool_call_objects else None
                emit("tool_call", {"name": call.tool_name})

                if before_tool:
                    approved = False
                    try:
                        approved = bool(before_tool(call.tool_name, call.args))
                    except Exception as hook_err:
                        emit("warn", {"message": f"before_tool hook error: {hook_err}"})
                    if not approved:
                        emit("tool_skipped", {"name": call.tool_name, "reason": "rejected by before_tool hook"})
                        messages.append(LLMMessage(role="user", content=f"[Tool Skipped] {call.tool_name} was not approved."))
                        continue

                loop_tracker.record(call.tool_name, call.args)

                tool_result = await registry.execute_tool(call.tool_name, call.args)
                state.tool_calls += 1

                if after_tool:
                    try:
                        hooked = after_tool(call.tool_name, call.args, tool_result)
                        if hasattr(hooked, "success") and hasattr(hooked, "output"):
                            tool_result = hooked
                        else:
                            emit("warn", {"message": "after_tool returned invalid ToolResult — ignored"})
                    except Exception as hook_err:
                        emit("warn", {"message": f"after_tool hook error: {hook_err}"})

                raw_output = (
                    tool_result.output if tool_result.success else f"Error: {tool_result.error}"
                )
                if isinstance(raw_output, list):
                    tool_output = raw_output
                    preview = "[Multimodal Array Content]"
                else:
                    tool_output = _evict_large_tool_result(call.tool_name, raw_output)
                    preview = tool_output[:preview_chars]

                emit("tool_result", {
                    "name": call.tool_name,
                    "success": tool_result.success,
                    "preview": preview,
                })

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
                    user_content = f"[Tool Result] {tool_output}" if isinstance(tool_output, str) else tool_output
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
                approved_calls = tool_calls
                if before_tool:
                    def _safe_check(c):
                        try:
                            return bool(before_tool(c.tool_name, c.args))
                        except Exception as hook_err:
                            emit("warn", {"message": f"before_tool hook error: {hook_err}"})
                            return False
                    approved_calls = [c for c in tool_calls if _safe_check(c)]
                    skipped = [c for c in tool_calls if c not in approved_calls]
                    for c in skipped:
                        emit("tool_skipped", {"name": c.tool_name, "reason": "rejected by before_tool hook"})
                    if not approved_calls:
                        messages.append(LLMMessage(role="user", content="[Tool Skipped] All tool calls were not approved."))
                        continue

                for call in approved_calls:
                    emit("tool_call", {"name": call.tool_name})
                loop_tracker.record_batch(approved_calls)

                results = await registry.execute_tools_parallel(approved_calls)
                state.tool_calls += len(approved_calls)

                if after_tool:
                    safe_results = []
                    for c, r in zip(approved_calls, results):
                        try:
                            hooked = after_tool(c.tool_name, c.args, r)
                            if hasattr(hooked, "success") and hasattr(hooked, "output"):
                                safe_results.append(hooked)
                            else:
                                emit("warn", {"message": "after_tool returned invalid ToolResult — ignored"})
                                safe_results.append(r)
                        except Exception as hook_err:
                            emit("warn", {"message": f"after_tool hook error: {hook_err}"})
                            safe_results.append(r)
                    results = safe_results

                # Build a map from ParsedToolCall index to NativeToolCall for ID lookup
                native_tc_map: dict[int, NativeToolCall] = {}
                if native_tool_call_objects:
                    # Match by index (approved_calls is a subset of tool_calls)
                    tc_idx = 0
                    for i, tc in enumerate(tool_calls):
                        if tc_idx < len(approved_calls) and tc is approved_calls[tc_idx]:
                            if i < len(native_tool_call_objects):
                                native_tc_map[tc_idx] = native_tool_call_objects[i]
                            tc_idx += 1

                call_summaries: list[str] = []
                tool_outputs: list[str] = []
                for call, result in zip(approved_calls, results):
                    raw_out = result.output if result.success else f"Error: {result.error}"
                    if isinstance(raw_out, list):
                        output = raw_out
                        preview = "[Multimodal Array Content]"
                    else:
                        output = _evict_large_tool_result(call.tool_name, raw_out)
                        preview = output[:preview_chars]
                        
                    emit("tool_result", {
                        "name": call.tool_name,
                        "success": result.success,
                        "preview": preview,
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
                        raw_p = result.output[:preview_chars] if result.success else (result.error or "")[:preview_chars]
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
    if learn and recorder and run_summary:
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

    emit("agent_done", {
        "tool_calls": state.tool_calls,
        "iterations": state.iterations,
        "elapsed": elapsed,
    })
    return state
