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

from clawagents.providers.llm import LLMProvider, LLMMessage, LLMResponse, NativeToolSchema, NativeToolCall
from clawagents.tools.registry import ToolRegistry, ParsedToolCall

logger = logging.getLogger(__name__)


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
_EVICTION_DIR = Path.cwd() / ".clawagents" / "large_results"


def _create_content_preview(content: str, head_lines: int = 5, tail_lines: int = 5) -> str:
    lines = content.split("\n")
    if len(lines) <= head_lines + tail_lines + 2:
        return content

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
        _EVICTION_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)
        file_path = _EVICTION_DIR / f"{sanitized}_{ts}.txt"
        file_path.write_text(output, "utf-8")

        preview = _create_content_preview(output)
        return (
            f"[Result too large ({len(output)} chars) — saved to {file_path}]\n"
            f"Use read_file to access the full result. Preview:\n\n{preview}"
        )
    except Exception:
        return output[:_EVICTION_CHARS_THRESHOLD] + "\n...(output truncated)"


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

_CHARS_PER_TOKEN = 4


def _estimate_tokens(content: str | list[dict], multiplier: float = 1.0) -> int:
    if isinstance(content, str):
        return math.ceil((len(content) / _CHARS_PER_TOKEN) * multiplier)
    else:
        # Multimodal roughly estimated: 500 tokens per image + text chars
        length = sum(len(p.get("text", "")) for p in content) + (len(content) * 2000)
        return math.ceil((length / _CHARS_PER_TOKEN) * multiplier)


def _estimate_messages_tokens(messages: list[LLMMessage], multiplier: float = 1.0) -> int:
    return sum(_estimate_tokens(m.content, multiplier) for m in messages)


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
    def __init__(self, window_size: int = 30, soft_limit: int = 3, hard_limit: int = 6):
        self._history: list[str] = []
        self._window_size = window_size
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._soft_warnings = 0

    def _key(self, tool_name: str, args: dict) -> str:
        try:
            return f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        except (TypeError, ValueError):
            return f"{tool_name}:{args}"

    def record(self, tool_name: str, args: dict) -> None:
        self._history.append(self._key(tool_name, args))
        if len(self._history) > self._window_size:
            self._history.pop(0)

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


# ─── Context Window Guard with Auto-Compaction ────────────────────────────

_CONTEXT_BUDGET_RATIO = 0.75
_RECENT_MESSAGES_TO_KEEP = 6


async def _compact_if_needed(
    messages: list[LLMMessage],
    context_window: int,
    llm: LLMProvider,
    emit: OnEvent,
    token_multiplier: float = 1.0,
    model_name: Optional[str] = None,
) -> list[LLMMessage]:
    # Phase 1: truncate tool args in older messages (cheap, no LLM call)
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

    # Single-pass split
    system_msgs: list[LLMMessage] = []
    non_system: list[LLMMessage] = []
    for m in messages:
        (system_msgs if m.role == "system" else non_system).append(m)

    if len(non_system) <= _RECENT_MESSAGES_TO_KEEP:
        return messages

    recent_count = min(_RECENT_MESSAGES_TO_KEEP, len(non_system))
    older = non_system[:-recent_count]
    recent = non_system[-recent_count:]

    # Phase 8: Offload full history to file before summarizing
    offload_path = _offload_history(older)
    if offload_path:
        emit("context", {"message": f"offloaded {len(older)} messages to {offload_path}"})

    text_log = "\n\n".join(
        f"[{m.role.upper()}]: {m.content}" for m in older
    )

    summary_prompt = (
        "Compress the following agent conversation history into a concise summary. "
        "Keep key facts, file paths, errors, and tool results. Be brief.\n\n"
        + text_log
    )

    try:
        resp = await llm.chat([LLMMessage(role="user", content=summary_prompt)])
        if not resp.content.strip():
            emit("context", {"message": "compaction returned empty summary — dropping oldest"})
            return [*system_msgs, *recent]
        summary = LLMMessage(
            role="assistant",
            content=f"[Compacted History] {resp.content}",
        )
        emit("context", {"message": f"compacted {len(older)} messages into summary"})
        return [*system_msgs, summary, *recent]
    except Exception:
        logger.debug("Compaction LLM call failed", exc_info=True)
        emit("context", {"message": "compaction failed — dropping oldest messages"})
        return [*system_msgs, *recent]


# ─── History Offloading ───────────────────────────────────────────────────



_HISTORY_DIR = Path.cwd() / ".clawagents" / "history"


def _offload_history(messages: list[LLMMessage]) -> str | None:
    """Save older messages to a JSON file before compaction."""
    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = _HISTORY_DIR / f"compacted_{ts}_{len(messages)}msgs.json"
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


# ─── ReAct Loop ──────────────────────────────────────────────────────────

MAX_TOOL_ROUNDS = 1000


async def run_agent_graph(
    task: str,
    llm: LLMProvider,
    tools: Optional[ToolRegistry] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = MAX_TOOL_ROUNDS,
    streaming: bool = True,
    context_window: int = 128_000,
    on_event: Optional[OnEvent] = None,
    before_llm: Optional[BeforeLLMHook] = None,
    before_tool: Optional[BeforeToolHook] = None,
    after_tool: Optional[AfterToolHook] = None,
    use_native_tools: bool = True,
) -> AgentState:
    """Single ReAct loop: LLM → tools → LLM → tools → ... → final answer."""
    registry = tools or ToolRegistry()
    native_schemas: list[NativeToolSchema] | None = (
        registry.to_native_schemas() if use_native_tools and tools else None
    )
    tool_desc = registry.describe_for_llm() if not use_native_tools else ""
    loop_tracker = _ToolCallTracker()
    emit = on_event or _default_on_event

    token_multiplier = 1.0
    resolved_model_name: Optional[str] = None

    prompt_to_use = system_prompt or BASE_SYSTEM_PROMPT

    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=f"{prompt_to_use}\n\n{tool_desc}"),
        LLMMessage(role="user", content=task),
    ]

    state = AgentState(
        messages=messages,
        current_task=task,
        status="running",
        result="",
        iterations=0,
        max_iterations=max_iterations,
        tool_calls=0,
    )

    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        emit("warn", {"message": "interrupted"})
        cancel_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except (NotImplementedError, OSError):
        pass

    effective_max_rounds = min(
        max_iterations if max_iterations > 0 else MAX_TOOL_ROUNDS,
        MAX_TOOL_ROUNDS,
    )

    t0 = time.monotonic()

    try:
        for round_idx in range(effective_max_rounds):
            if cancel_event.is_set():
                state.status = "done"
                state.result = state.result or "[cancelled]"
                break

            # Patch dangling tool calls before sending to LLM
            messages = _patch_dangling_tool_calls(messages)
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
                    observed_ratio = context_window / max(
                        _estimate_messages_tokens(messages, 1.0), 1,
                    )
                    token_multiplier = min(observed_ratio * 1.1, 3.0)
                    emit("context", {
                        "message": f"token overflow — calibrated multiplier to {token_multiplier:.2f}",
                    })
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
                state.result = response.content
                state.status = "done"
                state.iterations += 1
                emit("final_content", {"content": response.content})
                messages.append(LLMMessage(role="assistant", content=response.content))
                break

            if loop_tracker.is_hard_looping_batch(tool_calls):
                names = ", ".join(c.tool_name for c in tool_calls)
                emit("warn", {"message": f"tool loop detected ({names}) — breaking"})
                state.status = "done"
                state.result = f"Tool loop detected ({names}). Stopping."
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
                    preview = tool_output[:120]

                emit("tool_result", {
                    "name": call.tool_name,
                    "success": tool_result.success,
                    "preview": preview,
                })

                # Use proper tool role messages when native tools are enabled
                if use_native_tools and native_tc and native_tc.tool_call_id:
                    messages.append(LLMMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls_meta=[{"id": native_tc.tool_call_id, "name": call.tool_name, "args": call.args}],
                    ))
                    tool_content = f"{tool_output}" if isinstance(tool_output, str) else json.dumps(tool_output)
                    messages.append(LLMMessage(
                        role="tool",
                        content=tool_content,
                        tool_call_id=native_tc.tool_call_id,
                    ))
                else:
                    messages.append(
                        LLMMessage(role="assistant", content=f'{{"tool": "{call.tool_name}", "args": {json.dumps(call.args)}}}')
                    )
                    user_content = f"[Tool Result] {tool_output}" if isinstance(tool_output, str) else tool_output
                    messages.append(
                        LLMMessage(role="user", content=user_content)
                    )
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
                        preview = output[:120]
                        
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
                        LLMMessage(role="assistant", content=tool_call_str)
                    )
                    messages.append(
                        LLMMessage(
                            role="user",
                            content="[Tool Results]\n" + "\n".join(call_summaries),
                        )
                    )

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
    emit("agent_done", {
        "tool_calls": state.tool_calls,
        "iterations": state.iterations,
        "elapsed": elapsed,
    })
    return state
