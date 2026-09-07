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

# This module is a compatibility facade: many helpers were extracted into
# sibling ``graph/*`` modules but are still imported from here by tests and
# downstream integrations, so unused-import checks are disabled file-wide.
# ruff: noqa: F401

import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from clawagents.providers.llm import LLMProvider, LLMMessage, LLMResponse, NativeToolSchema
from clawagents.tools.registry import ToolRegistry, ParsedToolCall, ToolResult
from clawagents.run_context import RunContext
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
)
from clawagents.handoffs import Handoff
from clawagents.prompts import append_model_identity, build_system_prompt

logger = logging.getLogger(__name__)


# ─── Message Repair (extracted to graph/message_repair.py) ─────────────────
from clawagents.graph.message_repair import (
    _sanitize_assistant_text,
    _patch_dangling_tool_calls,
    _drop_leading_orphan_tools,
)


# ─── Tool Observation (extracted to graph/tool_observation.py) ─────────────
from clawagents.graph.tool_observation import (
    _evict_large_tool_result,
    _format_failed_exec_observation,
    # Compatibility re-exports for integrations that historically imported
    # these helpers from ``agent_loop`` before the observation module split.
    _post_tool_side_effects,
    _run_context_workspace,
    _tool_observation,
    _ui_tool_result_text,
    UI_TOOL_RESULT_CHARS,
    _CHARS_PER_TOKEN,
    _estimate_tokens,
    _truncate_old_tool_args,
    _wait_for_tool_approval,
)
from clawagents.graph.run_runtime import (
    HookDispatcher,
    RunEvents,
    SessionMessageJournal,
    session_get_items as _session_get_items,
)
from clawagents.graph.run_config import AgentRunConfig
from clawagents.graph.turn_llm import TurnLLMCaller
from clawagents.graph.turn_response import TurnResponseInterpreter
from clawagents.graph.handoff_router import HandoffRouter
from clawagents.graph.run_finalizer import RunFinalizer
from clawagents.graph.tool_turn import ToolTurnExecutor
from clawagents.graph.completion_handler import CompletionHandler
from clawagents.graph.turn_driver import IncrementalTokenLedger, TurnDriver
from clawagents.graph.round_dispatcher import RoundDispatcher
from clawagents.graph.round_scheduler import RoundScheduler
from clawagents.graph.tool_batch import (
    RethinkController,
    ToolBatchSafety,
    ToolCallRunner,
    ToolPolicyGate,
    ToolResultProcessor,
)


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


# Built-in base prompt now lives in clawagents.prompts.base (configurable via
# base_prompt=, CLAW_BASE_PROMPT[_FILE], .clawagents/base-prompt.md). This name
# is kept as a re-export for callers that imported it from here.
from clawagents.prompts.base import DEFAULT_BASE_SYSTEM_PROMPT as BASE_SYSTEM_PROMPT


# (Token estimation + arg truncation imported above from tool_observation)


# ─── Loop Detection (extracted to graph/loop_tracker.py) ───────────────────
from clawagents.graph.loop_tracker import (
    _ToolCallTracker,
    _FailureTracker,
    _RETHINK_THRESHOLD,
)


# ─── Context Management (extracted to graph/context_management.py) ─────────
from clawagents.graph.context_management import (
    _preflight_context_check,
    _COMPACTABLE_TOOLS,
    _extract_artifact_id,
    _micro_compact_stub,
    # Compatibility re-exports for callers that imported context helpers
    # from ``agent_loop`` before their extraction into context_management.
    _soft_trim_messages,
    _CONTEXT_BUDGET_RATIO,
    _find_safe_split_index,
    _content_key_text,
    _message_reuse_key,
    _reuse_messages_where_possible,
    _goal_llm_complete,
    _offload_history,
    _compact_if_needed,
    _looks_like_truncated_json,
)


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
    config = AgentRunConfig(
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
        features=features,
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
    if features is not None:
        from clawagents.config.features import temporary_overrides

        with temporary_overrides(features):
            return await _run_agent_graph_core(**config.core_kwargs())
    return await _run_agent_graph_core(**config.core_kwargs())


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
    """Internal ReAct loop body (feature overrides applied by :func:`run_agent_graph`).

    All initialization is delegated to :class:`RunBootstrapper`.  This
    function constructs a config, bootstraps, and hands off to the pure
    loop executor.
    """
    from .run_bootstrapper import RunBootstrapper, _bind_agent_loop_refs

    _bind_agent_loop_refs()

    config = AgentRunConfig(
        task=task, llm=llm, tools=tools, system_prompt=system_prompt,
        max_iterations=max_iterations, streaming=streaming,
        context_window=context_window, on_event=on_event,
        before_llm=before_llm, before_tool=before_tool, after_tool=after_tool,
        use_native_tools=use_native_tools, trajectory=trajectory,
        rethink=rethink, learn=learn, atlas=atlas, atlas_config=atlas_config,
        preview_chars=preview_chars, response_chars=response_chars,
        timeout_s=timeout_s, features=features, advisor_llm=advisor_llm,
        advisor_max_calls=advisor_max_calls, run_context=run_context,
        user_context=user_context, hooks=hooks, agent_hooks=agent_hooks,
        input_guardrails=input_guardrails, output_guardrails=output_guardrails,
        output_type=output_type, on_stream_event=on_stream_event,
        session=session, session_preload_limit=session_preload_limit,
        handoffs=handoffs, agent_name=agent_name, action_mode=action_mode,
        approval_handler=approval_handler,
        require_approval_tools=require_approval_tools,
        image_blocks=image_blocks, file_blocks=file_blocks,
        session_end_tail=session_end_tail,
    )
    rs = await RunBootstrapper(config).bootstrap()
    try:
        return await _execute_loop(rs)
    finally:
        rs.completion_handler.restore_output_budget()


async def _execute_loop(rs: Any) -> AgentState:
    """Pure ReAct control loop — no initialization, only control flow.

    ``rs`` is a :class:`RunSession` from the bootstrapper.
    """
    state = rs.state
    messages = rs.messages
    emit = rs.emit
    events = rs.events
    run_context = rs.run_context
    advisor = rs.advisor

    # RunHooks: on_run_start
    if rs.active_hooks:
        await rs.hook_dispatcher.fire("on_run_start", rs.task)
    events.typed("turn_started", {"iteration": 0, "task": rs.task})

    # Input guardrails (short-circuit before the first LLM call).
    if rs.input_guardrails:
        try:
            tripped = await _run_input_guardrails(
                rs.input_guardrails, run_context, rs.task,
            )
        except GuardrailTripwireTriggered as tripwire:
            state.status = "done"
            state.result = (
                tripwire.result.message
                or f"Input rejected by guardrail '{tripwire.guardrail_name}'"
            )
            state.guardrail_triggered = tripwire.guardrail_name
            events.typed("guardrail_tripped", {
                "guardrail_name": tripwire.guardrail_name,
                "where": "input",
                "behavior": tripwire.result.behavior.value,
                "message": state.result,
            })
            emit("warn", {"message": f"input guardrail tripped: {tripwire.guardrail_name}"})
            if rs.active_hooks:
                await rs.hook_dispatcher.fire("on_run_end", state.result)
            return state
        if tripped:
            messages.append(LLMMessage(role="user", content=tripped))
            events.typed("guardrail_tripped", {
                "guardrail_name": "input",
                "where": "input",
                "behavior": "reject_content",
                "message": tripped,
                "stage": "input",
                "rewrite": True,
            })

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

    t0 = time.monotonic()

    try:
        for round_idx in range(rs.max_rounds):
            scheduled = await rs.scheduler.begin(
                state,
                messages,
                round_index=round_idx,
                cancel_event=cancel_event,
            )
            messages = scheduled.messages
            if scheduled.action == "stop":
                break

            # ── Advisor: consult after initial orientation (first tool results in transcript)
            if advisor.available and round_idx == 1 and advisor.call_count == 0:
                await advisor.consult(messages, "planning")

            dispatched = await rs.dispatcher.dispatch(
                state,
                messages,
                round_index=round_idx,
                cancel_event=cancel_event,
            )
            messages = dispatched.messages
            if dispatched.action == "handoff":
                child_state = dispatched.child_state
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
            if dispatched.action == "stop":
                break

        else:
            emit("warn", {"message": f"reached max {rs.max_rounds} tool rounds"})
            # Act-invariant reconciliation gate: if state remains uncertain
            # even though the round budget is exhausted, report max_iterations
            # instead of done so the caller knows the run didn't finish cleanly.
            try:
                from clawagents.permissions.act_invariants import completion_block_reason

                _block = completion_block_reason(run_context)
            except Exception:
                _block = None
            if _block:
                state.status = "max_iterations"
                state.result = _block
            else:
                state.status = "done"
                state.result = state.result or f"Reached maximum of {rs.max_rounds} tool rounds."

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

    return await rs.finalizer.finalize(state, messages, elapsed=elapsed)
