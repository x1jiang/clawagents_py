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
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

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
from clawagents.graph.turn_driver import TurnDriver
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
    # Harness may tighten soft/hard loop thresholds (e.g. Luna → warn@2 / stop@3).
    _loop_soft, _loop_hard = 3, 6
    _loop_cfg = None
    try:
        from clawagents.harness_profiles import resolve_harness_profile as _rhp_loop
        from clawagents.loop_detection import LoopDetectionConfig

        _hp_loop = _rhp_loop(getattr(llm, "model", None))
        if _hp_loop and _hp_loop.loop_detection_overrides:
            ov = _hp_loop.loop_detection_overrides
            if ov.get("warning_threshold") is not None:
                _loop_soft = int(ov["warning_threshold"])
            if ov.get("critical_threshold") is not None:
                _loop_hard = int(ov["critical_threshold"])
            _loop_cfg = LoopDetectionConfig(
                warning_threshold=_loop_soft,
                critical_threshold=_loop_hard,
            )
    except Exception:
        pass
    loop_tracker = _ToolCallTracker(
        soft_limit=_loop_soft,
        hard_limit=_loop_hard,
        loop_config=_loop_cfg,
    )
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
    # The lightweight stop-and-classify guard is always valuable. ``rethink``
    # controls optional advisor/learning behavior, not basic loop safety.
    failure_tracker = _FailureTracker(threshold=adaptive_threshold)
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

    # Run-scoped side effects have explicit owners.  The local aliases keep
    # the existing loop code mechanically stable while subsequent extractions
    # move callers to these collaborators directly.
    events = RunEvents(emit, on_stream_event)
    _emit_typed = events.typed
    hook_dispatcher = HookDispatcher(active_hooks, run_context, events)
    _fire_hook = hook_dispatcher.fire
    llm_caller = TurnLLMCaller(
        llm=llm,
        events=events,
        hooks=hook_dispatcher,
        registry=registry,
        session_writer=session_writer,
        external_hooks=ext_hook_runner,
        accumulate_usage=_accumulate_usage,
    )
    response_interpreter = TurnResponseInterpreter(
        llm=llm,
        registry=registry,
        events=events,
    )
    tool_batch_safety = ToolBatchSafety(loop_tracker, events)
    rethink_controller = RethinkController(events)
    tool_call_runner = ToolCallRunner(
        registry=registry,
        tracker=loop_tracker,
        events=events,
        hooks=hook_dispatcher,
        run_context=run_context,
        legacy_on_event=on_event,
    )
    tool_result_processor = ToolResultProcessor(
        external_hooks=ext_hook_runner,
        taxonomy_dispatcher=taxonomy_dispatcher,
        after_tool=after_tool,
        events=events,
        session_writer=session_writer,
        run_context=run_context,
        preview_chars=preview_chars,
    )
    tool_policy_gate = ToolPolicyGate(
        external_hooks=ext_hook_runner,
        taxonomy_dispatcher=taxonomy_dispatcher,
        before_tool=before_tool,
        hook_result_type=HookResult,
        events=events,
    )

    _cached_sys_tokens: int = 0  # Feature D: cache system prompt token count

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

    tool_turn_executor = ToolTurnExecutor(
        registry=registry,
        run_context=run_context,
        events=events,
        policy_gate=tool_policy_gate,
        call_runner=tool_call_runner,
        result_processor=tool_result_processor,
        rethink_controller=rethink_controller,
        loop_tracker=loop_tracker,
        failure_tracker=failure_tracker,
        recorder=recorder,
        session_writer=session_writer,
        require_approval_set=require_approval_set,
        approval_handler=approval_handler,
        use_native_tools=use_native_tools,
        preview_chars=preview_chars,
        task_type=_task_type,
        learn=learn,
        consult_advisor=_consult_advisor,
        llm=llm,
    )
    completion_handler = CompletionHandler(
        registry=registry,
        run_context=run_context,
        events=events,
        recorder=recorder,
        llm=llm,
        before_tool=before_tool,
        action_mode=action_mode_norm,
        looks_like_truncated_json=_looks_like_truncated_json,
        sanitize_assistant_text=_sanitize_assistant_text,
        goal_llm_complete=_goal_llm_complete,
    )

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

    # Session history is an independent concern.  Its journal owns identity
    # tracking so transcript rewrites from compaction never leak into durable
    # conversation history.
    session_journal = SessionMessageJournal(session)
    try:
        messages = await session_journal.preload(
            messages,
            limit=session_preload_limit,
            repair=_patch_dangling_tool_calls,
            drop_leading_orphans=_drop_leading_orphan_tools,
        )
        state.messages = messages
    except Exception as err:
        # A broken backend must not prevent an otherwise valid agent run.
        session_journal.begin(messages)
        emit("warn", {"message": f"session load failed: {err}"})
    _session_initial_ids = session_journal.initial_ids
    handoff_router = HandoffRouter(
        handoffs=handoff_map,
        events=events,
        hooks=hook_dispatcher,
        run_context=run_context,
        from_agent=agent_name or "ClawAgent",
        task=task,
        use_native_tools=use_native_tools,
        session_initial_ids=_session_initial_ids,
        on_stream_event=on_stream_event,
    )
    run_finalizer = RunFinalizer(
        events=events,
        hooks=hook_dispatcher,
        run_context=run_context,
        session_journal=session_journal,
        session_writer=session_writer,
        recorder=recorder,
        llm=llm,
        task=task,
        learn=learn,
        output_guardrails=output_guardrails,
        output_type=output_type,
        run_output_guardrails=_run_output_guardrails,
        coerce_output_type=_coerce_output_type,
        accumulate_usage=_accumulate_usage,
        taxonomy_dispatcher=taxonomy_dispatcher,
        session_end_tail=session_end_tail,
    )
    turn_driver = TurnDriver(
        llm=llm,
        caller=llm_caller,
        events=events,
        run_context=run_context,
        session_journal=session_journal,
        external_hooks=ext_hook_runner,
        before_llm=before_llm,
        fire_hook=_fire_hook,
        taxonomy_dispatcher=taxonomy_dispatcher,
        native_schemas=native_schemas,
        handoffs=handoff_list,
        use_native_tools=use_native_tools,
        tools_supplied=tools is not None,
        streaming=streaming,
        output_type=output_type,
        context_window=context_window,
        resolved_model_name=None,
        cached_system_tokens=_cached_sys_tokens,
        compaction_savings=_compaction_savings,
    )

    def _should_run_final_advisor_check(current_state: AgentState) -> bool:
        return bool(
            advisor_llm is not None
            and _advisor_call_count > 0
            and _advisor_call_count < advisor_max_calls
            and current_state.tool_calls > 0
        )

    round_dispatcher = RoundDispatcher(
        driver=turn_driver,
        response_interpreter=response_interpreter,
        completion_handler=completion_handler,
        handoff_router=handoff_router,
        safety=tool_batch_safety,
        tool_executor=tool_turn_executor,
        run_context=run_context,
        use_native_tools=use_native_tools,
        consult_advisor=_consult_advisor,
        should_final_check=_should_run_final_advisor_check,
    )

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
    round_scheduler = RoundScheduler(
        run_context=run_context,
        events=events,
        session_writer=session_writer,
        timeout_s=timeout_s,
        started_at=t0,
    )

    try:
        for round_idx in range(effective_max_rounds):
            scheduled = await round_scheduler.begin(
                state,
                messages,
                round_index=round_idx,
                cancel_event=cancel_event,
            )
            messages = scheduled.messages
            if scheduled.action == "stop":
                break

            # ── Advisor: consult after initial orientation (first tool results in transcript)
            if advisor_llm and round_idx == 1 and _advisor_call_count == 0:
                await _consult_advisor(messages, "planning")

            dispatched = await round_dispatcher.dispatch(
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

    return await run_finalizer.finalize(state, messages, elapsed=elapsed)
