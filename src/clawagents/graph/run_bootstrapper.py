"""Agent run bootstrapper — converts configuration into a ready-to-run session.

Extracts the ~700-line linear initialization from ``_run_agent_graph_core``
into ordered, independently testable phases:

    resolve_config → init_runtime → build_messages → load_session → wire_collaborators

The result is a ``RunSession`` dataclass that the loop consumes without any
initialisation concerns.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional

from clawagents.guardrails import OutputGuardrail
from clawagents.handoffs import Handoff
from clawagents.lifecycle import RunHooks
from clawagents.prompts import append_model_identity, build_system_prompt
from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse, NativeToolSchema
from clawagents.run_context import RunContext
from clawagents.stream_events import StreamEvent
from clawagents.tools.registry import ToolRegistry
from clawagents.usage import RequestUsage, Usage

from .completion_handler import CompletionHandler
from .context_layers import build_default_layers, collect_dynamic_context
from .context_management import (
    _looks_like_truncated_json,
    _preflight_context_check,
    _goal_llm_complete,
)
from .handoff_router import HandoffRouter
from .loop_tracker import _FailureTracker, _RETHINK_THRESHOLD, _ToolCallTracker
from .message_repair import (
    _drop_leading_orphan_tools,
    _patch_dangling_tool_calls,
    _sanitize_assistant_text,
)
from .round_dispatcher import RoundDispatcher
from .round_scheduler import RoundScheduler
from .run_config import AgentRunConfig
from .run_finalizer import RunFinalizer
from .run_runtime import HookDispatcher, RunEvents, SessionMessageJournal
from .tool_batch import (
    RethinkController,
    ToolBatchSafety,
    ToolCallRunner,
    ToolPolicyGate,
    ToolResultProcessor,
)
from .tool_observation import _estimate_messages_tokens, _estimate_tokens
from .tool_turn import ToolTurnExecutor
from .turn_driver import IncrementalTokenLedger, TurnDriver
from .turn_llm import TurnLLMCaller
from .turn_response import TurnResponseInterpreter

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 1000

# Imported lazily in the agent_loop module — keep for the default event handler.
from .run_config import AgentRunConfig  # noqa: F811 (intentional re-import guard)


# ── Advisor Controller ──────────────────────────────────────────────────


class AdvisorController:
    """Encapsulates advisor LLM consultation state (replaces closure + nonlocal)."""

    def __init__(
        self,
        llm: LLMProvider | None,
        max_calls: int,
        emit: Callable[..., None],
    ) -> None:
        self.llm = llm
        self.max_calls = max_calls
        self.call_count = 0
        self._emit = emit

    @property
    def available(self) -> bool:
        return self.llm is not None

    async def consult(self, msgs: list[LLMMessage], trigger: str) -> None:
        if not self.llm or self.call_count >= self.max_calls:
            return
        self.call_count += 1
        self._emit(
            "context",
            {"message": f"advisor consultation #{self.call_count} ({trigger})"},
        )
        try:
            advisor_response = await self.llm.chat([
                LLMMessage(
                    role="system",
                    content=(
                        "You are a senior advisor. Review the agent's full transcript "
                        "and provide concise strategic guidance. Under 150 words. "
                        "Use numbered steps, not explanations."
                    ),
                ),
                *msgs,
                LLMMessage(
                    role="user",
                    content=(
                        f"[Advisor Request — {trigger}] Review the conversation "
                        "above and provide strategic guidance for the next steps."
                    ),
                ),
            ])
            if advisor_response.content:
                msgs.append(
                    LLMMessage(
                        role="user",
                        content=f"[Advisor Guidance]\n{advisor_response.content}",
                    )
                )
                self._emit(
                    "context",
                    {"message": f"advisor: {advisor_response.content[:120]}..."},
                )
        except Exception as err:
            self._emit("warn", {"message": f"advisor consultation failed: {err}"})

    def should_final_check(self, state: Any) -> bool:
        return bool(
            self.llm is not None
            and self.call_count > 0
            and self.call_count < self.max_calls
            and state.tool_calls > 0
        )


# ── Run Session (loop contract) ─────────────────────────────────────────


@dataclass
class RunSession:
    """Everything the agent loop needs to run — no initialization concerns."""

    # Core state
    state: Any  # AgentState (forward-ref safe)
    messages: list[LLMMessage]
    task: str

    # Runtime
    events: RunEvents
    emit: Callable[..., None]
    run_context: RunContext
    usage: Usage

    # Hooks
    hook_dispatcher: HookDispatcher
    active_hooks: list[RunHooks]

    # Collaborators
    scheduler: RoundScheduler
    dispatcher: RoundDispatcher
    finalizer: RunFinalizer
    advisor: AdvisorController

    # Config
    max_rounds: int
    input_guardrails: list[Any] | None


# ── Bootstrapper ─────────────────────────────────────────────────────────


class RunBootstrapper:
    """Converts an ``AgentRunConfig`` into a ready-to-run ``RunSession``.

    Phases execute in dependency order:

    1. ``_resolve_config``    — normalize params, build registry/schemas/loop tracker
    2. ``_init_runtime``      — RunContext, usage, session_id, hooks, events
    3. ``_init_infrastructure`` — recorder, session persistence, external hooks, taxonomy
    4. ``_wire_events``       — emit wrapper, RunEvents, HookDispatcher
    5. ``_build_messages``    — system prompt, context layers, preflight, token ledger
    6. ``_load_session``      — session journal, history preload, rewind snapshot
    7. ``_wire_collaborators`` — all 20+ collaborator objects
    """

    def __init__(self, config: AgentRunConfig) -> None:
        self.c = config  # shorthand for the config

        # Populated by phases — typed as Optional so each phase is explicit.
        self._registry: ToolRegistry = ToolRegistry()
        self._native_schemas: list[NativeToolSchema] | None = None
        self._tool_desc: str = ""
        self._handoff_list: list[Handoff] = []
        self._handoff_map: dict[str, Handoff] = {}
        self._loop_tracker: _ToolCallTracker | None = None
        self._failure_tracker: _FailureTracker | None = None
        self._base_emit: Callable[..., None] | None = None
        self._emit: Callable[..., None] | None = None
        self._run_context: RunContext | None = None
        self._usage: Usage | None = None
        self._provider_session_id: str = ""
        self._active_hooks: list[RunHooks] = []
        self._recorder: Any = None
        self._session_writer: Any = None
        self._ext_hook_runner: Any = None
        self._taxonomy_dispatcher: Any = None
        self._events: RunEvents | None = None
        self._hook_dispatcher: HookDispatcher | None = None
        self._session_journal: SessionMessageJournal | None = None
        self._task_type: str = "general"
        self._compaction_savings: list[float] = []
        self._cached_sys_tokens: int = 0
        self._token_ledger: IncrementalTokenLedger | None = None
        self._action_mode_norm: str = "tools"
        self._require_approval_set: set[str] = set()

    async def bootstrap(self) -> RunSession:
        """Run all phases and return a fully initialised RunSession."""
        # Import AgentState here to avoid circular imports at module level.
        from .agent_loop import AgentState

        self._resolve_config()
        await self._init_runtime()
        self._init_infrastructure()
        self._wire_events()
        messages = self._build_messages()
        messages = await self._load_session(messages)

        state = AgentState(
            messages=messages,
            current_task=self.c.task,
            status="running",
            result="",
            iterations=0,
            max_iterations=self.c.max_iterations,
            tool_calls=0,
            usage=self._usage,
            run_context=self._run_context,
        )

        await self._fire_taxonomy_start()
        self._snapshot_rewind(messages)

        collaborators = self._wire_collaborators(messages, state)
        advisor = collaborators["advisor"]

        effective_max_rounds = min(
            self.c.max_iterations if self.c.max_iterations > 0 else MAX_TOOL_ROUNDS,
            MAX_TOOL_ROUNDS,
        )

        return RunSession(
            state=state,
            messages=messages,
            task=self.c.task,
            events=self._events,
            emit=self._emit,
            run_context=self._run_context,
            usage=self._usage,
            hook_dispatcher=self._hook_dispatcher,
            active_hooks=self._active_hooks,
            scheduler=collaborators["scheduler"],
            dispatcher=collaborators["dispatcher"],
            finalizer=collaborators["finalizer"],
            advisor=advisor,
            max_rounds=effective_max_rounds,
            input_guardrails=self.c.input_guardrails,
        )

    # ── Phase 1: Config Resolution ──────────────────────────────────

    def _resolve_config(self) -> None:
        c = self.c
        self._registry = c.tools or ToolRegistry()
        self._action_mode_norm = (
            c.action_mode if c.action_mode in ("tools", "code") else "tools"
        )
        self._require_approval_set = {
            n for n in (c.require_approval_tools or []) if n
        }
        if c.approval_handler is not None:
            from clawagents.permissions.mode import WRITE_CLASS_TOOLS

            self._require_approval_set |= set(WRITE_CLASS_TOOLS)

        self._native_schemas = (
            self._registry.to_native_schemas()
            if c.use_native_tools and c.tools
            else None
        )
        self._tool_desc = (
            self._registry.describe_for_llm() if not c.use_native_tools else ""
        )

        # Loop detection thresholds
        _loop_soft, _loop_hard = 3, 6
        _loop_cfg = None
        try:
            from clawagents.harness_profiles import resolve_harness_profile
            from clawagents.loop_detection import LoopDetectionConfig

            hp = resolve_harness_profile(getattr(c.llm, "model", None))
            if hp and hp.loop_detection_overrides:
                ov = hp.loop_detection_overrides
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
        self._loop_tracker = _ToolCallTracker(
            soft_limit=_loop_soft,
            hard_limit=_loop_hard,
            loop_config=_loop_cfg,
        )
        self._base_emit = c.on_event or _default_on_event

        # Handoffs
        self._handoff_list = list(c.handoffs) if c.handoffs else []
        self._handoff_map = {h.name: h for h in self._handoff_list}
        if self._handoff_list:
            handoff_params = {
                "reason": {
                    "type": "string",
                    "description": "Free-text rationale for why the handoff is appropriate.",
                    "required": False,
                }
            }
            if c.use_native_tools:
                if self._native_schemas is None:
                    self._native_schemas = []
                for h in self._handoff_list:
                    self._native_schemas.append(
                        NativeToolSchema(
                            name=h.name,
                            description=h.description,
                            parameters=handoff_params,
                        )
                    )
            else:
                extra_lines = ["", "## Handoffs"]
                for h in self._handoff_list:
                    extra_lines.append(f"### {h.name}\n{h.description}")
                    extra_lines.append("Parameters:")
                    extra_lines.append(
                        "- `reason` (string): Free-text rationale."
                    )
                    extra_lines.append("")
                self._tool_desc = (self._tool_desc or "") + "\n" + "\n".join(
                    extra_lines
                )

    # ── Phase 2: Runtime Init ────────────────────────────────────────

    async def _init_runtime(self) -> None:
        c = self.c
        run_context = c.run_context
        if run_context is None:
            run_context = RunContext(context=c.user_context)
        elif c.user_context is not None and run_context.context is None:
            run_context.context = c.user_context

        # Provider session ID resolution
        import uuid as _uuid

        _meta_sid = run_context._metadata.get(
            "session_id"
        ) or run_context._metadata.get("sessionId")
        if getattr(c.session, "session_id", None):
            self._provider_session_id = str(c.session.session_id)
        elif getattr(run_context, "session_id", None):
            self._provider_session_id = str(run_context.session_id)
        elif isinstance(_meta_sid, str) and _meta_sid:
            self._provider_session_id = _meta_sid
        else:
            self._provider_session_id = f"run-{_uuid.uuid4().hex[:12]}"
        run_context.session_id = self._provider_session_id
        run_context._metadata["session_id"] = self._provider_session_id
        run_context._metadata["sessionId"] = self._provider_session_id
        self._usage = run_context.usage

        # Iteration budget
        _budget_size = (
            c.max_iterations if c.max_iterations > 0 else MAX_TOOL_ROUNDS
        )
        await run_context.ensure_iteration_budget(_budget_size)
        run_context.on_event = self._base_emit

        # Hooks
        self._active_hooks = []
        if c.hooks is not None:
            self._active_hooks.append(c.hooks)
        if c.agent_hooks is not None and c.agent_hooks is not c.hooks:
            self._active_hooks.append(c.agent_hooks)
        run_context._metadata["hooks"] = self._active_hooks
        run_context._metadata["agent_name"] = c.agent_name or "ClawAgent"

        # Rethink / task type detection
        if c.rethink or c.learn:
            try:
                from clawagents.trajectory.verifier import (
                    compute_adaptive_rethink_threshold,
                    detect_task_type,
                )

                self._task_type = detect_task_type(c.task)
                adaptive_threshold = compute_adaptive_rethink_threshold(
                    self._task_type, 0, 0
                )
            except Exception:
                adaptive_threshold = _RETHINK_THRESHOLD
        else:
            adaptive_threshold = _RETHINK_THRESHOLD
        self._failure_tracker = _FailureTracker(threshold=adaptive_threshold)

        self._run_context = run_context

    # ── Phase 3: Infrastructure ──────────────────────────────────────

    def _init_infrastructure(self) -> None:
        c = self.c
        run_context = self._run_context

        # Trajectory recorder
        if c.trajectory or c.learn:
            from clawagents.trajectory.recorder import TrajectoryRecorder

            self._recorder = TrajectoryRecorder(
                task=c.task, response_chars=c.response_chars
            )

        # Bind workspace + goal LLM for tools / final gate
        meta = run_context._metadata
        if not isinstance(meta.get("workspace"), str):
            meta["workspace"] = os.getcwd()
        if getattr(self._registry, "_permission_engine", None) is not None:
            meta.setdefault("permission_engine", self._registry._permission_engine)
        if c.before_tool is not None:
            meta["before_tool"] = c.before_tool
        if c.approval_handler is not None:
            meta["approval_handler"] = c.approval_handler

        llm = c.llm

        async def _bound_goal_llm(prompt: str) -> str:
            resp = await llm.chat([LLMMessage(role="user", content=prompt)])
            return str(getattr(resp, "content", "") or "")

        meta["goal_llm_complete"] = _bound_goal_llm

        try:
            from clawagents.config.features import is_enabled
            from clawagents.goal import (
                GoalTracker,
                attach_goal_to_run_context,
                get_goal_tracker,
            )

            _want_goal = bool(meta.get("goal_mode"))
            if (
                _want_goal
                and is_enabled("goal_autopilot")
                and get_goal_tracker(run_context) is None
            ):
                attach_goal_to_run_context(
                    run_context, GoalTracker(meta["workspace"])
                )
        except Exception:
            logger.debug("goal tracker bind failed", exc_info=True)

        # Session persistence
        from clawagents.config.features import is_enabled as _feat_enabled

        if _feat_enabled("session_persistence"):
            from clawagents.session.persistence import SessionWriter

            self._session_writer = SessionWriter()
            run_context.session_id = self._session_writer.session_id
            run_context._metadata["session_id"] = self._session_writer.session_id
            self._base_emit(
                "context",
                {
                    "message": (
                        f"session: {self._session_writer.session_id}"
                        f" → {self._session_writer.path}"
                    )
                },
            )

        # External hooks
        hooks_cfg = None
        if _feat_enabled("external_hooks"):
            from clawagents.hooks.external import (
                ExternalHookRunner,
                load_hooks_config,
            )

            hooks_cfg = load_hooks_config()
            if hooks_cfg:
                self._ext_hook_runner = ExternalHookRunner(hooks_cfg)
                self._base_emit("context", {"message": "external hooks: loaded"})

        # Taxonomy dispatcher
        try:
            from clawagents.hooks.external import build_taxonomy_dispatcher

            self._taxonomy_dispatcher = build_taxonomy_dispatcher(hooks_cfg)
            if self._taxonomy_dispatcher is not None:
                self._base_emit("context", {"message": "hook taxonomy: loaded"})
        except Exception:
            logger.debug("hook taxonomy load failed", exc_info=True)

        if self._taxonomy_dispatcher is not None and isinstance(
            getattr(run_context, "_metadata", None), dict
        ):
            run_context._metadata["taxonomy_dispatcher"] = self._taxonomy_dispatcher

    # ── Phase 4: Wire Events ─────────────────────────────────────────

    def _wire_events(self) -> None:
        _base_emit = self._base_emit
        taxonomy_dispatcher = self._taxonomy_dispatcher

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

        def emit(kind: str, data: dict[str, Any] | None = None) -> None:
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

        self._emit = emit
        self._events = RunEvents(emit, self.c.on_stream_event)
        self._run_context._metadata["_emit_typed_event"] = self._events.typed
        self._hook_dispatcher = HookDispatcher(
            self._active_hooks, self._run_context, self._events
        )

    # ── Phase 5: Build Messages ──────────────────────────────────────

    def _build_messages(self) -> list[LLMMessage]:
        c = self.c
        prompt_to_use = append_model_identity(
            c.system_prompt or _default_base_prompt(),
            getattr(c.llm, "name", None),
            getattr(c.llm, "model", None),
        )

        # Dynamic context layers
        context_layers = build_default_layers(learn=c.learn)
        lesson_preamble = collect_dynamic_context(
            context_layers, self._run_context, emit=self._emit,
        )

        system_content = build_system_prompt(
            base_prompt=prompt_to_use,
            tool_description=self._tool_desc,
            lesson_preamble=lesson_preamble,
        )

        # Multimodal blocks
        if c.image_blocks or c.file_blocks:
            first_user_content: Any = (
                ([{"type": "text", "text": c.task}] if c.task else [])
                + list(c.image_blocks or [])
                + list(c.file_blocks or [])
            )
        else:
            first_user_content = c.task

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=first_user_content),
        ]

        if self._session_writer:
            self._session_writer.write_system_prompt(system_content)
            # The prompt arrives as the ``task`` argument, so nothing else in
            # the run will record it. A log without it cannot be resumed: the
            # reader has no user turn to reconstruct, and ``get_task()``
            # returns "".
            self._session_writer.write_user_message(first_user_content)

        # Preflight context check
        messages, self._tool_desc, self._native_schemas = _preflight_context_check(
            messages,
            c.context_window,
            self._tool_desc,
            self._native_schemas,
            self._registry,
            self._emit,
        )

        # Cache system prompt tokens
        if messages:
            self._cached_sys_tokens = _estimate_tokens(messages[0].content)
            self._emit(
                "context",
                {
                    "message": (
                        f"system prompt: ~{self._cached_sys_tokens} tokens"
                        " (cached for budget calc)"
                    )
                },
            )

        # Token ledger
        self._token_ledger = IncrementalTokenLedger(
            partial(
                _estimate_messages_tokens,
                model=None,
                cached_system_tokens=self._cached_sys_tokens,
            )
        )
        self._token_ledger.rebase(messages)

        return messages

    # ── Phase 6: Load Session History ────────────────────────────────

    async def _load_session(
        self, messages: list[LLMMessage]
    ) -> list[LLMMessage]:
        self._session_journal = SessionMessageJournal(self.c.session)
        try:
            messages = await self._session_journal.preload(
                messages,
                limit=self.c.session_preload_limit,
                repair=_patch_dangling_tool_calls,
                drop_leading_orphans=_drop_leading_orphan_tools,
            )
        except Exception as err:
            self._session_journal.begin(messages)
            self._emit("warn", {"message": f"session load failed: {err}"})
        return messages

    # ── Taxonomy start hooks ─────────────────────────────────────────

    async def _fire_taxonomy_start(self) -> None:
        if self._taxonomy_dispatcher is None:
            return
        try:
            from clawagents.hooks.external import dispatch_taxonomy_hook
            from clawagents.hooks.taxonomy import HookEvent

            task = self.c.task
            await dispatch_taxonomy_hook(
                self._taxonomy_dispatcher,
                HookEvent.SESSION_START,
                {"task": task[:500] if task else ""},
                blocking=False,
            )
            await dispatch_taxonomy_hook(
                self._taxonomy_dispatcher,
                HookEvent.USER_PROMPT_SUBMIT,
                {"prompt": task[:2000] if task else ""},
                blocking=False,
            )
        except Exception:
            logger.debug("taxonomy session_start hook failed", exc_info=True)

    # ── Session rewind snapshot ──────────────────────────────────────

    def _snapshot_rewind(self, messages: list[LLMMessage]) -> None:
        try:
            from clawagents.config.features import is_enabled

            if not (is_enabled("session_rewind") or is_enabled("hunk_watcher")):
                return
            from clawagents.memory.hunk_watcher import get_watcher

            run_context = self._run_context
            _ws = None
            if isinstance(run_context._metadata, dict):
                _ws = run_context._metadata.get("workspace")
            w = get_watcher(_ws)
            meta = (
                run_context._metadata
                if isinstance(run_context._metadata, dict)
                else None
            )
            idx = int((meta or {}).get("prompt_index") or 0) + 1
            idx = max(idx, int(getattr(w, "_prompt_index", 0) or 0) + 1)
            if meta is not None:
                meta["prompt_index"] = idx
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
                user_text=(self.c.task or "")[:2000],
                message_count=len(messages),
                conversation_marker=_conv_marker,
            )
        except Exception:
            logger.debug("rewind snapshot failed", exc_info=True)

    # ── Phase 7: Wire Collaborators ──────────────────────────────────

    def _wire_collaborators(
        self, messages: list[LLMMessage], state: Any
    ) -> dict[str, Any]:
        c = self.c
        events = self._events
        hook_dispatcher = self._hook_dispatcher
        run_context = self._run_context
        session_journal = self._session_journal

        # Usage accumulator (closure over usage + events)
        usage = self._usage
        events_ref = events

        def _accumulate_usage(
            resp: LLMResponse,
            *,
            time_to_first_token_ms: float | None = None,
            peak_memory_bytes: int = 0,
        ) -> RequestUsage:
            prompt_t = int(getattr(resp, "prompt_tokens", 0) or 0)
            total_t = int(getattr(resp, "tokens_used", 0) or 0)
            output_t = int(
                getattr(resp, "completion_tokens", max(total_t - prompt_t, 0)) or 0
            )
            cache_read_t = int(getattr(resp, "cache_read_tokens", 0) or 0)
            cache_write_t = int(getattr(resp, "cache_creation_tokens", 0) or 0)
            uncached_input_t = int(
                getattr(
                    resp,
                    "uncached_input_tokens",
                    max(prompt_t - cache_read_t - cache_write_t, 0),
                )
                or 0
            )
            req = usage.add_response(
                model=getattr(resp, "model", None) or "",
                prompt_tokens=prompt_t,
                input_tokens=uncached_input_t,
                output_tokens=output_t,
                total_tokens=total_t,
                cached_input_tokens=cache_read_t,
                cache_creation_tokens=cache_write_t,
                time_to_first_token_ms=time_to_first_token_ms,
                peak_memory_bytes=peak_memory_bytes,
            )
            events_ref.typed(
                "usage",
                {
                    "prompt_tokens": req.prompt_tokens,
                    "input_tokens": req.input_tokens,
                    "output_tokens": req.output_tokens,
                    "total_tokens": req.total_tokens,
                    "cached_input_tokens": req.cached_input_tokens,
                    "cache_creation_tokens": req.cache_creation_tokens,
                    "time_to_first_token_ms": req.time_to_first_token_ms,
                    "peak_memory_bytes": usage.peak_memory_bytes,
                    "model": req.model,
                },
            )
            return req

        llm_caller = TurnLLMCaller(
            llm=c.llm,
            events=events,
            hooks=hook_dispatcher,
            registry=self._registry,
            session_writer=self._session_writer,
            external_hooks=self._ext_hook_runner,
            accumulate_usage=_accumulate_usage,
            provider_session_id=self._provider_session_id,
        )
        response_interpreter = TurnResponseInterpreter(
            llm=c.llm,
            registry=self._registry,
            events=events,
        )
        tool_batch_safety = ToolBatchSafety(self._loop_tracker, events)
        rethink_controller = RethinkController(events)
        tool_call_runner = ToolCallRunner(
            registry=self._registry,
            tracker=self._loop_tracker,
            events=events,
            hooks=hook_dispatcher,
            run_context=run_context,
            legacy_on_event=c.on_event,
        )
        tool_result_processor = ToolResultProcessor(
            external_hooks=self._ext_hook_runner,
            taxonomy_dispatcher=self._taxonomy_dispatcher,
            after_tool=c.after_tool,
            events=events,
            session_writer=self._session_writer,
            run_context=run_context,
            preview_chars=c.preview_chars,
        )
        tool_policy_gate = ToolPolicyGate(
            external_hooks=self._ext_hook_runner,
            taxonomy_dispatcher=self._taxonomy_dispatcher,
            before_tool=c.before_tool,
            hook_result_type=_HookResult,
            events=events,
        )

        advisor = AdvisorController(
            llm=c.advisor_llm,
            max_calls=c.advisor_max_calls,
            emit=self._emit,
        )

        tool_turn_executor = ToolTurnExecutor(
            registry=self._registry,
            run_context=run_context,
            events=events,
            policy_gate=tool_policy_gate,
            call_runner=tool_call_runner,
            result_processor=tool_result_processor,
            rethink_controller=rethink_controller,
            loop_tracker=self._loop_tracker,
            failure_tracker=self._failure_tracker,
            recorder=self._recorder,
            session_writer=self._session_writer,
            require_approval_set=self._require_approval_set,
            approval_handler=c.approval_handler,
            use_native_tools=c.use_native_tools,
            preview_chars=c.preview_chars,
            task_type=self._task_type,
            learn=c.learn,
            consult_advisor=advisor.consult,
            llm=c.llm,
        )
        completion_handler = CompletionHandler(
            registry=self._registry,
            run_context=run_context,
            events=events,
            recorder=self._recorder,
            llm=c.llm,
            before_tool=c.before_tool,
            action_mode=self._action_mode_norm,
            looks_like_truncated_json=_looks_like_truncated_json,
            sanitize_assistant_text=_sanitize_assistant_text,
            goal_llm_complete=_goal_llm_complete,
        )

        _session_initial_ids = session_journal.initial_ids
        handoff_router = HandoffRouter(
            handoffs=self._handoff_map,
            events=events,
            hooks=hook_dispatcher,
            run_context=run_context,
            from_agent=c.agent_name or "ClawAgent",
            task=c.task,
            use_native_tools=c.use_native_tools,
            session_initial_ids=_session_initial_ids,
            on_stream_event=c.on_stream_event,
        )
        run_finalizer = RunFinalizer(
            events=events,
            hooks=hook_dispatcher,
            run_context=run_context,
            session_journal=session_journal,
            session_writer=self._session_writer,
            recorder=self._recorder,
            llm=c.llm,
            task=c.task,
            learn=c.learn,
            output_guardrails=c.output_guardrails,
            output_type=c.output_type,
            run_output_guardrails=_run_output_guardrails_fn,
            coerce_output_type=_coerce_output_type_fn,
            accumulate_usage=_accumulate_usage,
            taxonomy_dispatcher=self._taxonomy_dispatcher,
            session_end_tail=c.session_end_tail,
        )
        turn_driver = TurnDriver(
            llm=c.llm,
            caller=llm_caller,
            events=events,
            run_context=run_context,
            session_journal=session_journal,
            external_hooks=self._ext_hook_runner,
            before_llm=c.before_llm,
            fire_hook=hook_dispatcher.fire,
            taxonomy_dispatcher=self._taxonomy_dispatcher,
            native_schemas=self._native_schemas,
            handoffs=self._handoff_list,
            use_native_tools=c.use_native_tools,
            tools_supplied=c.tools is not None,
            streaming=c.streaming,
            output_type=c.output_type,
            context_window=c.context_window,
            resolved_model_name=None,
            cached_system_tokens=self._cached_sys_tokens,
            compaction_savings=self._compaction_savings,
            token_ledger=self._token_ledger,
        )
        round_dispatcher = RoundDispatcher(
            driver=turn_driver,
            response_interpreter=response_interpreter,
            completion_handler=completion_handler,
            handoff_router=handoff_router,
            safety=tool_batch_safety,
            tool_executor=tool_turn_executor,
            run_context=run_context,
            use_native_tools=c.use_native_tools,
            consult_advisor=advisor.consult,
            should_final_check=advisor.should_final_check,
        )

        t0 = time.monotonic()
        scheduler = RoundScheduler(
            run_context=run_context,
            events=events,
            session_writer=self._session_writer,
            timeout_s=c.timeout_s,
            started_at=t0,
        )

        return {
            "scheduler": scheduler,
            "dispatcher": round_dispatcher,
            "finalizer": run_finalizer,
            "advisor": advisor,
            "started_at": t0,
        }


# ── Module-level helpers referenced by the bootstrapper ──────────────


def _default_on_event(kind: str, data: dict[str, Any]) -> None:
    """Default event handler: write to stderr (CLI mode)."""
    import sys

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


# Forward references to agent_loop helpers — resolved lazily to avoid
# circular imports.  The bootstrapper calls these through the refs.
_run_output_guardrails_fn: Any = None
_coerce_output_type_fn: Any = None
_HookResult: Any = None


def _default_base_prompt() -> str:
    """Base prompt for runs whose config carries no ``system_prompt``.

    Honours ``CLAW_BASE_PROMPT[_FILE]`` and ``.clawagents/base-prompt.md`` so
    a ``ClawAgent`` built directly (not via ``create_claw_agent``) still picks
    up the configured override.
    """
    from clawagents.prompts.base import resolve_base_system_prompt

    return resolve_base_system_prompt()


def _bind_agent_loop_refs() -> None:
    """Called once from agent_loop to set forward references."""
    global _run_output_guardrails_fn, _coerce_output_type_fn, _HookResult
    from .agent_loop import (
        _run_output_guardrails,
        _coerce_output_type,
        HookResult,
    )

    _run_output_guardrails_fn = _run_output_guardrails
    _coerce_output_type_fn = _coerce_output_type
    _HookResult = HookResult
