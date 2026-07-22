"""Policy objects shared by serial and parallel tool execution paths."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from clawagents.providers.llm import LLMMessage, NativeToolCall
from clawagents.session.heartbeat import (
    DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S,
    run_with_heartbeat,
)
from clawagents.tools.registry import ParsedToolCall, ToolResult

from .run_runtime import RunEvents
from .tool_observation import (
    _post_tool_side_effects,
    _run_context_workspace,
    _tool_observation,
    _ui_tool_result_text,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolBatchSafetyDecision:
    action: Literal["execute", "stop", "retry"]
    message: str = ""


class ToolBatchSafety:
    """Applies loop/circuit-breaker policy before a tool batch executes."""

    def __init__(self, tracker: Any, events: RunEvents) -> None:
        self._tracker = tracker
        self._events = events

    def check(self, calls: list[ParsedToolCall]) -> ToolBatchSafetyDecision:
        if self._tracker.is_circuit_broken():
            message = "Circuit breaker: too many tool calls with no progress. Stopping."
            self._events.emit(
                "warn",
                {
                    "message": (
                        "circuit breaker tripped "
                        f"({self._tracker._no_progress_count} no-progress calls) — breaking"
                    )
                },
            )
            return ToolBatchSafetyDecision("stop", message)

        poll = next(
            (
                result
                for call in calls
                if (result := self._tracker.check_known_poll_no_progress(call.tool_name, call.args))
                and result.level == "critical"
            ),
            None,
        )
        if poll and poll.stuck:
            self._events.emit("warn", {"message": poll.message})
            return ToolBatchSafetyDecision("stop", poll.message)

        if self._tracker.is_hard_looping_batch(calls):
            names = ", ".join(call.tool_name for call in calls)
            message = f"Tool loop detected ({names}). Stopping."
            self._events.emit("warn", {"message": f"tool loop detected ({names}) — breaking"})
            return ToolBatchSafetyDecision("stop", message)

        if self._tracker.is_ping_ponging():
            names = " ↔ ".join(set(self._tracker._history[-6:]))
            self._events.emit(
                "warn", {"message": f"ping-pong oscillation detected ({names}) — breaking"}
            )
            return ToolBatchSafetyDecision(
                "stop", "Ping-pong loop detected between tools. Stopping."
            )

        if not self._tracker.is_soft_looping_batch(calls):
            return ToolBatchSafetyDecision("execute")

        self._tracker.record_batch(calls)
        warning_number = self._tracker.bump_soft_warning()
        repeated = [
            call
            for call in calls
            if self._tracker.is_soft_looping(call.tool_name, call.args)
        ]
        names = ", ".join(call.tool_name for call in repeated)
        self._events.emit(
            "warn",
            {"message": f"repeated tool call warning #{warning_number}: {names}"},
        )
        if any(call.tool_name == "execute" for call in repeated):
            message = (
                "[System] You are re-calling the same execute command with the same arguments. "
                "The command already ran; if the previous result has success=false or a nonzero "
                "exit_code, treat stdout/stderr as diagnostic feedback, not as a tool failure. "
                "Read the prior output, then edit code or inspect new evidence before trying again. "
                "Do not rerun this command until something relevant changed. "
                "If you believe the task is complete, provide your final answer now."
            )
        else:
            message = (
                f"[System] You are re-calling {names} with the same arguments. "
                "You already have the result in the conversation above. "
                "Use the existing data instead of re-reading. "
                "If you believe the task is complete, provide your final answer now."
            )
        return ToolBatchSafetyDecision("retry", message)


@dataclass(frozen=True)
class ToolCandidate:
    """A parsed call paired with its original response index."""

    original_index: int
    call: ParsedToolCall


class ToolPolicyGate:
    """Applies pre-execution hooks consistently to one or many calls."""

    def __init__(
        self,
        *,
        external_hooks: Any,
        taxonomy_dispatcher: Any,
        before_tool: Any,
        hook_result_type: type[Any],
        events: RunEvents,
    ) -> None:
        self._external_hooks = external_hooks
        self._taxonomy_dispatcher = taxonomy_dispatcher
        self._before_tool = before_tool
        self._hook_result_type = hook_result_type
        self._events = events

    async def filter(
        self,
        candidates: list[ToolCandidate],
        *,
        messages: list[Any],
        on_skipped: Callable[[ToolCandidate, str, str], Awaitable[None]],
    ) -> list[ToolCandidate]:
        """Return the calls approved by configured pre-tool policy.

        ``on_skipped`` owns provider-specific transcript repair.  The gate
        owns policy decisions and their side-effect notifications, so a
        batch cannot bypass a rule that also applies to a single call.
        """
        filtered = await self._apply_external_hooks(candidates, on_skipped)
        filtered = await self._apply_taxonomy(filtered, on_skipped)
        return await self._apply_before_tool(filtered, messages, on_skipped)

    async def _apply_external_hooks(
        self,
        candidates: list[ToolCandidate],
        on_skipped: Callable[[ToolCandidate, str, str], Awaitable[None]],
    ) -> list[ToolCandidate]:
        if not self._external_hooks or self._taxonomy_dispatcher is not None:
            return candidates

        filtered: list[ToolCandidate] = []
        for candidate in candidates:
            call = candidate.call
            try:
                allowed, args = await self._external_hooks.pre_tool_use(
                    call.tool_name, call.args
                )
                if not allowed:
                    await self._deny(
                        candidate,
                        "blocked by external hook",
                        "external_hook",
                        on_skipped,
                    )
                    continue
                call = ParsedToolCall(tool_name=call.tool_name, args=args)
            except Exception as exc:
                self._events.emit(
                    "warn", {"message": f"external pre_tool_use hook error: {exc}"}
                )
            filtered.append(ToolCandidate(candidate.original_index, call))
        return filtered

    async def _apply_taxonomy(
        self,
        candidates: list[ToolCandidate],
        on_skipped: Callable[[ToolCandidate, str, str], Awaitable[None]],
    ) -> list[ToolCandidate]:
        if self._taxonomy_dispatcher is None:
            return candidates

        from clawagents.hooks.external import dispatch_taxonomy_hook
        from clawagents.hooks.taxonomy import HookEvent

        filtered: list[ToolCandidate] = []
        for candidate in candidates:
            call = candidate.call
            try:
                allowed, reason = await dispatch_taxonomy_hook(
                    self._taxonomy_dispatcher,
                    HookEvent.PRE_TOOL_USE,
                    {"tool": call.tool_name, "args": call.args},
                    blocking=True,
                )
                if not allowed:
                    await self._deny(
                        candidate,
                        reason or "blocked by taxonomy hook",
                        "taxonomy",
                        on_skipped,
                    )
                    continue
            except Exception as exc:
                self._events.emit(
                    "warn", {"message": f"taxonomy pre_tool_use hook error: {exc}"}
                )
            filtered.append(candidate)
        return filtered

    async def _apply_before_tool(
        self,
        candidates: list[ToolCandidate],
        messages: list[Any],
        on_skipped: Callable[[ToolCandidate, str, str], Awaitable[None]],
    ) -> list[ToolCandidate]:
        if self._before_tool is None:
            return candidates

        filtered: list[ToolCandidate] = []
        for candidate in candidates:
            call = candidate.call
            reason = "rejected by before_tool hook"
            try:
                hook_result = self._before_tool(call.tool_name, call.args)
                if isinstance(hook_result, self._hook_result_type):
                    if hook_result.reason:
                        reason = hook_result.reason
                    if hook_result.messages:
                        messages.extend(hook_result.messages)
                    if not hook_result.allowed:
                        await self._deny(candidate, reason, "before_tool", on_skipped)
                        continue
                    if hook_result.updated_args is not None:
                        call = ParsedToolCall(
                            tool_name=call.tool_name,
                            args=hook_result.updated_args,
                        )
                elif not bool(hook_result):
                    await self._deny(candidate, reason, "before_tool", on_skipped)
                    continue
            except Exception as exc:
                self._events.emit("warn", {"message": f"before_tool hook error: {exc}"})
                await self._deny(candidate, "hook error", "before_tool", on_skipped)
                continue
            filtered.append(ToolCandidate(candidate.original_index, call))
        return filtered

    async def _deny(
        self,
        candidate: ToolCandidate,
        reason: str,
        source: str,
        on_skipped: Callable[[ToolCandidate, str, str], Awaitable[None]],
    ) -> None:
        self._events.emit(
            "tool_skipped", {"name": candidate.call.tool_name, "reason": reason}
        )
        if self._taxonomy_dispatcher is not None:
            try:
                from clawagents.hooks.external import dispatch_taxonomy_hook
                from clawagents.hooks.taxonomy import HookEvent

                await dispatch_taxonomy_hook(
                    self._taxonomy_dispatcher,
                    HookEvent.PERMISSION_DENIED,
                    {
                        "tool": candidate.call.tool_name,
                        "reason": reason,
                        "source": source,
                    },
                    blocking=False,
                )
            except Exception:
                pass
        await on_skipped(candidate, reason, source)


class ToolTranscriptWriter:
    """Writes provider-valid observations for both native and text tools."""

    @staticmethod
    def append_single(
        messages: list[LLMMessage],
        *,
        response_content: str,
        call: ParsedToolCall,
        native_call: NativeToolCall | None,
        output: str | list[dict[str, Any]],
        thinking: str | None,
        gemini_parts: Any,
        use_native_tools: bool,
    ) -> None:
        if use_native_tools and native_call and native_call.tool_call_id:
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response_content,
                    tool_calls_meta=[
                        {
                            "id": native_call.tool_call_id,
                            "name": call.tool_name,
                            "args": call.args,
                        }
                    ],
                    gemini_parts=gemini_parts,
                    thinking=thinking,
                )
            )
            content = output if isinstance(output, str) else json.dumps(output)
            messages.append(
                LLMMessage(role="tool", content=content, tool_call_id=native_call.tool_call_id)
            )
            return
        messages.append(
            LLMMessage(
                role="assistant",
                content=json.dumps({"tool": call.tool_name, "args": call.args}),
                thinking=thinking,
            )
        )
        messages.append(
            LLMMessage(
                role="user",
                content=f"[Tool Result] {output}" if isinstance(output, str) else output,
            )
        )

    @staticmethod
    def append_batch(
        messages: list[LLMMessage],
        *,
        response_content: str,
        calls: list[ParsedToolCall],
        native_calls: dict[int, NativeToolCall],
        outputs: list[str],
        summaries: list[str],
        thinking: str | None,
        gemini_parts: Any,
        use_native_tools: bool,
    ) -> None:
        if use_native_tools and native_calls:
            metadata = []
            for index, call in enumerate(calls):
                native_call = native_calls.get(index)
                call_id = native_call.tool_call_id if native_call else f"fallback_{index}"
                metadata.append({"id": call_id, "name": call.tool_name, "args": call.args})
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response_content,
                    tool_calls_meta=metadata,
                    gemini_parts=gemini_parts,
                    thinking=thinking,
                )
            )
            for index, output in enumerate(outputs):
                native_call = native_calls.get(index)
                call_id = native_call.tool_call_id if native_call else f"fallback_{index}"
                messages.append(LLMMessage(role="tool", content=output, tool_call_id=call_id))
            return
        messages.append(
            LLMMessage(
                role="assistant",
                content=json.dumps([{"tool": call.tool_name, "args": call.args} for call in calls]),
                thinking=thinking,
            )
        )
        messages.append(
            LLMMessage(role="user", content="[Tool Results]\n" + "\n".join(summaries))
        )


class RethinkController:
    """Turns repeated tool failures into one consistent recovery prompt."""

    def __init__(self, events: RunEvents) -> None:
        self._events = events

    async def maybe_inject(
        self,
        messages: list[LLMMessage],
        *,
        tracker: Any,
        task_type: str,
        round_index: int,
        tool_calls: int,
        learn: bool,
        recorder: Any,
        consult_advisor: Callable[[list[LLMMessage], str], Awaitable[None]],
    ) -> None:
        if tracker is None:
            return
        try:
            from clawagents.trajectory.verifier import compute_adaptive_rethink_threshold

            tracker._threshold = compute_adaptive_rethink_threshold(
                task_type, round_index, tool_calls
            )
        except Exception:
            pass
        if not tracker.should_rethink():
            return
        await consult_advisor(messages, "stuck")
        failures = tracker.consecutive_failures
        rethink_number = tracker.bump_rethink()
        self._events.emit(
            "warn",
            {
                "message": (
                    f"rethink #{rethink_number}: {failures} consecutive failures "
                    f"(threshold={tracker._threshold})"
                )
            },
        )
        from clawagents.graph.loop_tracker import _RETHINK_MESSAGE

        prompt = _RETHINK_MESSAGE.format(n=failures)
        if learn:
            try:
                from clawagents.trajectory.lessons import build_rethink_with_lessons

                failures_by_type = [
                    call
                    for turn in (recorder.turns if recorder else [])
                    for call in turn.tool_calls
                    if not call.success
                ]
                prompt = build_rethink_with_lessons(
                    prompt,
                    sum(call.failure_type == "format" for call in failures_by_type),
                    sum(call.failure_type == "logic" for call in failures_by_type),
                )
            except Exception:
                pass
        messages.append(LLMMessage(role="user", content=prompt))


class ToolCallRunner:
    """Executes one approved tool call with reuse and lifecycle signalling.

    Approval and policy checks deliberately happen before this boundary.  Once
    called, this class guarantees the matching start/end hook sequence even
    when the underlying registry returns a failed ``ToolResult``.
    """

    def __init__(
        self,
        *,
        registry: Any,
        tracker: Any,
        events: RunEvents,
        hooks: Any,
        run_context: Any,
        legacy_on_event: Any,
    ) -> None:
        self._registry = registry
        self._tracker = tracker
        self._events = events
        self._hooks = hooks
        self._run_context = run_context
        self._legacy_on_event = legacy_on_event

    async def execute(
        self,
        call: ParsedToolCall,
        *,
        call_id: str,
    ) -> ToolResult:
        self._events.typed(
            "tool_started",
            {"tool_name": call.tool_name, "call_id": call_id, "args": call.args},
        )
        if self._hooks.hooks:
            await self._hooks.fire("on_tool_start", call.tool_name, call_id, call.args)
        reused = self._tracker.reuse_tool_output(call.tool_name, call.args)
        if reused is not None:
            result = ToolResult(success=True, output=reused)
            self._events.emit(
                "warn",
                {
                    "message": (
                        f"suppressed duplicate/overlapping {call.tool_name} "
                        "(reused prior result)"
                    )
                },
            )
        else:
            result = await run_with_heartbeat(
                self._registry.execute_tool(
                    call.tool_name, call.args, run_context=self._run_context
                ),
                on_event=self._legacy_on_event,
                kind="tool_heartbeat",
                payload={"tool_name": call.tool_name, "call_id": call_id},
                interval=DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S,
            )
            if result.success:
                self._tracker.cache_result_output(
                    call.tool_name, call.args, str(result.output or "")
                )
        if self._hooks.hooks:
            await self._hooks.fire(
                "on_tool_end",
                call.tool_name,
                call_id,
                result.success,
                str(result.output)[:2000] if result.output else "",
                result.error if not result.success else None,
            )
            if not result.success:
                await self._hooks.fire(
                    "on_tool_failure",
                    call.tool_name,
                    call_id,
                    result.error or str(result.output)[:500],
                )
        return result

    async def execute_parallel(
        self,
        calls: list[ParsedToolCall],
        *,
        call_ids: list[str],
    ) -> list[ToolResult]:
        """Execute an approved batch while preserving call-to-result order.

        The registry remains responsible for deciding which calls may actually
        overlap.  This boundary owns only the lifecycle contract around that
        batch: every call gets one start signal and one matching completion
        signal, while the gateway gets a single batch heartbeat.
        """
        if len(calls) != len(call_ids):
            raise ValueError("calls and call_ids must have matching lengths")

        for call, call_id in zip(calls, call_ids):
            self._events.typed(
                "tool_started",
                {"tool_name": call.tool_name, "call_id": call_id, "args": call.args},
            )
            if self._hooks.hooks:
                await self._hooks.fire("on_tool_start", call.tool_name, call_id, call.args)

        results = await run_with_heartbeat(
            self._registry.execute_tools_parallel(calls, run_context=self._run_context),
            on_event=self._legacy_on_event,
            kind="tool_heartbeat",
            payload={
                "parallel": True,
                "tool_names": [call.tool_name for call in calls],
                "call_ids": list(call_ids),
            },
            interval=DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S,
        )

        for call, call_id, result in zip(calls, call_ids, results):
            if self._hooks.hooks:
                await self._hooks.fire(
                    "on_tool_end",
                    call.tool_name,
                    call_id,
                    result.success,
                    str(result.output)[:2000] if result.output else "",
                    result.error if not result.success else None,
                )
                if not result.success:
                    await self._hooks.fire(
                        "on_tool_failure",
                        call.tool_name,
                        call_id,
                        result.error or str(result.output)[:500],
                    )
        return results


@dataclass(frozen=True)
class PreparedToolResult:
    """A result after policy hooks and transcript-safe observation rendering."""

    result: ToolResult
    output: str | list[dict[str, Any]]
    preview: str


class ToolResultProcessor:
    """Applies post-execution policy and prepares one tool observation.

    The processor intentionally keeps result middleware separate from the
    actual registry invocation.  That lets serial and parallel execution use
    identical transformations without accidentally changing the registry's
    scheduling or call/result ordering guarantees.
    """

    def __init__(
        self,
        *,
        external_hooks: Any,
        taxonomy_dispatcher: Any,
        after_tool: Any,
        events: RunEvents,
        session_writer: Any,
        run_context: Any,
        preview_chars: int,
    ) -> None:
        self._external_hooks = external_hooks
        self._taxonomy_dispatcher = taxonomy_dispatcher
        self._after_tool = after_tool
        self._events = events
        self._session_writer = session_writer
        self._run_context = run_context
        self._preview_chars = preview_chars

    async def apply_middleware(
        self,
        calls: list[ParsedToolCall],
        results: list[ToolResult],
    ) -> list[ToolResult]:
        """Apply post-tool hooks in the legacy batch order.

        Keeping this as three passes matters: existing hooks may observe the
        entire batch and rely on all external hooks running before taxonomy
        hooks, and all taxonomy hooks before the local ``after_tool`` hook.
        """
        transformed = list(results)
        if self._external_hooks and self._taxonomy_dispatcher is None:
            external_results: list[ToolResult] = []
            for call, result in zip(calls, transformed):
                try:
                    hook_result = await self._external_hooks.post_tool_use(
                        call.tool_name,
                        call.args,
                        {
                            "success": result.success,
                            "output": str(result.output)[:1000],
                        },
                    )
                    if "success" in hook_result and "output" in hook_result:
                        result = ToolResult(
                            success=hook_result["success"],
                            output=hook_result["output"],
                            error=hook_result.get("error"),
                        )
                except Exception as exc:
                    self._events.emit(
                        "warn", {"message": f"external post_tool_use hook error: {exc}"}
                    )
                external_results.append(result)
            transformed = external_results

        if self._taxonomy_dispatcher is not None:
            from clawagents.hooks.external import dispatch_taxonomy_hook
            from clawagents.hooks.taxonomy import HookEvent

            for call, result in zip(calls, transformed):
                try:
                    await dispatch_taxonomy_hook(
                        self._taxonomy_dispatcher,
                        HookEvent.POST_TOOL_USE,
                        {
                            "tool": call.tool_name,
                            "args": call.args,
                            "success": result.success,
                            "output": str(result.output)[:1000],
                        },
                        blocking=False,
                    )
                    if not result.success:
                        await dispatch_taxonomy_hook(
                            self._taxonomy_dispatcher,
                            HookEvent.POST_TOOL_USE_FAILURE,
                            {
                                "tool": call.tool_name,
                                "args": call.args,
                                "error": result.error or str(result.output)[:500],
                            },
                            blocking=False,
                        )
                except Exception as exc:
                    self._events.emit(
                        "warn", {"message": f"taxonomy post_tool_use hook error: {exc}"}
                    )

        if self._after_tool is None:
            return transformed

        after_results: list[ToolResult] = []
        for call, result in zip(calls, transformed):
            try:
                hooked_result = self._after_tool(call.tool_name, call.args, result)
                if hasattr(hooked_result, "success") and hasattr(hooked_result, "output"):
                    result = hooked_result
                else:
                    self._events.emit(
                        "warn", {"message": "after_tool returned invalid ToolResult — ignored"}
                    )
            except Exception as exc:
                self._events.emit("warn", {"message": f"after_tool hook error: {exc}"})
            after_results.append(result)
        return after_results

    def prepare(
        self,
        call: ParsedToolCall,
        result: ToolResult,
        *,
        call_id: str,
        session_call_id: str | None = None,
    ) -> PreparedToolResult:
        """Render and record a tool result for the model, UI, and session."""
        raw_output = _tool_observation(result)
        ui_text = _ui_tool_result_text(result, raw_output)
        if isinstance(raw_output, list):
            try:
                from clawagents.media.images import sanitize_tool_output

                output = sanitize_tool_output(raw_output)
            except Exception:
                logger.debug("sanitize_tool_output failed", exc_info=True)
                output = raw_output
            preview = "[Multimodal Array Content]"
        else:
            from clawagents.tool_output_artifacts import prepare_tool_output_for_context

            output, artifact_id = prepare_tool_output_for_context(
                tool_name=call.tool_name,
                tool_use_id=call_id,
                output=raw_output,
                workspace=_run_context_workspace(self._run_context),
                success=bool(result.success),
            )
            if artifact_id is not None:
                self._events.emit(
                    "context", {"message": f"tool output crushed/stored id={artifact_id}"}
                )
            preview = output[: self._preview_chars]

        output = _post_tool_side_effects(
            call.tool_name,
            call.args if isinstance(call.args, dict) else {},
            result.success,
            output,
            emit=self._events.emit,
            run_context=self._run_context,
        )
        if isinstance(output, str):
            preview = output[: self._preview_chars]

        self._events.emit(
            "tool_result",
            {
                "name": call.tool_name,
                "success": result.success,
                "preview": preview,
                "output": ui_text,
            },
        )
        self._events.typed(
            "tool_result",
            {
                "tool_name": call.tool_name,
                "call_id": call_id,
                "success": result.success,
                "output": ui_text,
                "error": result.error if not result.success else None,
            },
        )
        if self._session_writer:
            self._session_writer.write_tool_result(
                session_call_id if session_call_id is not None else call_id,
                call.tool_name,
                result.success,
                str(result.output)[:2000],
                error=result.error if not result.success else None,
            )
        return PreparedToolResult(result=result, output=output, preview=preview)
