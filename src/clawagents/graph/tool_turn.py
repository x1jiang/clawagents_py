"""Execution of a parsed tool-call turn.

This module is the orchestration boundary after the LLM has requested tools:
it records the assistant tool message, applies pre-tool policy, obtains
approval, executes serial or parallel calls, and appends their observations.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from clawagents.providers.llm import LLMMessage, NativeToolCall
from clawagents.tools.registry import ParsedToolCall

from .tool_batch import (
    RethinkController,
    ToolCallRunner,
    ToolCandidate,
    ToolPolicyGate,
    ToolResultProcessor,
    ToolTranscriptWriter,
)
from .tool_observation import _wait_for_tool_approval


class ToolTurnExecutor:
    """Coordinates one tool-bearing assistant response."""

    def __init__(
        self,
        *,
        registry: Any,
        run_context: Any,
        events: Any,
        policy_gate: ToolPolicyGate,
        call_runner: ToolCallRunner,
        result_processor: ToolResultProcessor,
        rethink_controller: RethinkController,
        loop_tracker: Any,
        failure_tracker: Any,
        recorder: Any,
        session_writer: Any,
        require_approval_set: set[str],
        approval_handler: Any,
        use_native_tools: bool,
        preview_chars: int,
        task_type: str,
        learn: bool,
        consult_advisor: Callable[[list[LLMMessage], str], Awaitable[None]],
        llm: Any,
    ) -> None:
        self._registry = registry
        self._run_context = run_context
        self._events = events
        self._policy_gate = policy_gate
        self._call_runner = call_runner
        self._result_processor = result_processor
        self._rethink_controller = rethink_controller
        self._loop_tracker = loop_tracker
        self._failure_tracker = failure_tracker
        self._recorder = recorder
        self._session_writer = session_writer
        self._require_approval_set = require_approval_set
        self._approval_handler = approval_handler
        self._use_native_tools = use_native_tools
        self._preview_chars = preview_chars
        self._task_type = task_type
        self._learn = learn
        self._consult_advisor = consult_advisor
        self._llm = llm
        self._last_memory_extraction_turn = 0

    async def execute(
        self,
        *,
        state: Any,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
        tool_calls: list[ParsedToolCall],
        native_tool_calls: list[NativeToolCall],
        round_index: int,
        record_assistant_message: bool = True,
    ) -> None:
        """Execute the requested calls and append the next model observation."""
        if record_assistant_message:
            if self._use_native_tools and response.content and response.content.strip():
                self._events.typed(
                    "assistant_message",
                    {"content": response.content, "thinking": thinking},
                )
            self._write_assistant_tool_message(response, native_tool_calls, thinking)

        if len(tool_calls) == 1:
            await self._execute_single(
                state=state,
                messages=messages,
                response=response,
                thinking=thinking,
                call=tool_calls[0],
                native_call=native_tool_calls[0] if native_tool_calls else None,
                round_index=round_index,
            )
            return
        await self._execute_batch(
            state=state,
            messages=messages,
            response=response,
            thinking=thinking,
            tool_calls=tool_calls,
            native_tool_calls=native_tool_calls,
            round_index=round_index,
        )

    def _write_assistant_tool_message(
        self,
        response: Any,
        native_tool_calls: list[NativeToolCall],
        thinking: str | None,
    ) -> None:
        if self._session_writer is None:
            return
        metadata = [
            {"id": call.tool_call_id, "name": call.tool_name, "args": call.args}
            for call in native_tool_calls
        ]
        self._session_writer.write_assistant_message(
            response.content or "",
            tool_calls=metadata or None,
            thinking=thinking,
        )

    async def _execute_single(
        self,
        *,
        state: Any,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
        call: ParsedToolCall,
        native_call: NativeToolCall | None,
        round_index: int,
    ) -> None:
        self._events.emit("tool_call", {"name": call.tool_name})

        async def on_skipped(
            candidate: ToolCandidate,
            reason: str,
            source: str,
        ) -> None:
            skipped = candidate.call
            content = (
                f"[Tool Skipped] {skipped.tool_name} was blocked by external hook."
                if source == "external_hook"
                else f"[Tool Skipped] {skipped.tool_name} was not approved: {reason}"
            )
            self._append_single_skip(
                messages,
                response=response,
                thinking=thinking,
                call=skipped,
                native_call=native_call,
                content=content,
            )

        approved = await self._policy_gate.filter(
            [ToolCandidate(0, call)],
            messages=messages,
            on_skipped=on_skipped,
        )
        if not approved:
            return
        call = approved[0].call
        self._loop_tracker.record(call.tool_name, call.args)

        call_id = native_call.tool_call_id if native_call else call.tool_name
        if not await self._is_single_call_approved(
            messages,
            response=response,
            thinking=thinking,
            call=call,
            native_call=native_call,
            call_id=call_id,
        ):
            return

        tool_result = await self._call_runner.execute(call, call_id=call_id)
        state.tool_calls += 1
        [tool_result] = await self._result_processor.apply_middleware([call], [tool_result])
        prepared = self._result_processor.prepare(
            call,
            tool_result,
            call_id=call_id,
            session_call_id=native_call.tool_call_id if native_call else "",
        )
        if isinstance(prepared.output, str):
            self._loop_tracker.record_result(call.tool_name, call.args, prepared.output)
        if self._failure_tracker:
            self._failure_tracker.record(tool_result.success, call.tool_name)
        self._record_single_trajectory(
            messages,
            response=response,
            thinking=thinking,
            call=call,
            result=tool_result,
            preview=prepared.preview,
        )
        ToolTranscriptWriter.append_single(
            messages,
            response_content=response.content or "",
            call=call,
            native_call=native_call,
            output=prepared.output,
            thinking=thinking,
            gemini_parts=getattr(response, "gemini_parts", None),
            use_native_tools=self._use_native_tools,
        )
        await self._maybe_rethink(messages, state, round_index)

    async def _is_single_call_approved(
        self,
        messages: list[LLMMessage],
        *,
        response: Any,
        thinking: str | None,
        call: ParsedToolCall,
        native_call: NativeToolCall | None,
        call_id: str,
    ) -> bool:
        approval_state = self._run_context.is_tool_approved(call_id, tool_name=call.tool_name)
        if approval_state is False:
            record = self._run_context.get_approval(call_id, tool_name=call.tool_name)
            reason = (record.reason if record else None) or "rejected via RunContext"
            self._events.emit("tool_skipped", {"name": call.tool_name, "reason": reason})
            self._append_single_skip(
                messages,
                response=response,
                thinking=thinking,
                call=call,
                native_call=native_call,
                content=f"[Tool Skipped] {call.tool_name} was rejected: {reason}",
            )
            return False
        if approval_state is not None:
            return True

        tool = self._registry.tools.get(call.tool_name)
        needs_approval = (
            call.tool_name in self._require_approval_set
            or bool(getattr(tool, "require_approval", False))
        )
        self._events.emit("approval_required", {"name": call.tool_name, "id": call_id})
        self._events.typed(
            "approval_required",
            {"tool_name": call.tool_name, "call_id": call_id, "args": call.args},
        )
        if not needs_approval or self._approval_handler is None:
            return True
        approved = await _wait_for_tool_approval(
            self._run_context,
            call_id,
            call.tool_name,
            call.args if isinstance(call.args, dict) else {},
            approval_handler=self._approval_handler,
            emit=self._events.emit,
        )
        if not approved:
            self._events.emit(
                "tool_skipped",
                {"name": call.tool_name, "reason": "approval denied or timed out"},
            )
            self._append_single_skip(
                messages,
                response=response,
                thinking=thinking,
                call=call,
                native_call=native_call,
                content=f"[Tool Skipped] {call.tool_name} was not approved",
            )
            return False
        self._run_context.approve_tool(call_id, tool_name=call.tool_name)
        return True

    def _append_single_skip(
        self,
        messages: list[LLMMessage],
        *,
        response: Any,
        thinking: str | None,
        call: ParsedToolCall,
        native_call: NativeToolCall | None,
        content: str,
    ) -> None:
        if self._use_native_tools and native_call and native_call.tool_call_id:
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls_meta=[
                        {
                            "id": native_call.tool_call_id,
                            "name": call.tool_name,
                            "args": call.args,
                        }
                    ],
                    gemini_parts=getattr(response, "gemini_parts", None),
                    thinking=thinking,
                )
            )
            messages.append(
                LLMMessage(role="tool", content=content, tool_call_id=native_call.tool_call_id)
            )
            return
        messages.append(LLMMessage(role="user", content=content))

    async def _execute_batch(
        self,
        *,
        state: Any,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
        tool_calls: list[ParsedToolCall],
        native_tool_calls: list[NativeToolCall],
        round_index: int,
    ) -> None:
        skipped_sources: set[str] = set()

        async def on_skipped(
            candidate: ToolCandidate,
            reason: str,
            source: str,
        ) -> None:
            skipped_sources.add(source)
            if source == "before_tool":
                return
            content = (
                f"[Tool Skipped] {candidate.call.tool_name} was blocked by external hook."
                if source == "external_hook"
                else f"[Tool Skipped] {candidate.call.tool_name} was not approved: {reason}"
            )
            messages.append(LLMMessage(role="user", content=content))

        candidates = await self._policy_gate.filter(
            [ToolCandidate(index, call) for index, call in enumerate(tool_calls)],
            messages=messages,
            on_skipped=on_skipped,
        )
        if not candidates:
            if "before_tool" in skipped_sources:
                messages.append(
                    LLMMessage(
                        role="user",
                        content="[Tool Skipped] All tool calls were not approved.",
                    )
                )
            return

        calls = [candidate.call for candidate in candidates]
        original_indices = [candidate.original_index for candidate in candidates]
        call_ids = self._resolve_call_ids(calls, original_indices, native_tool_calls)
        calls, call_ids, original_indices = self._filter_batch_approvals(
            calls,
            call_ids,
            original_indices,
            messages,
        )
        if not calls:
            return

        for call in calls:
            self._events.emit("tool_call", {"name": call.tool_name})
        self._loop_tracker.record_batch(calls)
        results = await self._call_runner.execute_parallel(calls, call_ids=call_ids)
        state.tool_calls += len(calls)
        results = await self._result_processor.apply_middleware(calls, results)

        native_map = {
            index: native_tool_calls[original_index]
            for index, original_index in enumerate(original_indices)
            if original_index < len(native_tool_calls)
        }
        summaries: list[str] = []
        outputs: list[str] = []
        for call, result, call_id in zip(calls, results, call_ids):
            prepared = self._result_processor.prepare(call, result, call_id=call_id)
            if isinstance(prepared.output, str):
                summaries.append(f"{call.tool_name}({json.dumps(call.args)}) => {prepared.output}")
                outputs.append(prepared.output)
            else:
                summaries.append(
                    f"{call.tool_name}({json.dumps(call.args)}) => "
                    f"[Multimodal Output Length: {len(prepared.output)}]"
                )
                summaries.append(json.dumps(prepared.output))
                outputs.append(json.dumps(prepared.output))

        for call, output in zip(calls, outputs):
            self._loop_tracker.record_result(call.tool_name, call.args, output)
        if self._failure_tracker:
            self._failure_tracker.record_batch(
                [(result.success, call.tool_name) for call, result in zip(calls, results)]
            )
        self._record_batch_trajectory(messages, response, thinking, calls, results)
        ToolTranscriptWriter.append_batch(
            messages,
            response_content=response.content or "",
            calls=calls,
            native_calls=native_map,
            outputs=outputs,
            summaries=summaries,
            thinking=thinking,
            gemini_parts=getattr(response, "gemini_parts", None),
            use_native_tools=self._use_native_tools,
        )
        await self._maybe_rethink(messages, state, round_index)
        await self._maybe_extract_memories(messages, round_index)

    def _resolve_call_ids(
        self,
        calls: list[ParsedToolCall],
        original_indices: list[int],
        native_tool_calls: list[NativeToolCall],
    ) -> list[str]:
        return [
            (
                native_tool_calls[original_index].tool_call_id
                if original_index < len(native_tool_calls)
                else None
            )
            or call.tool_name
            for call, original_index in zip(calls, original_indices)
        ]

    def _filter_batch_approvals(
        self,
        calls: list[ParsedToolCall],
        call_ids: list[str],
        original_indices: list[int],
        messages: list[LLMMessage],
    ) -> tuple[list[ParsedToolCall], list[str], list[int]]:
        runnable_calls: list[ParsedToolCall] = []
        runnable_ids: list[str] = []
        runnable_indices: list[int] = []
        for call, call_id, original_index in zip(calls, call_ids, original_indices):
            approval_state = self._run_context.is_tool_approved(call_id, tool_name=call.tool_name)
            if approval_state is False:
                record = self._run_context.get_approval(call_id, tool_name=call.tool_name)
                reason = (record.reason if record else None) or "rejected via RunContext"
                self._events.emit("tool_skipped", {"name": call.tool_name, "reason": reason})
                messages.append(
                    LLMMessage(
                        role="user",
                        content=f"[Tool Skipped] {call.tool_name} was rejected: {reason}",
                    )
                )
                continue
            if approval_state is None:
                self._events.emit("approval_required", {"name": call.tool_name, "id": call_id})
                self._events.typed(
                    "approval_required",
                    {"tool_name": call.tool_name, "call_id": call_id, "args": call.args},
                )
            runnable_calls.append(call)
            runnable_ids.append(call_id)
            runnable_indices.append(original_index)
        return runnable_calls, runnable_ids, runnable_indices

    def _record_single_trajectory(
        self,
        messages: list[LLMMessage],
        *,
        response: Any,
        thinking: str | None,
        call: ParsedToolCall,
        result: Any,
        preview: str,
    ) -> None:
        if self._recorder is None:
            return
        from clawagents.trajectory.recorder import ToolCallRecord

        self._recorder.record_turn(
            response_text=response.content or "",
            model=response.model,
            tokens_used=response.tokens_used,
            tool_calls=[
                ToolCallRecord(
                    tool_name=call.tool_name,
                    args=call.args,
                    success=result.success,
                    output_preview=preview,
                    error=result.error if not result.success else None,
                )
            ],
            observation_context=self._observation_context(messages, "[Tool Result]"),
            thinking=thinking,
        )

    def _record_batch_trajectory(
        self,
        messages: list[LLMMessage],
        response: Any,
        thinking: str | None,
        calls: list[ParsedToolCall],
        results: list[Any],
    ) -> None:
        if self._recorder is None:
            return
        from clawagents.trajectory.recorder import ToolCallRecord

        records = []
        for call, result in zip(calls, results):
            if not result.success:
                preview = (result.error or "")[: self._preview_chars]
            elif isinstance(result.output, str):
                preview = result.output[: self._preview_chars]
            else:
                preview = "[multimodal]"
            records.append(
                ToolCallRecord(
                    tool_name=call.tool_name,
                    args=call.args,
                    success=result.success,
                    output_preview=preview,
                    error=result.error if not result.success else None,
                )
            )
        self._recorder.record_turn(
            response_text=response.content or "",
            model=response.model,
            tokens_used=response.tokens_used,
            tool_calls=records,
            observation_context=self._observation_context(messages, "[Tool Result"),
            thinking=thinking,
        )

    @staticmethod
    def _observation_context(messages: list[LLMMessage], prefix: str) -> str:
        for message in reversed(messages):
            if (
                message.role in ("user", "tool")
                and isinstance(message.content, str)
                and message.content.startswith(prefix)
            ):
                return message.content[:300]
        return ""

    async def _maybe_rethink(
        self,
        messages: list[LLMMessage],
        state: Any,
        round_index: int,
    ) -> None:
        await self._rethink_controller.maybe_inject(
            messages,
            tracker=self._failure_tracker,
            task_type=self._task_type,
            round_index=round_index,
            tool_calls=state.tool_calls,
            learn=self._learn,
            recorder=self._recorder,
            consult_advisor=self._consult_advisor,
        )

    async def _maybe_extract_memories(
        self,
        messages: list[LLMMessage],
        round_index: int,
    ) -> None:
        if not (
            self._learn
            and self._recorder
            and not getattr(self._run_context, "skip_memory", False)
        ):
            return
        try:
            from clawagents.trajectory.background_memory import maybe_extract_memories

            self._last_memory_extraction_turn = await maybe_extract_memories(
                self._llm,
                messages,
                round_index,
                self._last_memory_extraction_turn,
            )
        except Exception:
            pass
