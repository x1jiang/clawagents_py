"""Handoff routing for an agent turn.

The graph loop decides *when* a handoff is eligible.  This module owns the
exclusive transfer itself: transcript completion, filtering, lifecycle
notifications, target invocation, and the child result returned to the
parent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from clawagents.handoffs import Handoff, HandoffInputData
from clawagents.providers.llm import LLMMessage, NativeToolCall
from clawagents.tools.registry import ParsedToolCall
from clawagents.tracing import handoff_span

from .run_runtime import HookDispatcher, RunEvents


@dataclass(frozen=True)
class HandoffDispatchResult:
    """Outcome of checking a turn for a handoff request."""

    handled: bool
    child_state: Any | None = None


class _TransientSession:
    """Read-only preload used to give a target agent filtered history."""

    def __init__(self, items: list[LLMMessage]) -> None:
        self._items = items

    async def get_items(self) -> list[LLMMessage]:
        return list(self._items)

    async def add_items(self, _new: list[LLMMessage]) -> None:
        return None


class HandoffRouter:
    """Per-run router for synthetic handoff tool calls."""

    def __init__(
        self,
        *,
        handoffs: dict[str, Handoff],
        events: RunEvents,
        hooks: HookDispatcher,
        run_context: Any,
        from_agent: str,
        task: str,
        use_native_tools: bool,
        session_initial_ids: set[int],
        on_stream_event: Any,
    ) -> None:
        self._handoffs = handoffs
        self._events = events
        self._hooks = hooks
        self._run_context = run_context
        self._from_agent = from_agent
        self._task = task
        self._use_native_tools = use_native_tools
        self._session_initial_ids = session_initial_ids
        self._on_stream_event = on_stream_event

    async def dispatch(
        self,
        tool_calls: list[ParsedToolCall],
        native_tool_calls: list[NativeToolCall],
        *,
        response_content: str,
        thinking: str | None,
        messages: list[LLMMessage],
    ) -> HandoffDispatchResult:
        """Run the first requested handoff, if this turn contains one.

        Failed resolution and failed child invocation are considered handled:
        they append an observation for the parent LLM, which can then choose a
        recovery action in its next turn.
        """
        selected = self._select(tool_calls, native_tool_calls)
        if selected is None:
            return HandoffDispatchResult(handled=False)
        call, native_call = selected
        handoff = self._handoffs[call.tool_name]
        reason = str(call.args.get("reason", "")) if isinstance(call.args, dict) else ""

        try:
            target_agent = handoff.resolve_target()
        except Exception as exc:
            self._events.emit("warn", {"message": f"handoff target resolution failed: {exc}"})
            messages.append(
                LLMMessage(
                    role="user",
                    content=f"[Handoff Error] Could not resolve target agent: {exc}",
                )
            )
            return HandoffDispatchResult(handled=True)

        target_name = getattr(target_agent, "name", None) or call.tool_name
        self._append_trigger_messages(
            messages,
            call=call,
            native_call=native_call,
            response_content=response_content,
            thinking=thinking,
            target_name=target_name,
        )
        filtered_messages = await self._filter_input(handoff, messages)
        await self._notify_handoff(handoff, call.tool_name, target_name, reason)

        forward_task = self._forward_task(filtered_messages)
        preload = self._preload_messages(filtered_messages, forward_task)
        try:
            with handoff_span(
                call.tool_name,
                from_agent=self._from_agent,
                to_agent=target_name,
            ):
                child_state = await target_agent.invoke(
                    forward_task,
                    run_context=self._run_context,
                    session=_TransientSession(preload) if preload else None,
                    on_stream_event=self._on_stream_event,
                    session_end_tail=False,
                )
        except Exception as exc:
            self._events.emit("warn", {"message": f"handoff target raised: {exc}"})
            messages.append(
                LLMMessage(role="user", content=f"[Handoff Error] Target agent failed: {exc}")
            )
            return HandoffDispatchResult(handled=True)
        return HandoffDispatchResult(handled=True, child_state=child_state)

    def _select(
        self,
        tool_calls: list[ParsedToolCall],
        native_tool_calls: list[NativeToolCall],
    ) -> tuple[ParsedToolCall, NativeToolCall | None] | None:
        for index, call in enumerate(tool_calls):
            if call.tool_name in self._handoffs:
                native_call = native_tool_calls[index] if index < len(native_tool_calls) else None
                return call, native_call
        return None

    def _append_trigger_messages(
        self,
        messages: list[LLMMessage],
        *,
        call: ParsedToolCall,
        native_call: NativeToolCall | None,
        response_content: str,
        thinking: str | None,
        target_name: str,
    ) -> None:
        if self._use_native_tools and native_call and native_call.tool_call_id:
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
                    thinking=thinking,
                )
            )
            messages.append(
                LLMMessage(
                    role="tool",
                    content=f"[Handoff] transferred to {target_name}",
                    tool_call_id=native_call.tool_call_id,
                )
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
            LLMMessage(role="user", content=f"[Handoff] transferred to {target_name}")
        )

    async def _filter_input(
        self,
        handoff: Handoff,
        messages: list[LLMMessage],
    ) -> list[LLMMessage]:
        payload = HandoffInputData(
            input_history=list(messages),
            pre_handoff_items=[
                message for message in messages if id(message) in self._session_initial_ids
            ],
            new_items=[
                message for message in messages if id(message) not in self._session_initial_ids
            ],
            run_context=self._run_context,
        )
        if handoff.input_filter is not None:
            try:
                payload = handoff.input_filter(payload)
            except Exception as exc:
                self._events.emit("warn", {"message": f"handoff input_filter raised: {exc}"})
        return list(payload.input_history)

    async def _notify_handoff(
        self,
        handoff: Handoff,
        tool_name: str,
        target_name: str,
        reason: str,
    ) -> None:
        if handoff.on_handoff is not None:
            try:
                await handoff.on_handoff(self._run_context)
            except Exception as exc:
                self._events.emit("warn", {"message": f"handoff on_handoff raised: {exc}"})
        if self._hooks.hooks:
            await self._hooks.fire("on_handoff", self._from_agent, target_name)
        self._events.emit(
            "warn", {"message": f"handoff: {self._from_agent} → {target_name}"}
        )
        self._events.typed(
            "handoff_occurred",
            {
                "from_agent": self._from_agent,
                "to_agent": target_name,
                "tool_name": tool_name,
                "reason": reason,
            },
        )

    def _forward_task(self, messages: list[LLMMessage]) -> str:
        last_user = next(
            (
                message
                for message in reversed(messages)
                if message.role == "user" and isinstance(message.content, str)
            ),
            None,
        )
        return last_user.content if last_user is not None else self._task

    @staticmethod
    def _preload_messages(
        messages: list[LLMMessage],
        forward_task: str,
    ) -> list[LLMMessage]:
        preload = [message for message in messages if message.role != "system"]
        if (
            preload
            and preload[-1].role == "user"
            and isinstance(preload[-1].content, str)
            and preload[-1].content == forward_task
        ):
            return preload[:-1]
        return preload
