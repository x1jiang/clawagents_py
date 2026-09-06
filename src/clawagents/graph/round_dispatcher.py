"""Dispatch one prepared LLM turn to its terminal or tool-bearing outcome."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from clawagents.providers.llm import LLMMessage

from .completion_handler import CompletionHandler
from .handoff_router import HandoffRouter
from .tool_batch import ToolBatchSafety
from .tool_turn import ToolTurnExecutor
from .turn_driver import TurnDriver
from .turn_response import TurnResponseInterpreter


@dataclass(frozen=True)
class RoundDispatch:
    action: Literal["continue", "stop", "handoff"]
    messages: list[LLMMessage]
    child_state: Any | None = None


class RoundDispatcher:
    """Implements the response-state branch of the ReAct state machine."""

    def __init__(
        self,
        *,
        driver: TurnDriver,
        response_interpreter: TurnResponseInterpreter,
        completion_handler: CompletionHandler,
        handoff_router: HandoffRouter,
        safety: ToolBatchSafety,
        tool_executor: ToolTurnExecutor,
        run_context: Any,
        use_native_tools: bool,
        consult_advisor: Callable[[list[LLMMessage], str], Awaitable[None]],
        should_final_check: Callable[[Any], bool],
    ) -> None:
        self._driver = driver
        self._response_interpreter = response_interpreter
        self._completion_handler = completion_handler
        self._handoff_router = handoff_router
        self._safety = safety
        self._tool_executor = tool_executor
        self._run_context = run_context
        self._use_native_tools = use_native_tools
        self._consult_advisor = consult_advisor
        self._should_final_check = should_final_check

    async def dispatch(
        self,
        state: Any,
        messages: list[LLMMessage],
        *,
        round_index: int,
        cancel_event: Any,
    ) -> RoundDispatch:
        called = await self._driver.call(
            messages,
            state=state,
            round_index=round_index,
            cancel_event=cancel_event,
        )
        messages = called.messages
        if called.action == "retry":
            return RoundDispatch("continue", messages)
        if called.action == "stop":
            return RoundDispatch("stop", messages)

        parsed = self._response_interpreter.parse(
            called.response,
            use_native_tools=self._use_native_tools,
            run_context=self._run_context,
        )
        if parsed.should_resample:
            return RoundDispatch("continue", messages)
        if not parsed.tool_calls:
            completion = await self._completion_handler.handle(
                state=state,
                messages=messages,
                response=parsed.response,
                thinking=parsed.thinking,
                use_native_tools=self._use_native_tools,
                consult_advisor=self._consult_advisor,
                should_final_check=self._should_final_check(state),
            )
            return RoundDispatch(
                "continue" if completion.action == "continue" else "stop", messages
            )

        handoff = await self._handoff_router.dispatch(
            parsed.tool_calls,
            parsed.native_tool_calls or [],
            response_content=parsed.response.content or "",
            thinking=parsed.thinking,
            messages=messages,
        )
        if handoff.handled:
            if handoff.child_state is None:
                return RoundDispatch("continue", messages)
            return RoundDispatch("handoff", messages, handoff.child_state)

        safety = self._safety.check(parsed.tool_calls)
        if safety.action == "stop":
            state.status = "error"
            state.result = safety.message
            return RoundDispatch("stop", messages)
        if safety.action == "retry":
            messages.append(LLMMessage(role="user", content=safety.message))
            return RoundDispatch("continue", messages)

        await self._tool_executor.execute(
            state=state,
            messages=messages,
            response=parsed.response,
            thinking=parsed.thinking,
            tool_calls=parsed.tool_calls,
            native_tool_calls=parsed.native_tool_calls or [],
            round_index=round_index,
        )
        if state.status == "done":
            return RoundDispatch("stop", messages)
        return RoundDispatch("continue", messages)
