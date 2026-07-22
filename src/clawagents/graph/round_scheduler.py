"""Start-of-round scheduling concerns for the agent graph."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from clawagents.providers.llm import LLMMessage

from .context_management import _drain_interject_messages, _sync_goal_reminder_into_system
from .run_runtime import RunEvents

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundStart:
    action: Literal["proceed", "stop"]
    messages: list[LLMMessage]


class RoundScheduler:
    """Enforces cancellation, budget, timeout and per-round maintenance."""

    def __init__(
        self,
        *,
        run_context: Any,
        events: RunEvents,
        session_writer: Any,
        timeout_s: float,
        started_at: float | None = None,
    ) -> None:
        self._run_context = run_context
        self._events = events
        self._session_writer = session_writer
        self._timeout_s = timeout_s
        self._started_at = started_at if started_at is not None else time.monotonic()

    async def begin(
        self,
        state: Any,
        messages: list[LLMMessage],
        *,
        round_index: int,
        cancel_event: Any,
    ) -> RoundStart:
        if cancel_event.is_set():
            state.status = "done"
            state.result = state.result or "[cancelled]"
            return RoundStart("stop", messages)
        if not self._run_context.iteration_budget.consume():
            self._events.emit(
                "warn",
                {
                    "message": (
                        "iteration budget exhausted "
                        f"({self._run_context.iteration_budget.used}/"
                        f"{self._run_context.iteration_budget.max_total})"
                    )
                },
            )
            state.status = "max_iterations"
            state.result = state.result or "[iteration budget exhausted]"
            return RoundStart("stop", messages)
        state.iterations += 1
        await self._apply_interjections(messages)
        self._sync_goal_reminder(messages)
        if self._session_writer is not None:
            self._session_writer.write_turn_started(round_index)
        if self._timed_out():
            state.status = "error"
            state.result = f"Agent run exceeded {self._timeout_s}s global timeout"
            self._events.emit("warn", {"message": state.result})
            return RoundStart("stop", messages)
        return RoundStart("proceed", messages)

    async def _apply_interjections(self, messages: list[LLMMessage]) -> None:
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("mid_turn_interject"):
                return
            interjections = _drain_interject_messages(self._run_context)
            if interjections:
                messages.extend(interjections)
                self._events.emit(
                    "context",
                    {
                        "message": (
                            f"mid-turn interjection applied ({len(interjections)} turn(s))"
                        )
                    },
                )
        except Exception:
            logger.debug("mid-turn interject drain failed", exc_info=True)

    def _sync_goal_reminder(self, messages: list[LLMMessage]) -> None:
        try:
            _sync_goal_reminder_into_system(messages, self._run_context)
        except Exception:
            logger.debug("goal reminder sync failed", exc_info=True)

    def _timed_out(self) -> bool:
        return self._timeout_s > 0 and time.monotonic() - self._started_at > self._timeout_s
