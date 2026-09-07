"""Start-of-round scheduling concerns for the agent graph."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

from clawagents.providers.llm import LLMMessage

from .context_management import _drain_interject_messages, _sync_goal_reminder_into_system
from .run_runtime import RunEvents

logger = logging.getLogger(__name__)

DEADLINE_MARKER = "[System] Run deadline approaching"
_DEADLINE_KEY = "_run_deadline_monotonic"
_RESERVE_KEY = "_run_deadline_reserve_s"


def remaining_run_seconds(run_context: Any) -> float | None:
    """Return this run's live wall budget, or None for an unlimited run."""
    metadata = getattr(run_context, "_metadata", None)
    deadline = metadata.get(_DEADLINE_KEY) if isinstance(metadata, dict) else None
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        return None
    return max(0.0, deadline - time.monotonic())


def deadline_reserve_seconds(run_context: Any) -> float:
    metadata = getattr(run_context, "_metadata", None)
    reserve = metadata.get(_RESERVE_KEY) if isinstance(metadata, dict) else None
    return float(reserve) if isinstance(reserve, (int, float)) else 60.0


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
        self._deadline_reminded = False
        metadata = getattr(run_context, "_metadata", None)
        if isinstance(metadata, dict):
            # A RunContext may be reused. Never inherit the previous run's timer.
            metadata.pop(_DEADLINE_KEY, None)
            metadata.pop(_RESERVE_KEY, None)
            if timeout_s > 0 and math.isfinite(timeout_s):
                metadata[_DEADLINE_KEY] = self._started_at + timeout_s
                metadata[_RESERVE_KEY] = min(60.0, timeout_s * 0.25)

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
        remaining = remaining_run_seconds(self._run_context)
        if (
            remaining is not None
            and remaining <= deadline_reserve_seconds(self._run_context)
            and not self._deadline_reminded
        ):
            # Append only once; changing the system prefix loses provider cache.
            messages.append(LLMMessage(
                role="user",
                content=(
                    f"{DEADLINE_MARKER}: about {math.ceil(remaining)} seconds remain. "
                    "Use the evidence already gathered to take the next justified action. "
                    "Keep reasoning brief and leave time for verification. "
                    "If the task cannot be finished, state what remains incomplete; "
                    "do not claim unperformed work or checks."
                ),
            ))
            self._deadline_reminded = True
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
        return self._timeout_s > 0 and time.monotonic() - self._started_at >= self._timeout_s
