"""Mid-run steering and next-turn queueing.

Two primitives that let an operator nudge a running agent without
interrupting it:

- :class:`SteerQueue` — messages to inject **into the very next LLM call**
  (e.g. ``/steer please switch to Python``). Drained on the next
  ``on_llm_start`` hook firing, so the model sees the nudge before its next
  decision.
- :class:`NextTurnQueue` — messages to surface **after the current run
  finishes** (e.g. ``/queue summarise findings``). The library never reads
  this on its own — your CLI / gateway is expected to pop from it when
  picking the next user message.

Both queues live on :class:`~clawagents.run_context.RunContext`. They are
populated by the consumer (typically a slash-command dispatcher; see
:mod:`clawagents.commands`) and drained by the agent loop via
:class:`SteerHook` (Python ``RunHooks`` adapter).

Concurrency
-----------
The queues use a simple :class:`threading.Lock` so a separate input thread
(reading from a TTY, websocket, or chat gateway) can safely push while the
agent loop's asyncio task drains.

Example
-------
::

    from clawagents import RunContext, SteerHook, steer, run_agent_graph

    rc = RunContext()

    # … operator types ``/steer please be brief`` in another thread …
    steer(rc, "please be brief")

    # Hooked into the run so on_llm_start drains the queue before each LLM call.
    await run_agent_graph(
        ...,
        run_context=rc,
        run_hooks=SteerHook(),
    )
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, TypeVar

from clawagents.lifecycle import RunHooks
from clawagents.run_context import RunContext
from clawagents.usage import RequestUsage

TContext = TypeVar("TContext")


@dataclass
class SteerMessage:
    """A single pending nudge.

    Attributes:
        text: The message text to inject.
        role: Conversation role to use when injecting. Defaults to
            ``"user"`` since most LLMs treat user messages as authoritative
            and the operator typing ``/steer`` is the user.
    """
    text: str
    role: str = "user"


class _ThreadSafeQueue:
    """Tiny FIFO queue that's safe to push from one thread / drain from another."""

    def __init__(self) -> None:
        self._items: list[SteerMessage] = []
        self._lock = threading.Lock()

    def push(self, msg: str | SteerMessage, *, role: str = "user") -> None:
        """Append a message. Returns immediately."""
        if isinstance(msg, SteerMessage):
            item = msg
        else:
            item = SteerMessage(text=str(msg), role=role)
        with self._lock:
            self._items.append(item)

    def extend(self, items: Iterable[str | SteerMessage], *, role: str = "user") -> None:
        with self._lock:
            for it in items:
                if isinstance(it, SteerMessage):
                    self._items.append(it)
                else:
                    self._items.append(SteerMessage(text=str(it), role=role))

    def drain(self) -> list[SteerMessage]:
        """Return and remove all pending messages atomically."""
        with self._lock:
            out, self._items = self._items, []
        return out

    def peek(self) -> list[SteerMessage]:
        """Return a copy of pending messages without consuming them."""
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __bool__(self) -> bool:
        return len(self) > 0


class SteerQueue(_ThreadSafeQueue):
    """Mid-run nudges injected into the next LLM call.

    Drained by :class:`SteerHook` on the ``on_llm_start`` lifecycle hook.
    """


class NextTurnQueue(_ThreadSafeQueue):
    """Messages saved for **after** the current run.

    The agent loop ignores this queue; consumers should drain it themselves
    when picking the next user message (e.g. between calls to
    ``run_agent_graph``).
    """


# ─── RunContext attachment helpers ────────────────────────────────────────


def _get_steer_queue(rc: RunContext[Any]) -> SteerQueue:
    """Return (and lazily create) the :class:`SteerQueue` on ``rc``.

    Stored under :attr:`RunContext._metadata` so we don't have to expand the
    public dataclass. The first call attaches the queue; subsequent calls
    return the same instance.
    """
    q = rc._metadata.get("__claw_steer_queue__")
    if q is None or not isinstance(q, SteerQueue):
        q = SteerQueue()
        rc._metadata["__claw_steer_queue__"] = q
    return q


def _get_next_turn_queue(rc: RunContext[Any]) -> NextTurnQueue:
    """Return (and lazily create) the :class:`NextTurnQueue` on ``rc``."""
    q = rc._metadata.get("__claw_next_turn_queue__")
    if q is None or not isinstance(q, NextTurnQueue):
        q = NextTurnQueue()
        rc._metadata["__claw_next_turn_queue__"] = q
    return q


def steer(rc: RunContext[Any], message: str, *, role: str = "user") -> None:
    """Push a mid-run nudge onto ``rc``'s :class:`SteerQueue`.

    The message will be injected into the conversation just before the
    next LLM call when :class:`SteerHook` is installed.
    """
    _get_steer_queue(rc).push(message, role=role)


def queue_message(rc: RunContext[Any], message: str, *, role: str = "user") -> None:
    """Push a message onto the next-turn queue.

    Saved for the operator to consume between runs; the agent loop never
    reads it on its own.
    """
    _get_next_turn_queue(rc).push(message, role=role)


def drain_steer(rc: RunContext[Any]) -> list[SteerMessage]:
    """Drain pending steer messages."""
    return _get_steer_queue(rc).drain()


def drain_next_turn(rc: RunContext[Any]) -> list[SteerMessage]:
    """Drain pending next-turn messages."""
    return _get_next_turn_queue(rc).drain()


def peek_steer(rc: RunContext[Any]) -> list[SteerMessage]:
    """Peek pending steer messages without consuming them."""
    return _get_steer_queue(rc).peek()


def peek_next_turn(rc: RunContext[Any]) -> list[SteerMessage]:
    """Peek pending next-turn messages without consuming them."""
    return _get_next_turn_queue(rc).peek()


# ─── RunHooks adapter ─────────────────────────────────────────────────────


class SteerHook(RunHooks[Any]):
    """:class:`RunHooks` that drains ``rc.steer_queue`` on each LLM call.

    Drains pending :class:`SteerMessage`\\ s on ``on_llm_start`` and appends
    each as a fresh dict entry on the live ``messages`` list passed to the
    LLM. The agent loop does not copy this list before invoking
    ``llm.chat``, so in-place mutation here is observed by the next call.

    Args:
        prefix: Optional string prepended to each injected message body.
            Defaults to ``"[steer]"`` so the model can distinguish operator
            nudges from in-conversation user turns.
    """

    def __init__(self, *, prefix: str | None = "[steer]") -> None:
        self.prefix = prefix

    async def on_llm_start(
        self,
        context: RunContext[Any],
        model: str,  # noqa: ARG002 -- unused but required by signature
        messages: list[Any],
    ) -> None:
        pending = drain_steer(context)
        if not pending:
            return
        for nudge in pending:
            text = nudge.text
            if self.prefix:
                text = f"{self.prefix} {text}"
            messages.append({"role": nudge.role, "content": text})


__all__ = [
    "SteerMessage",
    "SteerQueue",
    "NextTurnQueue",
    "SteerHook",
    "steer",
    "queue_message",
    "drain_steer",
    "drain_next_turn",
    "peek_steer",
    "peek_next_turn",
]
