"""Run-scoped infrastructure shared by the agent-loop collaborators.

This module deliberately owns *mechanical* run state only: event dispatch,
lifecycle hooks, and the identity-based session journal.  Policy belongs in
the turn runner and tool executor; keeping it out of these helpers makes the
side effects explicit and independently testable.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from clawagents.providers.llm import LLMMessage
from clawagents.run_context import RunContext
from clawagents.stream_events import StreamEvent, stream_event_from_kind


EventEmitter = Callable[[str, dict[str, Any]], None]
TypedEventEmitter = Callable[[StreamEvent], None]


class RunEvents:
    """Single owner for legacy and typed event emission.

    The legacy ``on_event`` callback remains the compatibility surface.  The
    typed stream is an additive projection, so a typed callback failure must
    never interrupt the agent run.
    """

    def __init__(
        self,
        emit: EventEmitter,
        on_stream_event: TypedEventEmitter | None = None,
    ) -> None:
        self._emit = emit
        self._on_stream_event = on_stream_event

    def emit(self, kind: str, data: dict[str, Any] | None = None) -> None:
        self._emit(kind, data or {})

    def typed(self, kind: str, data: dict[str, Any] | None = None) -> None:
        if self._on_stream_event is None:
            return
        try:
            self._on_stream_event(stream_event_from_kind(kind, data or {}))
        except Exception as error:  # event consumers are observational
            self.emit("warn", {"message": f"on_stream_event error: {error}"})


class HookDispatcher:
    """Runs lifecycle hooks as best-effort observers.

    A hook is intentionally unable to break the agent loop by raising.  The
    dispatcher centralises that guarantee instead of duplicating it at each
    hook call site.
    """

    def __init__(
        self,
        hooks: list[Any],
        run_context: RunContext,
        events: RunEvents,
    ) -> None:
        self.hooks = hooks
        self._context = run_context
        self._events = events

    async def fire(self, method_name: str, *args: Any) -> None:
        for hook in self.hooks:
            callback = getattr(hook, method_name, None)
            if callback is None:
                continue
            try:
                result = callback(self._context, *args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as error:
                self._events.emit(
                    "warn", {"message": f"{method_name} hook error: {error}"}
                )


async def session_get_items(
    session: Any,
    limit: int | None = None,
) -> list[LLMMessage]:
    """Read session history across the supported session protocol variants."""
    if session is None:
        return []
    getter = getattr(session, "get_items", None)
    if getter is None:
        return []
    accepts_limit = False
    if limit is not None:
        try:
            signature = inspect.signature(getter)
            accepts_limit = "limit" in signature.parameters or any(
                parameter.kind
                in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_limit = True
    result = getter(limit=limit) if accepts_limit else getter()
    if hasattr(result, "__await__"):
        result = await result
    if not isinstance(result, list):
        return []
    messages: list[LLMMessage] = []
    for item in result:
        if isinstance(item, LLMMessage):
            messages.append(item)
        elif isinstance(item, dict) and "role" in item:
            messages.append(
                LLMMessage(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                    tool_call_id=item.get("tool_call_id"),
                    tool_calls_meta=item.get("tool_calls_meta"),
                    thinking=item.get("thinking"),
                )
            )
    return messages


async def session_add_items(session: Any, items: list[LLMMessage]) -> None:
    """Append only durable, run-authored messages to a session backend."""
    if session is None or not items:
        return
    adder = getattr(session, "add_items", None)
    if adder is None:
        return
    result = adder(items)
    if hasattr(result, "__await__"):
        await result


class SessionMessageJournal:
    """Tracks which message objects are durable user-visible transcript turns.

    Compaction and transcript repair construct temporary message objects.  The
    journal records identity rather than a list index so those transformations
    cannot make persistence duplicate or lose a real turn.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self.initial_ids: frozenset[int] = frozenset()
        self._seen: dict[int, LLMMessage] = {}
        self._pending: list[LLMMessage] = []

    @property
    def enabled(self) -> bool:
        return self._session is not None

    async def preload(
        self,
        messages: list[LLMMessage],
        *,
        limit: int | None,
        repair: Callable[[list[LLMMessage]], list[LLMMessage]],
        drop_leading_orphans: Callable[[list[LLMMessage]], list[LLMMessage]],
    ) -> list[LLMMessage]:
        """Insert repaired history before the current user task."""
        task_message = next(
            (message for message in messages if message.role == "user"), None
        )
        if not self.enabled:
            self.begin(messages)
            return messages
        prior = await session_get_items(self._session, limit=limit)
        if prior:
            prior = repair(drop_leading_orphans(prior))
            insert_at = next(
                (index for index, message in enumerate(messages) if message.role == "user"),
                len(messages),
            )
            messages = [*messages[:insert_at], *prior, *messages[insert_at:]]
        self.begin(messages)
        if task_message is not None:
            self._pending.append(task_message)
        return messages

    def begin(self, messages: list[LLMMessage]) -> None:
        self.initial_ids = frozenset(id(message) for message in messages)
        self._seen = {id(message): message for message in messages}

    def note(self, messages: list[LLMMessage], *, durable: bool) -> None:
        for message in messages:
            identity = id(message)
            if identity in self._seen:
                continue
            self._seen[identity] = message
            if durable:
                self._pending.append(message)

    async def persist(self, messages: list[LLMMessage]) -> None:
        self.note(messages, durable=True)
        await session_add_items(self._session, self._pending)
