"""Session heartbeat and auto-cleanup, plus per-tool activity heartbeat.

Sessions without heartbeat auto-release resources after timeout. The
``run_with_heartbeat`` helper additionally emits periodic activity events
during long-running coroutines (such as a slow tool execution) so upstream
gateways do not flag the connection as idle. Mirrors Hermes' "activity
heartbeats prevent false gateway inactivity timeouts" pattern.
"""
import asyncio
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


class SessionHeartbeat:
    def __init__(
        self,
        timeout_s: float = 300.0,
        cleanup_fn: Callable[[str], None] | None = None,
    ):
        self._sessions: dict[str, float] = {}  # session_id -> last_heartbeat
        self._timeout_s = timeout_s
        self._cleanup_fn = cleanup_fn
        self._task: asyncio.Task | None = None

    def heartbeat(self, session_id: str) -> None:
        self._sessions[session_id] = time.monotonic()

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._monitor())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(self._timeout_s / 2)
            now = time.monotonic()
            stale = [
                sid for sid, ts in self._sessions.items()
                if now - ts > self._timeout_s
            ]
            for sid in stale:
                del self._sessions[sid]
                if self._cleanup_fn:
                    self._cleanup_fn(sid)


# Default cadence for per-tool activity heartbeats. ~20s comfortably stays
# under the conservative 30s "idle" thresholds typical of HTTP/WS proxies
# (nginx default proxy_read_timeout, common chat-platform gateways) while
# still keeping noise low for fast tools.
DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S = 20.0


async def run_with_heartbeat(
    coro: Awaitable[T],
    on_event: Optional[Callable[..., Any]] = None,
    *,
    kind: str = "heartbeat",
    payload: Optional[dict] = None,
    interval: float = DEFAULT_ACTIVITY_HEARTBEAT_INTERVAL_S,
    first_after: Optional[float] = None,
) -> T:
    """Await ``coro`` while emitting periodic activity events.

    The coroutine is run as an asyncio Task; in parallel, a heartbeat task
    sleeps for ``interval`` seconds and then calls ``on_event(kind, payload)``
    repeatedly until the work coroutine finishes or raises. The heartbeat
    task is cancelled cleanly in either case, including on user cancellation.

    Behaviour:
      * If ``on_event`` is None or ``interval <= 0``, this degenerates to
        a plain ``await coro`` with no extra task scheduling — zero overhead
        for callers that haven't wired up a gateway listener.
      * The first heartbeat fires after ``first_after`` seconds (default:
        ``interval``); subsequent heartbeats fire every ``interval`` seconds.
        Use ``first_after=0`` to emit immediately on entry (e.g., for
        explicit "started" beats).
      * ``on_event`` is invoked with ``(kind, payload)``. ``payload`` is a
        shallow-copied dict per beat with ``elapsed_s`` overwritten to the
        wall-clock time since this helper was entered, so listeners can
        render progress without doing their own bookkeeping.
      * Exceptions from ``on_event`` are swallowed (best-effort, like
        every other event sink in the loop) to avoid masking the real
        result of ``coro``.

    Args:
        coro: The work coroutine. Its result is returned.
        on_event: Callable that accepts ``(kind, payload)``. May be sync
            or async; both are handled.
        kind: Event kind string. Defaults to ``"heartbeat"``.
        payload: Static fields included in every beat. ``elapsed_s`` is
            always overwritten by this helper.
        interval: Seconds between successive beats.
        first_after: Seconds before the first beat. Defaults to ``interval``.
    """
    if on_event is None or interval <= 0:
        return await coro

    payload_base = dict(payload or {})
    delay_first = interval if first_after is None else max(0.0, first_after)
    start = time.monotonic()

    async def _emit_one() -> None:
        beat = dict(payload_base)
        beat["elapsed_s"] = round(time.monotonic() - start, 3)
        try:
            res = on_event(kind, beat)
            if asyncio.iscoroutine(res):
                await res
        except Exception:  # best effort
            pass

    async def _heartbeat_loop() -> None:
        try:
            await asyncio.sleep(delay_first)
            while True:
                await _emit_one()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    work_task = asyncio.ensure_future(coro)
    beat_task = asyncio.ensure_future(_heartbeat_loop())
    try:
        return await work_task
    finally:
        # If we're exiting because the *caller* was cancelled (not because
        # the work finished), propagate the cancellation to the work task —
        # otherwise it kept running (and mutating the filesystem) after the
        # agent was cancelled.
        if not work_task.done():
            work_task.cancel()
            try:
                await work_task
            except (asyncio.CancelledError, Exception):
                pass
        beat_task.cancel()
        try:
            await beat_task
        except (asyncio.CancelledError, Exception):
            pass
