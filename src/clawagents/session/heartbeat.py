"""Session heartbeat and auto-cleanup.

Sessions without heartbeat auto-release resources after timeout.
"""
import asyncio
import time
from typing import Callable


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
