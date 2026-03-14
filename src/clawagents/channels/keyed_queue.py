"""
Per-key serialization queue.

Ensures that async work for the same key (e.g. a conversation session)
is executed sequentially, while different keys run in parallel.
Inspired by OpenClaw's KeyedAsyncQueue.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable


class KeyedAsyncQueue:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def enqueue(self, key: str, task: Callable[[], Awaitable[None]]) -> None:
        lock = self._locks[key]
        async with lock:
            await task()

    @property
    def active_keys(self) -> int:
        return sum(1 for lock in self._locks.values() if lock.locked())
