"""Deferred post-step bookkeeping overlapped with LLM network wait.

Inspired by DeepInfer's ``DeferredRelease`` (``deepinfer/engine/llm_engine.py``),
which collects step-output cleanup and telemetry work during step *t* and executes
it overlapped with device execution on step *t + 1* so post-step I/O never adds
tail latency between tool execution and the next model request.

Safety contract (per ``AGENTS.md`` §7):
* Never used for causal model-visible observations or failure receipts — only for
  out-of-band persistence/telemetry (``session_writer.write_usage``, post-LLM
  observer hooks, Context Observatory / trajectory flush callbacks).
* Every queued callback is deterministically drained before ``RunFinalizer``
  completes (and on cancellation/error) via :meth:`DeferredStepFlush.drain`.
* Exceptions in deferred callbacks are logged and isolated so one slow or broken
  telemetry sink cannot crash the agent turn.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

DeferredCallback = Callable[[], Any]


class DeferredStepFlush:
    """Collect post-turn bookkeeping callbacks and flush them off the critical path."""

    def __init__(self, *, max_pending: int = 64) -> None:
        self._max_pending = max(1, int(max_pending))
        self._queue: deque[DeferredCallback] = deque()
        self._in_flight: set[asyncio.Task[None]] = set()
        self._flushed_count = 0
        self._error_count = 0

    @property
    def pending_count(self) -> int:
        return len(self._queue) + len(self._in_flight)

    @property
    def flushed_count(self) -> int:
        return self._flushed_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def defer(self, fn: DeferredCallback) -> None:
        """Enqueue a sync or async zero-arg callback to run during the next wait window."""
        self._queue.append(fn)
        if len(self._queue) >= self._max_pending:
            self.kick()

    def kick(self) -> asyncio.Task[None] | None:
        """Schedule all queued callbacks as a background task overlapped with I/O wait."""
        if not self._queue:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop: execute synchronous callbacks inline.
            self._drain_sync_only()
            return None

        batch = list(self._queue)
        self._queue.clear()
        task = loop.create_task(self._run_batch(batch))
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
        return task

    def _drain_sync_only(self) -> None:
        while self._queue:
            fn = self._queue.popleft()
            try:
                res = fn()
                if inspect.isawaitable(res):
                    # Close unawaited coroutine cleanly when no loop is running.
                    close = getattr(res, "close", None)
                    if callable(close):
                        close()
                self._flushed_count += 1
            except Exception:
                self._error_count += 1
                logger.debug("DeferredStepFlush sync callback failed", exc_info=True)

    async def _run_batch(self, batch: list[DeferredCallback]) -> None:
        for fn in batch:
            try:
                res = fn()
                if inspect.isawaitable(res):
                    await res
                else:
                    # Yield once so pure-sync disk writes interleave cleanly with socket reads.
                    await asyncio.sleep(0)
                self._flushed_count += 1
            except Exception:
                self._error_count += 1
                logger.debug("DeferredStepFlush callback failed", exc_info=True)

    async def drain(self) -> None:
        """Flush any remaining queued items and await all in-flight tasks."""
        while self._queue or self._in_flight:
            if self._queue:
                batch = list(self._queue)
                self._queue.clear()
                await self._run_batch(batch)
            if self._in_flight:
                pending = tuple(self._in_flight)
                await asyncio.gather(*pending, return_exceptions=True)


__all__ = [
    "DeferredCallback",
    "DeferredStepFlush",
]
