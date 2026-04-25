"""Async scheduler that drives a :class:`JobRunner` from due jobs.

The scheduler is intentionally tiny: it polls :func:`get_due_jobs` on
an interval, hands each due job to a user-supplied
:class:`JobRunner`, and persists the result via :func:`mark_job_run`
(success or failure) and :func:`save_job_output`.

Two operating modes:

- **In-process**: ``await scheduler.start()`` runs the poll loop as a
  background task and ``await scheduler.stop()`` cancels it.
- **One-shot tick**: ``await scheduler.tick()`` runs every currently
  due job once and returns. Useful for the ``clawagents cron tick``
  CLI subcommand and for tests.

Crash recovery: before each run we call :func:`advance_next_run` so
that a mid-run crash does not re-fire the same recurring job on
restart. One-shot jobs are left untouched for retry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Protocol

from clawagents.cron.errors import SchedulerError
from clawagents.cron.jobs import (
    advance_next_run,
    get_due_jobs,
    mark_job_run,
    save_job_output,
)

logger = logging.getLogger("clawagents.cron.scheduler")


class JobRunner(Protocol):
    """Caller-supplied executor that turns a ``job`` into output text."""

    async def __call__(self, job: dict) -> str:  # noqa: D401
        ...


JobRunnerFn = Callable[[dict], Awaitable[str]]


class Scheduler:
    """Polls :func:`get_due_jobs` and dispatches them through a runner.

    Parameters
    ----------
    runner:
        Async callable invoked with the job dict; returns the textual
        output to persist alongside the run record.
    interval_seconds:
        Polling interval. Default 30 seconds (matches Hermes default).
    save_output:
        When ``True`` (default), runner output is persisted via
        :func:`save_job_output`. Set ``False`` for ephemeral pings
        such as health checks.
    """

    def __init__(
        self,
        runner: JobRunnerFn,
        *,
        interval_seconds: float = 30.0,
        save_output: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise SchedulerError("interval_seconds must be > 0")
        self._runner = runner
        self._interval = interval_seconds
        self._save_output = save_output
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._last_tick_at: float | None = None
        self._tick_count: int = 0
        self._error_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def stats(self) -> dict:
        return {
            "running": self.is_running,
            "interval_seconds": self._interval,
            "tick_count": self._tick_count,
            "error_count": self._error_count,
            "last_tick_at": self._last_tick_at,
        }

    async def start(self) -> None:
        """Start the poll loop. Idempotent."""
        if self.is_running:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="clawagents.cron.scheduler")

    async def stop(self) -> None:
        """Stop the poll loop. Idempotent and safe from outside the loop."""
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        finally:
            self._task = None

    async def tick(self) -> int:
        """Run every currently-due job once. Returns count of jobs fired."""
        due = get_due_jobs()
        for job in due:
            await self._dispatch(job)
        self._tick_count += 1
        self._last_tick_at = time.time()
        return len(due)

    async def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                try:
                    await self.tick()
                except Exception as exc:  # noqa: BLE001
                    self._error_count += 1
                    logger.exception("scheduler tick failed: %s", exc)
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(), timeout=self._interval
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

    async def _dispatch(self, job: dict) -> None:
        job_id = job["id"]
        kind = job.get("schedule", {}).get("kind")
        if kind in ("cron", "interval"):
            advance_next_run(job_id)
        try:
            output = await self._runner(job)
        except Exception as exc:  # noqa: BLE001
            self._error_count += 1
            logger.exception("Job %s failed: %s", job_id, exc)
            mark_job_run(job_id, success=False, error=str(exc))
            return
        if self._save_output and output:
            try:
                save_job_output(job_id, output)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save output for %s: %s", job_id, exc)
        mark_job_run(job_id, success=True)
