"""Background jobs with optional notify-on-complete callbacks.

Long-running shell commands (test suites, builds, deploys, model training,
data pipelines) shouldn't block the agent loop. This module provides a
small, framework-agnostic primitive for running subprocesses in the
background and getting notified exactly once when they exit.

Design goals
------------
- **Cheap**: pure ``asyncio.subprocess``, no extra dependencies.
- **Inspectable**: each job has a stable id, captured stdout/stderr (in
  memory by default), exit code, and timestamps.
- **One-shot notification**: ``notify_on_complete`` fires exactly once
  when the process exits. The callback is invoked from the watcher task,
  so it should be cheap and non-blocking — push a message onto a queue,
  enqueue a follow-up turn, or write to a session log.
- **Cooperative cancel**: ``cancel(job_id)`` sends ``SIGTERM`` and then
  ``SIGKILL`` after a small grace period.

Mirrors ``clawagents/src/background.ts``.

Hermes ships a much richer terminal tool with PTY, watch-patterns, rate
limits, and Docker/Singularity/Modal/Daytona backends; this module is
intentionally a smaller primitive that the rest of the agent (or a
custom tool) can build on top of.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Sequence

JobNotifier = Callable[["BackgroundJob"], Optional[Awaitable[None]]]
"""Callback signature. Returning a coroutine awaits it; otherwise sync."""


@dataclass
class BackgroundJob:
    """Snapshot-style record describing a background job.

    Attributes:
        id: Stable, unique identifier (callers can keep this around).
        command: The argv that was launched.
        cwd: Working directory passed to ``asyncio.create_subprocess_exec``.
        pid: OS pid (``None`` until the process has been spawned).
        started_at: Wall-clock seconds when the job was started.
        ended_at: Wall-clock seconds when the job exited, else ``None``.
        exit_code: Process exit code (``None`` while running).
        stdout: Captured stdout bytes (decoded as UTF-8 best-effort).
        stderr: Captured stderr bytes.
        cancelled: True if cancellation was requested.
    """

    id: str
    command: list[str]
    cwd: Optional[str]
    pid: Optional[int] = None
    started_at: float = 0.0
    ended_at: Optional[float] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    cancelled: bool = False
    _process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    _watcher: Optional[asyncio.Task[None]] = field(default=None, repr=False)
    _done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def running(self) -> bool:
        """True while the process is alive."""
        return self.exit_code is None and not self._done_event.is_set()


class BackgroundJobManager:
    """Process-wide registry for :class:`BackgroundJob` instances.

    Construct one per agent run. ``start`` spawns a subprocess and
    returns the :class:`BackgroundJob` immediately; the watcher task
    runs in the background, captures output, fires the optional
    ``notify_on_complete`` callback once the process exits, and updates
    the job record in place.
    """

    def __init__(self, *, kill_grace_seconds: float = 2.0) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._kill_grace = kill_grace_seconds

    async def start(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        notify_on_complete: Optional[JobNotifier] = None,
        capture_output: bool = True,
        job_id: Optional[str] = None,
    ) -> BackgroundJob:
        """Launch a subprocess and return the live :class:`BackgroundJob`.

        Args:
            command: argv list. The first element is the program; the
                rest are arguments. We never go through a shell, so
                command injection isn't a concern at this layer.
            cwd: Working directory for the subprocess.
            env: Environment dict. ``None`` inherits :data:`os.environ`.
            notify_on_complete: Fires exactly once when the process
                exits, with the (now-final) :class:`BackgroundJob`.
            capture_output: When ``True`` (default), stdout/stderr are
                buffered onto the job record. When ``False``, both are
                inherited from the parent — useful for very chatty
                processes whose output you don't want in memory.
            job_id: Override the auto-generated id (mainly for tests).

        Raises:
            FileNotFoundError: If the program isn't on PATH.
        """
        if not command:
            raise ValueError("BackgroundJobManager.start: empty command")

        jid = job_id or uuid.uuid4().hex
        if jid in self._jobs:
            raise ValueError(f"BackgroundJobManager.start: duplicate job_id {jid!r}")

        stdout = asyncio.subprocess.PIPE if capture_output else None
        stderr = asyncio.subprocess.PIPE if capture_output else None

        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

        job = BackgroundJob(
            id=jid,
            command=list(command),
            cwd=cwd,
            pid=proc.pid,
            started_at=time.time(),
            _process=proc,
        )
        self._jobs[jid] = job

        async def _watch() -> None:
            try:
                if capture_output and proc.stdout and proc.stderr:
                    out_bytes, err_bytes = await proc.communicate()
                    job.stdout = (out_bytes or b"").decode("utf-8", errors="replace")
                    job.stderr = (err_bytes or b"").decode("utf-8", errors="replace")
                else:
                    await proc.wait()
                job.exit_code = proc.returncode
            except asyncio.CancelledError:
                job.cancelled = True
                raise
            finally:
                job.ended_at = time.time()
                job._done_event.set()
                if notify_on_complete is not None:
                    try:
                        result = notify_on_complete(job)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:  # noqa: BLE001 — never let a callback kill us
                        pass

        job._watcher = asyncio.create_task(_watch(), name=f"bgjob-watch-{jid}")
        return job

    def status(self, job_id: str) -> BackgroundJob:
        """Return the :class:`BackgroundJob` record (mutates in place)."""
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"unknown background job_id {job_id!r}") from None

    def list(self) -> list[BackgroundJob]:
        """Return all known jobs (running + completed)."""
        return list(self._jobs.values())

    async def await_complete(
        self,
        job_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> BackgroundJob:
        """Wait until the job exits (or ``timeout`` elapses)."""
        job = self.status(job_id)
        if job._done_event.is_set():
            return job
        await asyncio.wait_for(job._done_event.wait(), timeout=timeout)
        return job

    async def cancel(self, job_id: str) -> BackgroundJob:
        """Request cancellation. Sends SIGTERM, then SIGKILL after a grace."""
        job = self.status(job_id)
        proc = job._process
        if proc is None or proc.returncode is not None:
            return job
        job.cancelled = True
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return job
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._kill_grace)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        return job

    async def shutdown(self) -> None:
        """Cancel every still-running job and wait for the watchers."""
        for job in list(self._jobs.values()):
            if job.running:
                await self.cancel(job.id)
            if job._watcher is not None and not job._watcher.done():
                try:
                    await job._watcher
                except asyncio.CancelledError:
                    pass

    # Keep file-handle leakage diagnostics easy to chase in tests.
    def __len__(self) -> int:
        return len(self._jobs)


__all__ = [
    "BackgroundJob",
    "BackgroundJobManager",
    "JobNotifier",
]


# Silence "unused" linters on Linux where SIGTERM is unconditionally used.
_ = os.name
