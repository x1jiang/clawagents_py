"""Tests for clawagents.background."""

from __future__ import annotations

import asyncio
import sys

import pytest

from clawagents.background import BackgroundJob, BackgroundJobManager


pytestmark = pytest.mark.asyncio


async def test_short_command_completes_and_captures_output() -> None:
    mgr = BackgroundJobManager()
    job = await mgr.start([sys.executable, "-c", "print('hi'); import sys; sys.exit(0)"])
    final = await mgr.await_complete(job.id, timeout=10)
    assert final.exit_code == 0
    assert "hi" in final.stdout
    assert final.running is False
    assert final.ended_at is not None and final.ended_at >= final.started_at


async def test_notify_on_complete_fires_exactly_once() -> None:
    mgr = BackgroundJobManager()
    fired: list[BackgroundJob] = []

    def cb(job: BackgroundJob) -> None:
        fired.append(job)

    job = await mgr.start(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        notify_on_complete=cb,
    )
    await mgr.await_complete(job.id, timeout=10)
    # The watcher fires the callback inside its finally block, which can
    # run on the next loop tick. Yield once so the callback is observed.
    await asyncio.sleep(0)
    assert len(fired) == 1
    assert fired[0].id == job.id
    assert fired[0].exit_code == 0


async def test_notify_callback_can_be_async() -> None:
    mgr = BackgroundJobManager()
    fired: list[int] = []

    async def cb(job: BackgroundJob) -> None:
        await asyncio.sleep(0)
        fired.append(-1 if job.exit_code is None else job.exit_code)

    job = await mgr.start([sys.executable, "-c", "pass"], notify_on_complete=cb)
    await mgr.await_complete(job.id, timeout=10)
    # Allow the watcher's awaited callback to finish.
    for _ in range(10):
        if fired:
            break
        await asyncio.sleep(0.01)
    assert fired == [0]


async def test_failing_command_records_nonzero_exit_code() -> None:
    mgr = BackgroundJobManager()
    job = await mgr.start([sys.executable, "-c", "import sys; sys.exit(7)"])
    final = await mgr.await_complete(job.id, timeout=10)
    assert final.exit_code == 7


async def test_status_and_list_track_jobs() -> None:
    mgr = BackgroundJobManager()
    job = await mgr.start([sys.executable, "-c", "pass"])
    await mgr.await_complete(job.id, timeout=10)
    assert mgr.status(job.id) is job
    assert job in mgr.list()


async def test_status_unknown_id_raises() -> None:
    mgr = BackgroundJobManager()
    with pytest.raises(KeyError):
        mgr.status("nope")


async def test_cancel_stops_a_long_running_process() -> None:
    mgr = BackgroundJobManager(kill_grace_seconds=0.5)
    job = await mgr.start([sys.executable, "-c", "import time; time.sleep(30)"])
    await mgr.cancel(job.id)
    final = await mgr.await_complete(job.id, timeout=5)
    assert final.cancelled is True
    assert final.exit_code is not None  # something — SIGTERM-induced


async def test_cancel_idempotent_after_exit() -> None:
    mgr = BackgroundJobManager()
    job = await mgr.start([sys.executable, "-c", "pass"])
    await mgr.await_complete(job.id, timeout=10)
    again = await mgr.cancel(job.id)
    assert again.exit_code == 0


async def test_callback_exceptions_do_not_break_watcher() -> None:
    mgr = BackgroundJobManager()

    def boom(_: BackgroundJob) -> None:
        raise RuntimeError("callback error")

    job = await mgr.start([sys.executable, "-c", "pass"], notify_on_complete=boom)
    final = await mgr.await_complete(job.id, timeout=10)
    assert final.exit_code == 0


async def test_duplicate_job_id_is_rejected() -> None:
    mgr = BackgroundJobManager()
    await mgr.start([sys.executable, "-c", "pass"], job_id="abc")
    with pytest.raises(ValueError):
        await mgr.start([sys.executable, "-c", "pass"], job_id="abc")


async def test_empty_command_is_rejected() -> None:
    mgr = BackgroundJobManager()
    with pytest.raises(ValueError):
        await mgr.start([])


async def test_shutdown_cancels_outstanding_jobs() -> None:
    mgr = BackgroundJobManager(kill_grace_seconds=0.5)
    await mgr.start([sys.executable, "-c", "import time; time.sleep(30)"])
    await mgr.shutdown()
    for j in mgr.list():
        assert not j.running
