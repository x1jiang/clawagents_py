"""Cancelled foreground execute must not leave an orphan subprocess."""

from __future__ import annotations

import asyncio
import os

import pytest

from clawagents.background import BackgroundJobManager
from clawagents.tools.exec import _exec_foreground_with_autobg


@pytest.mark.asyncio
async def test_cancelled_foreground_kills_process_group():
    mgr = BackgroundJobManager()
    # Long sleep in a new session so we can observe killpg.
    task = asyncio.create_task(
        _exec_foreground_with_autobg(
            "sleep 60",
            cwd=os.getcwd(),
            timeout_ms=60_000,
            mgr=mgr,
            streaming=False,
        )
    )
    await asyncio.sleep(0.15)
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Give the OS a beat to reap; no running sleep children from this test.
    await asyncio.sleep(0.1)
    # Manager should not have adopted a background job on cancel.
    assert mgr.list() == []
