"""Kill a subprocess together with everything it spawned."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

# How long to keep reading a finished command's output pipes. A command such
# as ``server & echo started`` exits at once but leaves a background child
# holding the pipe open; reading to EOF would block until that child dies.
PIPE_DRAIN_GRACE_S = 1.0

PIPE_HELD_NOTE = (
    "[output may be incomplete: a background process started by this command "
    "is still holding the output pipe open]"
)


async def wait_for_exit(proc: Any, poll_s: float = 0.05) -> int:
    """Wait until ``proc`` itself has exited and return its exit code.

    ``asyncio.subprocess.Process.wait()`` is not enough: asyncio resolves it
    only once the process has exited *and* every pipe has closed, so a
    background child that inherited stdout keeps ``wait()`` pending long after
    the command finished. ``returncode`` is set as soon as the process is
    reaped, so poll that instead.
    """
    while proc.returncode is None:
        await asyncio.sleep(poll_s)
    return int(proc.returncode)


def kill_process_tree(proc: Any) -> None:
    """SIGKILL ``proc``'s process group, falling back to the process itself.

    Subprocesses here are started with ``start_new_session=True``, so the
    shell and every child it spawned share one group. Never raises: the
    process may already be gone, and ``os.killpg`` does not exist on Windows.
    """
    pid = getattr(proc, "pid", None)
    killpg = getattr(os, "killpg", None)
    if pid is not None and killpg is not None:
        try:
            killpg(pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except Exception:  # already gone, or not a process object at all
        pass
