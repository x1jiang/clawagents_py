"""Regression tests: bounded output drain, kill-on-cancel, cancellable lane
commands and WebSocket turns, retried-stream de-duplication, and the bwrap
wrapper never creating files in the workspace."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from clawagents.background import BackgroundJobManager
from clawagents.process.command_queue import (
    enqueue_command_in_lane,
    get_lane_state,
    get_queue_size,
)
from clawagents.providers.llm import _StreamReplayGuard
from clawagents.sandbox.local import LocalBackend
from clawagents.tools.exec import _exec_foreground_with_autobg
from clawagents.utils.proc_kill import PIPE_HELD_NOTE, kill_process_tree

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


async def _wait_gone(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


def _read_pid(path: Path, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = path.read_text().strip() if path.exists() else ""
        if text:
            return int(text)
        time.sleep(0.02)
    raise AssertionError(f"{path} never got a pid")


# ── 1. A background child holding the pipe must not hang the command ─────


@posix_only
async def test_execute_returns_when_background_child_holds_pipe(tmp_path):
    pidfile = tmp_path / "bg.pid"
    started = time.monotonic()
    result, backgrounded, job_id = await _exec_foreground_with_autobg(
        f"(sleep 30 & echo $! > {pidfile}); echo started",
        cwd=str(tmp_path),
        timeout_ms=20_000,
        mgr=BackgroundJobManager(),
    )
    elapsed = time.monotonic() - started
    pid = _read_pid(pidfile)
    try:
        assert elapsed < 6, f"blocked {elapsed:.1f}s on a pipe held by a background child"
        assert not backgrounded and job_id is None
        assert "started" in result.stdout
        assert result.exit_code == 0
        assert PIPE_HELD_NOTE in result.stderr
        assert _alive(pid), "the deliberately backgrounded process is left running"
    finally:
        os.kill(pid, signal.SIGKILL)


@posix_only
async def test_execute_normal_command_has_no_pipe_note(tmp_path):
    result, _, _ = await _exec_foreground_with_autobg(
        "echo out; echo err 1>&2",
        cwd=str(tmp_path),
        timeout_ms=10_000,
        mgr=BackgroundJobManager(),
    )
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


@posix_only
async def test_local_backend_returns_when_background_child_holds_pipe(tmp_path):
    pidfile = tmp_path / "bg.pid"
    started = time.monotonic()
    result = await LocalBackend(root=str(tmp_path)).exec(
        f"(sleep 30 & echo $! > {pidfile}); echo started", timeout=20_000
    )
    pid = _read_pid(pidfile)
    try:
        assert time.monotonic() - started < 6
        assert "started" in result.stdout
        assert result.exit_code == 0 and not result.killed
        assert PIPE_HELD_NOTE in result.stderr
    finally:
        os.kill(pid, signal.SIGKILL)


async def test_local_backend_plain_command_unchanged(tmp_path):
    result = await LocalBackend(root=str(tmp_path)).exec("echo hi", timeout=10_000)
    assert result.stdout.strip() == "hi"
    assert result.exit_code == 0 and not result.killed
    assert PIPE_HELD_NOTE not in result.stderr


async def test_local_backend_reports_nonzero_exit(tmp_path):
    result = await LocalBackend(root=str(tmp_path)).exec("exit 7", timeout=10_000)
    assert result.exit_code == 7


# ── 2. Cancellation and hook timeouts must kill the process ──────────────


@posix_only
async def test_local_backend_cancel_kills_process_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    task = asyncio.create_task(
        LocalBackend(root=str(tmp_path)).exec(
            f"sh -c 'echo $$ > {pidfile}; sleep 60'", timeout=60_000
        )
    )
    pid = await asyncio.to_thread(_read_pid, pidfile)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _wait_gone(pid), "cancelled command must not keep running"


@posix_only
async def test_local_backend_timeout_still_kills(tmp_path):
    pidfile = tmp_path / "child.pid"
    result = await LocalBackend(root=str(tmp_path)).exec(
        f"sh -c 'echo $$ > {pidfile}; sleep 60'", timeout=300
    )
    assert result.killed and result.exit_code == 1
    assert await _wait_gone(_read_pid(pidfile))


@posix_only
async def test_hook_timeout_kills_the_hook(tmp_path):
    from clawagents.hooks.external import run_hook

    pidfile = tmp_path / "hook.pid"
    out = await run_hook(f"sh -c 'echo $$ > {pidfile}; sleep 60'", {"a": 1}, timeout=0.4)
    assert out is None
    assert await _wait_gone(_read_pid(pidfile)), "timed-out hook must be killed"


@posix_only
async def test_hook_cancel_kills_the_hook(tmp_path):
    from clawagents.hooks.external import run_hook

    pidfile = tmp_path / "hook.pid"
    task = asyncio.create_task(
        run_hook(f"sh -c 'echo $$ > {pidfile}; sleep 60'", {}, timeout=60)
    )
    pid = await asyncio.to_thread(_read_pid, pidfile)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _wait_gone(pid)


async def test_hook_success_path_unchanged():
    from clawagents.hooks.external import run_hook

    cmd = f"{sys.executable} -c \"import sys, json; print(json.dumps({{'got': json.load(sys.stdin)['a']}}))\""
    assert await run_hook(cmd, {"a": 5}, timeout=20) == {"got": 5}


def test_kill_process_tree_never_raises():
    class Gone:
        pid = 2**22 + 12345  # not a live pid

        def kill(self):
            raise ProcessLookupError

    kill_process_tree(Gone())
    kill_process_tree(object())


# ── 3. Lane commands and WebSocket turns are cancellable ─────────────────


async def test_cancelling_the_wait_cancels_the_running_lane_command():
    lane = f"test-cancel-{time.time_ns()}"
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    waiter = asyncio.create_task(enqueue_command_in_lane(lane, work))
    await asyncio.wait_for(started.wait(), 2)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.wait_for(cancelled.wait(), 2)
    await asyncio.sleep(0)

    # The slot is released: the next command in this lane runs.
    assert get_lane_state(lane).active_task_ids == set()

    async def quick():
        return "ran"

    assert await asyncio.wait_for(enqueue_command_in_lane(lane, quick), 2) == "ran"
    assert get_queue_size(lane) == 0


async def test_cancelling_a_queued_lane_command_removes_it():
    lane = f"test-queued-{time.time_ns()}"
    release = asyncio.Event()
    ran: list[str] = []

    async def first():
        await release.wait()
        return "first"

    async def second():
        ran.append("second")

    first_waiter = asyncio.create_task(enqueue_command_in_lane(lane, first))
    await asyncio.sleep(0.02)
    second_waiter = asyncio.create_task(enqueue_command_in_lane(lane, second))
    await asyncio.sleep(0.02)
    second_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_waiter
    release.set()
    assert await asyncio.wait_for(first_waiter, 2) == "first"
    await asyncio.sleep(0.05)
    assert ran == [], "a command whose caller gave up before it started must not run"


@pytest.fixture
def ws_app(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))
    state = {"started": False, "cancelled": False, "finished": False}

    async def invoke(task, on_event=None, hooks=None):
        state["started"] = True
        try:
            if on_event:
                on_event("assistant_delta", {"delta": "partial"})
            await asyncio.sleep(30)
            state["finished"] = True
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    with patch("clawagents.gateway.ws.create_claw_agent") as mock_create, patch(
        "clawagents.gateway.server.create_claw_agent"
    ):
        agent = AsyncMock()
        agent.invoke.side_effect = invoke
        mock_create.return_value = agent
        from clawagents.gateway.server import create_app

        app, _, _ = create_app()
        yield app, state


def _poll(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_ws_disconnect_cancels_the_running_turn(ws_app):
    from fastapi.testclient import TestClient

    app, state = ws_app
    lane = "subagent"
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "req", "id": "1", "method": "chat.send",
                          "params": {"task": "long", "lane": lane}})
            assert _poll(lambda: state["started"]), "turn never started"
            # While the turn runs the socket still answers other requests.
            ws.send_json({"type": "req", "id": "2", "method": "ping", "params": {}})
            seen_pong = False
            for _ in range(20):
                frame = ws.receive_json()
                if frame.get("id") == "2":
                    seen_pong = frame.get("ok") is True
                    break
            assert seen_pong, "ping must be answered while chat.send is running"
        # Socket closed by the client.
        assert _poll(lambda: state["cancelled"]), "agent kept running after disconnect"
        assert state["finished"] is False
        assert _poll(lambda: get_lane_state(lane).active_task_ids == set()), "lane slot leaked"


# ── 4. A retried stream must not show the same text twice ────────────────


class _Recorder:
    def __init__(self, with_reset: bool = True):
        self.events: list[tuple[str, str]] = []
        if with_reset:
            self.reset = lambda: self.events.append(("reset", ""))

    def __call__(self, text: str) -> None:
        self.events.append(("text", text))

    @property
    def shown(self) -> str:
        out = ""
        for kind, text in self.events:
            out = "" if kind == "reset" else out + text
        return out


async def _play(guard: _StreamReplayGuard, *attempts: list[str]) -> None:
    for chunks in attempts:
        guard.begin_attempt()
        for chunk in chunks:
            await guard.feed(chunk)


async def test_single_attempt_passes_through_unchanged():
    rec = _Recorder()
    await _play(_StreamReplayGuard(rec), ["Hel", "lo ", "world"])
    assert rec.events == [("text", "Hel"), ("text", "lo "), ("text", "world")]


async def test_identical_retry_only_forwards_the_new_part():
    rec = _Recorder()
    await _play(_StreamReplayGuard(rec), ["The answer ", "is"], ["The ans", "wer is 42", "."])
    assert ("reset", "") not in rec.events
    assert rec.shown == "The answer is 42."
    assert "".join(t for k, t in rec.events if k == "text") == "The answer is 42."


async def test_diverging_retry_resets_then_replays():
    rec = _Recorder()
    await _play(_StreamReplayGuard(rec), ["The answer is 4"], ["The ", "result is 42."])
    assert rec.events.count(("reset", "")) == 1
    assert rec.shown == "The result is 42."


async def test_diverging_retry_without_reset_support_still_delivers_everything():
    rec = _Recorder(with_reset=False)
    await _play(_StreamReplayGuard(rec), ["abc"], ["xyz"])
    assert "".join(t for _, t in rec.events) == "abcxyz"  # legacy consumers: unchanged


async def test_third_attempt_is_compared_with_what_is_shown_now():
    rec = _Recorder()
    await _play(_StreamReplayGuard(rec), ["one two"], ["uno dos tres"], ["uno dos", " tres cuatro"])
    assert rec.events.count(("reset", "")) == 1
    assert rec.shown == "uno dos tres cuatro"


async def test_async_callback_and_broken_reset_are_tolerated():
    got: list[str] = []

    async def cb(text: str) -> None:
        got.append(text)

    def bad_reset():
        raise RuntimeError("consumer bug")

    cb.reset = bad_reset  # type: ignore[attr-defined]
    await _play(_StreamReplayGuard(cb), ["abc"], ["xyz"])
    assert got == ["abc", "xyz"]


async def test_none_callback_is_a_no_op():
    await _play(_StreamReplayGuard(None), ["a"], ["b"])


def test_every_stream_retry_loop_uses_the_guard():
    providers = Path(__file__).resolve().parents[1].joinpath("src/clawagents/providers")
    src = "".join((providers / f"{n}_provider.py").read_text() for n in ("openai", "gemini", "anthropic"))
    assert src.count("replay_guard = _StreamReplayGuard(on_chunk)") == 4
    assert src.count("replay_guard.begin_attempt()") == 4


def test_turn_emits_assistant_reset_event():
    src = Path(__file__).resolve().parents[1].joinpath("src/clawagents/graph/turn_llm.py").read_text()
    assert 'self._events.typed("assistant_reset"' in src
    assert "on_chunk.reset = _reset_stream" in src

    from clawagents.stream_events import stream_event_from_kind

    assert stream_event_from_kind("assistant_reset", {"reason": "stream_retry"}).kind == "assistant_reset"


# ── 6. The bwrap wrapper must not create files in the workspace ──────────


def test_bwrap_wrap_does_not_create_secret_placeholders(tmp_path):
    from clawagents.sandbox.profiles import OSSandboxProfile, ProfileBackend

    (tmp_path / "real.pem").write_text("key")
    backend = ProfileBackend(
        LocalBackend(root=str(tmp_path)),
        OSSandboxProfile(name="workspace", backend="bwrap", require_binary=False),
    )
    with patch("clawagents.sandbox.profiles.shutil.which", return_value="/usr/bin/bwrap"):
        wrapped = backend.wrap_command("echo hi", cwd=str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["real.pem"], "no placeholder files"
    assert str((tmp_path / "real.pem").resolve()) in wrapped, "existing secrets are still masked"
    assert ".env" not in wrapped
