"""A backgrounded ``execute`` must not lose its result.

``execute`` can go to the background two ways: explicitly (``is_background``)
or by being adopted when its foreground wait times out. Both hand back only a
``job_id``, so unless the harness keeps following that job, the run can end on
"started it" and the actual exit code is never seen by anyone. These tests pin
the three things that make the result come back:

* the job is *owned*, so its completion is announced unprompted,
* ``task_wait`` exists, so the model can stay on the job instead of ending its
  turn,
* waiting twice actually waits twice, rather than replaying a cached answer.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from clawagents.background import BackgroundJobManager
from clawagents.graph.loop_tracker import _ToolCallTracker
from clawagents.tools.background_task import (
    _TaskCreateTool,
    _TaskWaitTool,
    create_background_task_tools,
)
from clawagents.tools.registry import ToolRegistry, ToolResult, _extract_job_id


class _Echo:
    name = "echo"
    description = "echo"
    keywords: list[str] = []
    parameters: dict = {}

    async def execute(self, args):
        return ToolResult(True, "echo output")


@pytest.fixture
def bg_enabled(monkeypatch: pytest.MonkeyPatch):
    """Background execute + auto-background are feature-gated."""
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]
    yield
    feat._resolved = None  # type: ignore[attr-defined]


def _exec_registry(manager: BackgroundJobManager) -> ToolRegistry:
    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    reg = ToolRegistry()
    reg.register(_Echo())
    reg.register(ExecTool(LocalBackend()))
    # task_create supplies the manager _append_background_notice queries.
    reg.register(_TaskCreateTool(manager))
    reg.register(_TaskWaitTool(manager))
    return reg


class _Ctx:
    """Minimal run_context: exec binds its job manager onto this."""

    def __init__(self, manager: BackgroundJobManager) -> None:
        self.background_manager = manager


# ─── job id extraction ────────────────────────────────────────────────────


def test_job_id_survives_the_warning_prefix():
    """execute prefixes its JSON payload with sandbox/session warnings."""
    body = '[shell_session: cwd=/tmp]\n{\n  "backgrounded": true,\n  "job_id": "abc123"\n}'
    assert _extract_job_id(body) == "abc123"


def test_job_id_survives_output_truncation():
    """A clipped payload no longer parses as JSON, but the id is still there."""
    assert _extract_job_id('{"backgrounded": true, "job_id": "abc123", "cwd": "/tm') == "abc123"


def test_unrelated_output_yields_no_job_id():
    assert _extract_job_id("total 4\ndrwxr-xr-x  2 me  staff") is None
    assert _extract_job_id(None) is None


# ─── ownership ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_explicit_background_execute_announces_its_completion(bg_enabled):
    """Previously only task_create was tracked, so this notice never fired."""
    manager = BackgroundJobManager()
    reg = _exec_registry(manager)

    started = await reg.execute_tool(
        "execute",
        {"command": "echo bg-done", "is_background": True},
        run_context=_Ctx(manager),
    )
    assert started.success, started.error
    job_id = _extract_job_id(started.output)
    assert job_id, started.output
    assert job_id in reg.owned_job_ids()

    await manager.await_complete(job_id, timeout=10)
    assert "Background job(s) finished" in (await reg.execute_tool("echo", {})).output


@pytest.mark.asyncio
async def test_auto_backgrounded_execute_announces_its_completion(bg_enabled):
    """The timeout-adoption path is where a result goes missing most quietly."""
    manager = BackgroundJobManager()
    reg = _exec_registry(manager)

    started = await reg.execute_tool(
        "execute",
        {"command": "sleep 0.6; echo late-done", "block_until_ms": 150},
        run_context=_Ctx(manager),
    )
    assert started.success, started.error
    job_id = _extract_job_id(started.output)
    assert job_id, started.output

    await manager.await_complete(job_id, timeout=10)
    notice = (await reg.execute_tool("echo", {})).output
    assert "Background job(s) finished" in notice


@pytest.mark.asyncio
async def test_ownership_can_be_restored_for_a_rebuilt_registry():
    """Hosts that build a fresh agent per turn must not drop the job."""
    manager = BackgroundJobManager()
    first = _exec_registry(manager)
    created = await first.execute_tool("task_create", {"command": ["/bin/sh", "-c", "exit 0"]})
    job_id = json.loads(created.raw_output)["job_id"]
    await manager.await_complete(job_id, timeout=10)

    next_turn = _exec_registry(manager)
    assert "Background job(s) finished" not in (await next_turn.execute_tool("echo", {})).output

    next_turn.adopt_owned_jobs([job_id])
    assert "Background job(s) finished" in (await next_turn.execute_tool("echo", {})).output


# ─── task_wait ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_wait_returns_exit_code_and_output():
    manager = BackgroundJobManager()
    wait = _TaskWaitTool(manager)
    job = await manager.start(["/bin/sh", "-c", "echo hello; exit 7"])

    result = await wait.execute({"job_id": job.id})
    assert result.success
    payload = json.loads(result.output[: result.output.index("\n\nstdout:")])
    assert payload["exit_code"] == 7
    assert payload["timed_out"] is False
    assert payload["running"] is False
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_task_wait_reports_a_still_running_job_without_killing_it():
    manager = BackgroundJobManager()
    wait = _TaskWaitTool(manager)
    job = await manager.start(["/bin/sh", "-c", "sleep 30"])
    try:
        result = await wait.execute({"job_id": job.id, "timeout_ms": 1000})
        payload = json.loads(result.output[: result.output.index("\n\nstdout:")])
        assert result.success  # "still running" is a fact, not a failure
        assert payload["timed_out"] is True
        assert payload["running"] is True
        assert manager.status(job.id).running
    finally:
        await manager.cancel(job.id)


@pytest.mark.asyncio
async def test_task_wait_is_registered_alongside_the_other_task_tools():
    assert "task_wait" in {t.name for t in create_background_task_tools()}


def test_an_empty_job_manager_is_still_truthy():
    """__len__ would otherwise make a fresh manager falsy, and every
    ``manager or default`` fallback would quietly use the wrong registry."""
    assert bool(BackgroundJobManager()) is True
    assert len(BackgroundJobManager()) == 0


@pytest.mark.asyncio
async def test_tools_use_the_manager_they_were_given_before_it_has_jobs():
    """Jobs started on one manager must not be looked up on another."""
    manager = BackgroundJobManager()
    tools = {t.name: t for t in create_background_task_tools(manager)}
    job = await manager.start(["/bin/sh", "-c", "echo mine"])
    await manager.await_complete(job.id, timeout=10)

    result = await tools["task_output"].execute({"job_id": job.id})
    assert result.success, result.error
    assert "mine" in result.output


@pytest.mark.asyncio
async def test_task_wait_rejects_an_unknown_job():
    result = await _TaskWaitTool(BackgroundJobManager()).execute({"job_id": "nope"})
    assert not result.success


@pytest.mark.asyncio
async def test_task_output_keeps_the_tail_of_a_chatty_job():
    """The verdict of a long build is at the end, and dumps must stay bounded."""
    manager = BackgroundJobManager()
    out = next(t for t in create_background_task_tools(manager) if t.name == "task_output")
    job = await manager.start(["/bin/sh", "-c", "seq 1 5000"])
    await manager.await_complete(job.id, timeout=20)

    result = await out.execute({"job_id": job.id, "max_chars": 400})
    assert "5000" in result.output  # tail kept
    assert "earlier chars omitted" in result.output
    assert len(result.output) < 2000


# ─── repeat-suppression must not swallow a wait ───────────────────────────


def test_waiting_again_is_not_served_from_the_duplicate_cache():
    """"Call task_wait again to keep waiting" only works if it really runs."""
    tracker = _ToolCallTracker()
    args = {"job_id": "j1"}

    tracker.record_result("task_wait", args, "still running")
    assert tracker.reuse_tool_output("task_wait", args) is None

    # Contrast: a normal tool is still deduped within a turn.
    tracker.record_result("read_file", {"path": "a.py"}, "contents")
    assert tracker.reuse_tool_output("read_file", {"path": "a.py"}) is not None


def test_repeated_waiting_does_not_trip_the_loop_breaker():
    """A job outliving the per-call budget needs more calls than hard_limit."""
    tracker = _ToolCallTracker()
    args = {"job_id": "j1"}
    for _ in range(12):
        tracker.record("task_wait", args)

    assert not tracker.is_soft_looping("task_wait", args)
    assert not tracker.is_hard_looping("task_wait", args)

    # Contrast: an ordinary repeated call still trips it.
    for _ in range(12):
        tracker.record("execute", {"command": "ls"})
    assert tracker.is_hard_looping("execute", {"command": "ls"})


def test_a_stalled_job_still_trips_the_no_progress_breaker():
    """Exempting waits from repeat counting must not remove all protection."""
    tracker = _ToolCallTracker(circuit_breaker_limit=5)
    for _ in range(8):
        tracker.record_result("task_wait", {"job_id": "j1"}, "identical stalled output")
    assert tracker.is_circuit_broken()


def test_task_wait_stays_inside_the_registry_tool_timeout():
    """A wait budget above the tool timeout would get killed mid-wait, so the
    caller would see a tool error instead of the job's real state."""
    from clawagents.tools.background_task import _DEFAULT_WAIT_MS, _MAX_WAIT_MS
    from clawagents.tools.registry import DEFAULT_TOOL_TIMEOUT_S

    assert _DEFAULT_WAIT_MS <= _MAX_WAIT_MS
    assert _MAX_WAIT_MS / 1000 < DEFAULT_TOOL_TIMEOUT_S


@pytest.mark.asyncio
async def test_task_wait_clamps_an_oversized_budget(monkeypatch: pytest.MonkeyPatch):
    """A model asking to wait "forever" must still return within the cap."""
    monkeypatch.setattr(
        "clawagents.tools.background_task._MAX_WAIT_MS", 1200, raising=True
    )
    manager = BackgroundJobManager()
    job = await manager.start(["/bin/sh", "-c", "sleep 30"])
    try:
        result = await asyncio.wait_for(
            _TaskWaitTool(manager).execute({"job_id": job.id, "timeout_ms": 10_000_000}),
            timeout=15,
        )
        payload = json.loads(result.output[: result.output.index("\n\nstdout:")])
        assert payload["timed_out"] is True
    finally:
        await manager.cancel(job.id)


# ─── the result has to be legible when it comes back ──────────────────────
#
# Carrying the result back is only half the job. What arrives is shaped by the
# shell-session wrapper exec puts around every command -- a cd, the command,
# then trailers that print cwd and env so the session can stay sticky. For a
# foreground run that wrapper is unwound before anyone sees it. A backgrounded
# run skips that path entirely, so without care the user is shown a status row
# describing a `cd` and output ending in shell bookkeeping.


@pytest.mark.asyncio
async def test_a_backgrounded_job_is_described_by_the_command_asked_for(bg_enabled):
    """Not by the wrapper it became.

    ``command`` is truncated for display, and the wrapper front-loads the
    boring part, so without a label the useful text is exactly what gets cut.
    """
    manager = BackgroundJobManager()
    reg = _exec_registry(manager)

    started = await reg.execute_tool(
        "execute",
        {"command": "echo needle-cmd", "is_background": True},
        run_context=_Ctx(manager),
    )
    job_id = _extract_job_id(started.output)
    assert job_id, started.output

    assert manager.status(job_id).display_command == "echo needle-cmd"

    await manager.await_complete(job_id, timeout=10)
    notice = (await reg.execute_tool("echo", {})).output
    assert "echo needle-cmd" in notice
    assert "__CLAW_PWD__" not in notice


@pytest.mark.asyncio
async def test_the_auto_background_path_is_labelled_too(bg_enabled):
    """Adoption happens deep in the foreground path; the label has to reach it."""
    manager = BackgroundJobManager()
    reg = _exec_registry(manager)

    started = await reg.execute_tool(
        "execute",
        {"command": "sleep 0.6; echo adopted-needle", "block_until_ms": 150},
        run_context=_Ctx(manager),
    )
    job_id = _extract_job_id(started.output)
    assert job_id, started.output
    assert manager.status(job_id).display_command == "sleep 0.6; echo adopted-needle"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"command": "echo clean-needle", "is_background": True},
        {"command": "sleep 0.5; echo clean-needle", "block_until_ms": 150},
    ],
    ids=["explicit", "adopted"],
)
async def test_job_output_carries_no_shell_bookkeeping(bg_enabled, args):
    """Whatever the wrapper appended is the wrapper's business, not the user's."""
    manager = BackgroundJobManager()
    reg = _exec_registry(manager)

    started = await reg.execute_tool("execute", args, run_context=_Ctx(manager))
    job_id = _extract_job_id(started.output)
    assert job_id, started.output

    job = await manager.await_complete(job_id, timeout=10)
    assert "clean-needle" in job.stdout
    assert "__CLAW_PWD__" not in job.stdout
    assert "__CLAW_ENV__" not in job.stdout
    assert "__SQZ_CMD" not in job.stdout

    waited = await reg.execute_tool("task_wait", {"job_id": job_id})
    assert "clean-needle" in waited.output
    assert "__CLAW_PWD__" not in waited.output


def test_stripping_trailers_leaves_an_ordinary_job_alone():
    """The filter runs on every background job, including unwrapped ones."""
    from clawagents.tools.shell_session import strip_session_trailers

    for text in ("", "plain output\n", "a line with __CLAW_PWD__-ish text but no trailer\n"):
        assert strip_session_trailers(text) == text


def test_stripping_trailers_does_not_move_the_callers_shell(tmp_path):
    """A job finishing later must not retroactively cd the live session.

    ``consume_stdout`` exists to adopt that state, which is right for a
    foreground command and wrong here -- hence a separate entry point.
    """
    import os

    from clawagents.tools.shell_session import (
        ENV_MARKER,
        PWD_MARKER,
        ShellSession,
        strip_session_trailers,
    )

    session = ShellSession(cwd=str(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    trailer = f"real output\n{PWD_MARKER}{elsewhere}\n{ENV_MARKER}{{}}\n"

    assert strip_session_trailers(trailer) == "real output\n"
    assert session.cwd == str(tmp_path)

    # For contrast: the foreground entry point is *supposed* to move it.
    cwd_before = os.getcwd()
    try:
        assert session.consume_stdout(trailer) == "real output\n"
        assert session.cwd == str(elsewhere.resolve())
    finally:
        os.chdir(cwd_before)


def test_a_job_without_a_label_still_describes_itself():
    """task_create passes argv and no label; display must not come back empty."""
    from clawagents.background import BackgroundJob

    job = BackgroundJob(id="j", command=["pytest", "-q"], cwd=None)
    assert job.display_command == "pytest -q"


def test_output_that_merely_mentions_a_marker_is_not_truncated(tmp_path):
    """Trailer detection keys off a path, not the marker alone.

    A mid-line marker used to be assumed to be a trailer for a command that
    printed no newline, which meant a command echoing the marker text -- say,
    grepping this repo -- lost everything after it.
    """
    from clawagents.tools.shell_session import (
        PWD_MARKER,
        ShellSession,
        strip_session_trailers,
    )

    noise = f'printf "%s" "{PWD_MARKER}"  # not a real trailer\nmore output\n'
    assert strip_session_trailers(noise) == noise
    assert ShellSession(cwd=str(tmp_path)).consume_stdout(noise) == noise

    # A genuine no-trailing-newline trailer must still be peeled.
    assert strip_session_trailers(f"no newline here{PWD_MARKER}{tmp_path}\n") == (
        "no newline here"
    )
