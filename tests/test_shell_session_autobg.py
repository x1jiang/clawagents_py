"""Grok-inspired shell session cwd + auto-background-on-timeout."""

from __future__ import annotations

import asyncio
import json
import subprocess

import pytest

from clawagents.tools.shell_session import PWD_MARKER, ShellSession


def test_shell_session_wrap_and_consume(tmp_path):
    sess = ShellSession(cwd=str(tmp_path))
    wrapped = sess.wrap("echo hi")
    assert f"cd '{tmp_path}'" in wrapped or f'cd "{tmp_path}"' in wrapped or str(tmp_path) in wrapped
    assert PWD_MARKER in wrapped

    fake_out = f"hi\n{PWD_MARKER}{tmp_path}\n"
    clean = sess.consume_stdout(fake_out)
    assert clean == "hi\n"
    assert sess.cwd == str(tmp_path.resolve())


def test_shell_session_updates_on_cd(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    sess = ShellSession(cwd=str(tmp_path))
    out = f"{PWD_MARKER}{sub.resolve()}\n"
    sess.consume_stdout(out)
    assert sess.cwd == str(sub.resolve())


@pytest.mark.parametrize("sticky_env", [False, True])
def test_shell_session_wrap_preserves_quoted_heredoc(tmp_path, sticky_env):
    sub = tmp_path / "nested"
    sub.mkdir()
    sess = ShellSession(cwd=str(tmp_path))
    command = """cd nested && python3 - <<'PY'
from pathlib import Path
print(Path.cwd().name)
PY"""

    proc = subprocess.run(
        ["bash", "-c", sess.wrap(command, sticky_env=sticky_env)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "SyntaxError" not in proc.stderr
    assert sess.consume_stdout(proc.stdout, sticky_env=sticky_env) == "nested\n"
    assert sess.cwd == str(sub.resolve())


def test_shell_session_heredoc_preserves_user_exit_code(tmp_path):
    sess = ShellSession(cwd=str(tmp_path))
    command = """python3 - <<'PY'
raise SystemExit(7)
PY"""

    proc = subprocess.run(
        ["bash", "-c", sess.wrap(command)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 7
    assert "SyntaxError" not in proc.stderr
    assert sess.consume_stdout(proc.stdout) == ""


@pytest.mark.asyncio
async def test_execute_cwd_persists(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "0")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    class Ctx:
        pass

    ctx = Ctx()
    sub = tmp_path / "nested"
    sub.mkdir()
    tool = ExecTool(LocalBackend(root=str(tmp_path)))

    r1 = await tool.execute({"command": f"cd {sub.name}"}, run_context=ctx)
    assert r1.success, r1.error
    assert getattr(ctx, "shell_session").cwd == str(sub.resolve())

    # Relative write should land in nested/
    r2 = await tool.execute(
        {"command": "pwd && echo ok > marker.txt"},
        run_context=ctx,
    )
    assert r2.success, r2.error
    assert (sub / "marker.txt").is_file()


@pytest.mark.asyncio
async def test_execute_shell_session_runs_heredoc_without_corrupting_terminator(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_ENV", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "0")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_STREAMING", "0")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    class Ctx:
        pass

    tool = ExecTool(LocalBackend(root=str(tmp_path)))
    result = await tool.execute(
        {
            "command": """python3 - <<'PY'
print('heredoc-ok')
PY""",
        },
        run_context=Ctx(),
    )

    assert result.success, result.error
    assert str(result.output).endswith("heredoc-ok\n")
    assert "__claw_ec" not in str(result.output)


@pytest.mark.asyncio
async def test_execute_auto_background_on_timeout(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "0")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool
    from clawagents.tools.background_task import create_background_task_tools

    class Ctx:
        pass

    ctx = Ctx()
    tool = ExecTool(LocalBackend(root=str(tmp_path)))
    # Sleep longer than timeout — should auto-background, not hard-fail.
    r = await tool.execute(
        {"command": "sleep 2 && echo DONE_AUTO_BG", "timeout": 200},
        run_context=ctx,
    )
    assert r.success, r.error
    payload = json.loads(r.output.split("\n", 1)[-1] if r.output.strip().startswith("[") else r.output)
    # warning prefixes may prepend; find JSON object
    if "job_id" not in payload:
        start = r.output.find("{")
        payload = json.loads(r.output[start:])
    assert payload.get("auto_background_on_timeout") is True
    job_id = payload["job_id"]

    status_t = next(t for t in create_background_task_tools() if t.name == "task_status")
    out_t = next(t for t in create_background_task_tools() if t.name == "task_output")
    # Prefer the manager attached to ctx
    if getattr(ctx, "background_manager", None) is not None:
        from clawagents.tools.background_task import create_background_task_tools as cbt

        tools = cbt(ctx.background_manager)
        status_t = next(t for t in tools if t.name == "task_status")
        out_t = next(t for t in tools if t.name == "task_output")

    for _ in range(60):
        st = await status_t.execute({"job_id": job_id})
        data = json.loads(st.output)
        if not data.get("running"):
            break
        await asyncio.sleep(0.1)
    out = await out_t.execute({"job_id": job_id})
    assert "DONE_AUTO_BG" in out.output


@pytest.mark.asyncio
async def test_execute_profile_backend_auto_backgrounds_on_timeout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "0")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    class ProfileStub:
        kind = "profile:workspace:local"

        def __init__(self):
            self._inner = LocalBackend(root=str(tmp_path))
            self.cwd = str(tmp_path)
            self.profile_warnings = []
            self.wrapped = False

        def wrap_command(self, command, cwd=None):
            self.wrapped = True
            return command

    class Ctx:
        pass

    backend = ProfileStub()
    result = await ExecTool(backend).execute(
        {"command": "sleep 1", "timeout": 100},
        run_context=Ctx(),
    )

    assert result.success, result.error
    payload = json.loads(result.output[result.output.find("{") :])
    assert payload["auto_background_on_timeout"] is True
    assert backend.wrapped is True


def test_edit_file_unicode_hint():
    from clawagents.tools.filesystem import _nearest_edit_hint

    # Curly vs straight apostrophe
    content = "it\u2019s fine\n"
    target = "it's fine\n"
    hint = _nearest_edit_hint(content, target)
    assert "NFKC" in hint or "Nearest similar" in hint


def test_shell_session_sticky_env_marker_roundtrip(tmp_path):
    from clawagents.tools.shell_session import ENV_MARKER, ShellSession

    sess = ShellSession(cwd=str(tmp_path))
    wrapped = sess.wrap("export CLAW_TEST_FOO=bar", sticky_env=True)
    assert "python" in wrapped
    assert str(tmp_path) in wrapped

    # Simulate dump of a changed safe var (trailers at end only)
    fake = (
        f"ok\n{PWD_MARKER}{tmp_path.resolve()}\n"
        f'{ENV_MARKER}{{"CLAW_TEST_FOO":"bar","PATH":"/usr/bin"}}\n'
    )
    clean = sess.consume_stdout(fake, sticky_env=True)
    assert "ok" in clean
    assert ENV_MARKER not in clean
    assert PWD_MARKER not in clean
    assert sess.env.get("CLAW_TEST_FOO") == "bar"
    assert "PATH" not in sess.env  # common env never sticks
    # Secrets / deny substr must not stick even if dumped
    sess2 = ShellSession(cwd=str(tmp_path))
    sess2.consume_stdout(
        f'{PWD_MARKER}{tmp_path.resolve()}\n'
        f'{ENV_MARKER}{{"AWS_SECRET_ACCESS_KEY":"x","SSH_AUTH_SOCK":"/tmp/s"}}\n',
        sticky_env=True,
    )
    assert "AWS_SECRET_ACCESS_KEY" not in sess2.env
    assert "SSH_AUTH_SOCK" not in sess2.env


def test_shell_session_ignores_mid_output_marker_poison(tmp_path):
    from clawagents.tools.shell_session import ENV_MARKER, ShellSession

    sess = ShellSession(cwd=str(tmp_path))
    # Mid-output fake markers must not rewrite cwd/env; only trailers do.
    poisoned = (
        f"hello\n{PWD_MARKER}/nonexistent_poison_cwd\n"
        f'{ENV_MARKER}{{"CLAW_POISON":"1"}}\n'
        f"still printing\n"
        f"{PWD_MARKER}{tmp_path.resolve()}\n"
        f'{ENV_MARKER}{{"CLAW_REAL":"2"}}\n'
    )
    clean = sess.consume_stdout(poisoned, sticky_env=True)
    assert "hello" in clean
    assert "still printing" in clean
    assert "CLAW_POISON" in clean  # mid fake ENV left in output
    assert sess.cwd == str(tmp_path.resolve())
    assert sess.env.get("CLAW_REAL") == "2"
    assert "CLAW_POISON" not in sess.env


@pytest.mark.asyncio
async def test_execute_sticky_env_persists(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_ENV", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "0")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_STREAMING", "0")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    class Ctx:
        pass

    ctx = Ctx()
    tool = ExecTool(LocalBackend(root=str(tmp_path)))
    r1 = await tool.execute(
        {"command": "export CLAW_STICKY_MARK=from_session"},
        run_context=ctx,
    )
    assert r1.success, r1.error
    sess = getattr(ctx, "shell_session")
    assert sess.env.get("CLAW_STICKY_MARK") == "from_session"

    r2 = await tool.execute(
        {"command": 'printf "%s\\n" "$CLAW_STICKY_MARK"'},
        run_context=ctx,
    )
    assert r2.success, r2.error
    assert "from_session" in r2.output


@pytest.mark.asyncio
async def test_pty_start_uses_shell_session_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_PTY_SESSIONS", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.tools.pty_session import create_pty_tools
    from clawagents.tools.shell_session import ShellSession

    try:
        import pexpect  # noqa: F401
        import pyte  # noqa: F401
    except ImportError:
        pytest.skip("clawagents[pty] not installed")

    class Ctx:
        shell_session = ShellSession(cwd=str(tmp_path))

    tools = create_pty_tools()
    start = next(t for t in tools if t.name == "pty_start")
    # Short-lived command so we don't leave a hung shell
    r = await start.execute(
        {"command": "pwd; exit 0"},
        run_context=Ctx(),
    )
    assert r.success, r.error
    assert str(tmp_path) in (r.output or "") or "session_id=" in (r.output or "")


@pytest.mark.asyncio
async def test_pty_retains_screen_when_command_exits_during_startup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAW_FEATURE_PTY_SESSIONS", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.tools.pty_session import create_pty_tools

    try:
        import pexpect  # noqa: F401
        import pyte  # noqa: F401
    except ImportError:
        pytest.skip("clawagents[pty] not installed")

    tools = create_pty_tools()
    start = next(t for t in tools if t.name == "pty_start")
    screen = next(t for t in tools if t.name == "pty_screen")
    started = await start.execute(
        {"command": "sh -c 'echo PTY_BOOT_FAILED; exit 7'", "cwd": str(tmp_path)}
    )

    assert started.success is False
    assert "PTY command exited during startup" in (started.error or "")
    assert "PTY_BOOT_FAILED" in (started.output or "")
    session_id = str(started.output).split("session_id=", 1)[1].splitlines()[0]

    retained = await screen.execute({"session_id": session_id})
    assert retained.success is True
    assert "alive=False" in retained.output
    assert "exit_code=7" in retained.output
    assert "PTY_BOOT_FAILED" in retained.output


@pytest.mark.asyncio
async def test_pty_completed_session_retention_without_optional_runtime(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAW_FEATURE_PTY_SESSIONS", "1")
    from clawagents.config import features as feat
    from clawagents.tools import pty_session as module

    feat._resolved = None  # type: ignore[attr-defined]

    class Child:
        exitstatus = 7
        signalstatus = None

        @staticmethod
        def isalive():
            return False

    class CompletedSession:
        def __init__(self, *args, **kwargs):
            self.session_id = "pty_completed_test"
            self._last_used = module.time.time()
            self._ended = True
            self._child = Child()

        @staticmethod
        def screen_text(include_empty=False):
            return "PTY_BOOT_FAILED"

        @staticmethod
        def cursor():
            return (1, 1)

        def status(self):
            return {
                "alive": False,
                "exit_code": 7,
                "signal": None,
            }

        def stop(self):
            self._ended = True

    monkeypatch.setattr(module, "_pty_available", lambda: True)
    monkeypatch.setattr(module, "PtySession", CompletedSession)
    module._SESSIONS.clear()
    tools = module.create_pty_tools()
    start = next(t for t in tools if t.name == "pty_start")
    screen = next(t for t in tools if t.name == "pty_screen")

    started = await start.execute({"command": "ignored", "cwd": str(tmp_path)})
    assert started.success is False
    assert "PTY_BOOT_FAILED" in started.output

    retained = await screen.execute({"session_id": "pty_completed_test"})
    assert retained.success is True
    assert "alive=False" in retained.output
    assert "exit_code=7" in retained.output
    module._SESSIONS.clear()
