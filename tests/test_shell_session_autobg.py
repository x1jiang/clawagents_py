"""Grok-inspired shell session cwd + auto-background-on-timeout."""

from __future__ import annotations

import asyncio
import json

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


def test_edit_file_unicode_hint():
    from clawagents.tools.filesystem import _nearest_edit_hint

    # Curly vs straight apostrophe
    content = "it\u2019s fine\n"
    target = "it's fine\n"
    hint = _nearest_edit_hint(content, target)
    assert "NFKC" in hint or "Nearest similar" in hint
