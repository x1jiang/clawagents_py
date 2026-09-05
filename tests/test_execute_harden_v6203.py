"""Execute/sandbox hardening regressions (v6.20.3)."""

from __future__ import annotations

import asyncio
import json

import pytest


def test_child_env_strips_sensitive_and_secret_names(monkeypatch):
    from clawagents.tools.exec import _child_env

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ADVISOR_MODEL", "should-strip")
    monkeypatch.setenv("CLAW_SAFE_MARKER", "ok")
    env = _child_env()
    assert "OPENAI_API_KEY" not in env
    assert "ADVISOR_MODEL" not in env
    assert env.get("CLAW_SAFE_MARKER") == "ok"
    assert env.get("PAGER") == "cat"


@pytest.mark.asyncio
async def test_background_uses_sanitized_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.background_task import create_background_task_tools
    from clawagents.tools.exec import ExecTool

    class Ctx:
        pass

    ctx = Ctx()
    tool = ExecTool(LocalBackend(root=str(tmp_path)))
    r = await tool.execute(
        {
            "command": 'python3 -c "import os; print(os.environ.get(\'OPENAI_API_KEY\',\'MISSING\'))"',
            "is_background": True,
            "description": "env scrub check",
        },
        run_context=ctx,
    )
    assert r.success, r.error
    start = r.output.find("{")
    payload = json.loads(r.output[start:])
    job_id = payload["job_id"]

    tools = create_background_task_tools(ctx.background_manager)
    status_t = next(t for t in tools if t.name == "task_status")
    out_t = next(t for t in tools if t.name == "task_output")
    for _ in range(50):
        st = await status_t.execute({"job_id": job_id})
        data = json.loads(st.output)
        if not data.get("running"):
            break
        await asyncio.sleep(0.05)
    out = await out_t.execute({"job_id": job_id})
    assert "sk-should-not-leak" not in out.output
    assert "MISSING" in out.output


@pytest.mark.asyncio
async def test_profile_soft_fallback_warning_surfaces(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "0")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "0")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from unittest.mock import patch

    from clawagents.sandbox.local import LocalBackend
    from clawagents.sandbox.profiles import OSSandboxProfile, ProfileBackend
    from clawagents.tools.exec import ExecTool

    sb = ProfileBackend(
        LocalBackend(root=str(tmp_path)),
        OSSandboxProfile(
            name="workspace",
            backend="seatbelt",
            network=False,
            require_binary=False,
        ),
    )
    tool = ExecTool(sb)
    with patch("clawagents.sandbox.profiles.shutil.which", return_value=None):
        r = await tool.execute({"command": "echo soft-fallback-ok", "timeout": 5000})
    assert r.success, r.error
    assert "sandbox_profile" in r.output
    assert "sandbox-exec unavailable" in r.output
    assert "soft-fallback-ok" in r.output


def test_bwrap_overlay_touches_missing_env(tmp_path, monkeypatch):
    """Missing .env must not crash bwrap wrap — placeholder is created."""
    from unittest.mock import patch

    from clawagents.sandbox.local import LocalBackend
    from clawagents.sandbox.profiles import OSSandboxProfile, ProfileBackend

    assert not (tmp_path / ".env").exists()
    sb = ProfileBackend(
        LocalBackend(root=str(tmp_path)),
        OSSandboxProfile(
            name="workspace",
            backend="bwrap",
            network=False,
            require_binary=False,
            secret_deny_paths=(".env",),
        ),
    )
    fake_bwrap = tmp_path / "fake-bwrap"
    fake_bwrap.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bwrap.chmod(0o755)
    with patch(
        "clawagents.sandbox.profiles.shutil.which",
        return_value=str(fake_bwrap),
    ):
        wrapped = sb.wrap_command("echo ok", cwd=str(tmp_path))
    assert "bwrap" in wrapped or str(fake_bwrap) in wrapped
    assert (tmp_path / ".env").is_file()
