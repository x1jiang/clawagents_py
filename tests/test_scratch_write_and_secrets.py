"""Regression: /tmp write_file parity, CRLF secret scrub, python3 verify."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def test_workspace_profile_allows_tmp_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_OS_SANDBOX_PROFILES", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.sandbox.profiles import resolve_sandbox

    sb = resolve_sandbox("workspace", workspace=str(tmp_path), default="workspace")
    scratch = Path(tempfile.gettempdir()) / f"clawagents-scratch-{os.getpid()}.txt"
    try:
        resolved = sb.safe_path(str(scratch))
        assert resolved == str(scratch.resolve()) or resolved.endswith(scratch.name)
        # /tmp spelling (macOS may realpath to /private/tmp)
        tmp_path_str = f"/tmp/clawagents-scratch-{os.getpid()}.txt"
        resolved_tmp = sb.safe_path(tmp_path_str)
        assert "tmp" in resolved_tmp.lower()
        with pytest.raises(ValueError):
            sb.safe_path("/etc/passwd")
    finally:
        scratch.unlink(missing_ok=True)


def test_write_file_to_tmp_via_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import asyncio

    monkeypatch.setenv("CLAW_FEATURE_OS_SANDBOX_PROFILES", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.sandbox.profiles import resolve_sandbox

    sb = resolve_sandbox("workspace", workspace=str(tmp_path), default="workspace")
    target = Path(tempfile.gettempdir()) / f"clawagents-wf-{os.getpid()}.txt"
    try:
        asyncio.run(sb.write_file(str(target), "scratch-ok\n"))
        assert target.read_text(encoding="utf-8") == "scratch-ok\n"
    finally:
        target.unlink(missing_ok=True)


def test_read_only_profile_still_blocks_tmp_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_OS_SANDBOX_PROFILES", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.sandbox.profiles import resolve_sandbox

    sb = resolve_sandbox("read-only", workspace=str(tmp_path), default="read-only")
    with pytest.raises(ValueError):
        sb.safe_path(f"/tmp/clawagents-ro-{os.getpid()}.txt")


def test_strip_cr_from_secret_env(monkeypatch: pytest.MonkeyPatch):
    from clawagents.config.config import _strip_cr_from_secret_env

    monkeypatch.setenv("SMB_PASSWORD", "secret\r")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test\r")
    monkeypatch.setenv("NORMAL_PATH", "/tmp/foo\r")  # not a secret name — leave alone
    _strip_cr_from_secret_env()
    assert os.environ["SMB_PASSWORD"] == "secret"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    assert os.environ["NORMAL_PATH"] == "/tmp/foo\r"


def test_auto_verify_prefers_python3(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\n[tool.pytest.ini_options]\n',
        encoding="utf-8",
    )
    from clawagents.tools.auto_verify import detect_verify_commands

    cmds = detect_verify_commands(tmp_path)
    assert cmds, "expected pytest verify command"
    assert any("python3" in c[0] or Path(c[0]).name.startswith("python") for c in cmds)
    assert all(c[0] != "python" for c in cmds), "bare 'python' must not be used"
