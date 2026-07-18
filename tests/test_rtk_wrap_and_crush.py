"""Loop-side RTK wrap + aggressive tool crush (not hooks)."""

from __future__ import annotations

import pytest

from clawagents.tools.rtk_wrap import maybe_wrap_with_rtk, reset_rtk_cache


@pytest.fixture(autouse=True)
def _reset_features_and_rtk(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "1")
    monkeypatch.setenv("CLAW_FEATURE_AGGRESSIVE_TOOL_CRUSH", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]
    reset_rtk_cache()
    yield
    feat._resolved = None  # type: ignore[attr-defined]
    reset_rtk_cache()


def test_rtk_wrap_pytest(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("CLAW_RTK_BIN", str(fake))
    reset_rtk_cache()

    cmd, reason = maybe_wrap_with_rtk("pytest tests/ -q")
    assert reason == "rtk test"
    assert cmd == f"{fake} test pytest tests/ -q"


def test_rtk_wrap_git_status(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("CLAW_RTK_BIN", str(fake))
    reset_rtk_cache()

    cmd, reason = maybe_wrap_with_rtk("git status -sb")
    assert reason == "rtk git status"
    assert cmd == f"{fake} git status -sb"


def test_rtk_wrap_ls_and_skip_echo(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("CLAW_RTK_BIN", str(fake))
    reset_rtk_cache()

    cmd, reason = maybe_wrap_with_rtk("ls -la src")
    assert reason == "rtk ls"
    assert cmd.endswith("ls -la src")

    cmd2, reason2 = maybe_wrap_with_rtk("echo hello")
    assert reason2 is None
    assert cmd2 == "echo hello"


def test_rtk_wrap_skips_pipelines(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("CLAW_RTK_BIN", str(fake))
    reset_rtk_cache()

    cmd, reason = maybe_wrap_with_rtk("ls | wc -l")
    assert reason is None
    assert cmd == "ls | wc -l"


def test_rtk_wrap_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("CLAW_RTK_BIN", str(fake))
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]
    reset_rtk_cache()

    cmd, reason = maybe_wrap_with_rtk("pytest")
    assert reason is None
    assert cmd == "pytest"


def test_aggressive_crush_triggers_earlier(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

    # Sized to sit between aggressive (1200) and legacy (2000) thresholds.
    lines = [f"src/a.py:{i}: match hit" for i in range(200)]
    text = "\n".join(lines)
    # Trim/pad into the (1200, 2000) band while keeping many lines for search crush.
    text = text[:1600]
    assert 1200 < len(text) < 2000
    assert text.count("\n") > 60

    prompt, aid = prepare_tool_output_for_context(
        tool_name="grep",
        tool_use_id="t1",
        output=text,
        workspace=str(tmp_path),
    )
    assert aid is not None
    assert "[Crushed tool output" in prompt


def test_aggressive_crush_off_keeps_medium(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_AGGRESSIVE_TOOL_CRUSH", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]
    monkeypatch.chdir(tmp_path)
    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

    text = "y" * 1500  # under legacy 2000 threshold
    prompt, aid = prepare_tool_output_for_context(
        tool_name="execute",
        tool_use_id="t2",
        output=text,
        workspace=str(tmp_path),
    )
    assert prompt == text
    assert aid is None


@pytest.mark.asyncio
async def test_execute_applies_rtk_wrap(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake = tmp_path / "rtk"
    # Proxy that echoes argv so we can see the wrap happened.
    fake.write_text(
        "#!/bin/sh\n"
        "shift\n"  # drop subcommand "ls"
        "echo RTK_OK \"$@\"\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("CLAW_RTK_BIN", str(fake))
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]
    reset_rtk_cache()

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    tool = ExecTool(LocalBackend(root=str(tmp_path)))
    # Create a file so ls has something; wrap should invoke our fake rtk.
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    r = await tool.execute({"command": "ls a.txt"})
    assert r.success, r.error
    assert "[rtk_wrap: rtk ls]" in r.output
    assert "RTK_OK" in r.output
