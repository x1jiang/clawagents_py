"""snapshot_diff tool + code-kind crush floor."""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from clawagents.tools.context_tools import SnapshotDiffTool


def test_code_read_not_crushed_under_4k(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

    # Numbered file view ~2.5K — previously crushed aggressively; must stay intact.
    lines = [f"{str(i).rjust(4)}: const x{i} = {i};" for i in range(120)]
    text = "File: src/a.js (120 lines)\n" + "\n".join(lines)
    assert 2000 < len(text) < 4000
    prompt, aid = prepare_tool_output_for_context(
        tool_name="read_file",
        tool_use_id="r1",
        output=text,
        workspace=str(tmp_path),
    )
    assert "[Crushed tool output" not in prompt
    assert prompt == text


def test_snapshot_diff_shows_change(tmp_path: Path):
    ws = tmp_path
    target = ws / "deploy.sh"
    target.write_text("echo one\n", encoding="utf-8")
    snap = ws / ".clawagents" / "snapshots" / str(int(time.time()) - 10)
    snap.mkdir(parents=True)
    shutil.copy2(target, snap / "deploy.sh")
    target.write_text("echo two\n", encoding="utf-8")

    tool = SnapshotDiffTool(str(ws))
    result = asyncio.run(tool.execute({"path": "deploy.sh"}))
    assert result.success, result.error
    assert "echo one" in (result.output or "") or "-echo one" in (result.output or "")
    assert "echo two" in (result.output or "") or "+echo two" in (result.output or "")
