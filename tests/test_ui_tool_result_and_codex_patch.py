"""UI tool-result visibility + Codex apply_patch envelope regressions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from clawagents.graph.agent_loop import (
    UI_TOOL_RESULT_CHARS,
    _format_failed_exec_observation,
    _tool_observation,
    _ui_tool_result_text,
)
from clawagents.sandbox.local import LocalBackend
from clawagents.tools.apply_patch import ApplyPatchTool
from clawagents.tools.exec import _format_nonzero_command_output, _short_exit_error
from clawagents.tools.registry import ToolResult


def test_failed_exec_observation_puts_stderr_before_long_command():
    long_cmd = "cd /repo4/home/xjiang2/hca_data_split && " + ("x" * 200)
    payload = _format_nonzero_command_output(
        long_cmd,
        1,
        "stdout-line\n",
        "STATUS_LOGON_FAILURE: bad password\n",
        "",
    )
    obs = _format_failed_exec_observation(payload)
    assert obs is not None
    assert obs.index("STATUS_LOGON_FAILURE") < obs.index("command:")
    assert "Command exited with code 1" in obs
    # Even a 120-char window must reach stderr now.
    assert "STATUS_LOGON_FAILURE" in obs[:120] or "stderr" in obs[:80]


def test_tool_observation_uses_exec_reorder():
    long_cmd = "cd /somewhere && " + ("a" * 180)
    out = _format_nonzero_command_output(
        long_cmd, 1, "", "Authentication failed: STATUS_LOGON_FAILURE\n", ""
    )
    result = ToolResult(success=False, output=out, error=_short_exit_error(1))
    obs = _tool_observation(result)
    assert isinstance(obs, str)
    assert "STATUS_LOGON_FAILURE" in obs
    assert obs.index("STATUS_LOGON_FAILURE") < obs.index("command:")
    ui = _ui_tool_result_text(result, obs)
    assert "STATUS_LOGON_FAILURE" in ui
    assert len(ui) <= UI_TOOL_RESULT_CHARS + 20


def test_ui_tool_result_bounded():
    huge = "stderr-marker\n" + ("z" * 20_000)
    payload = json.dumps(
        {
            "command_executed": True,
            "success": False,
            "exit_code": 2,
            "command": "echo hi",
            "stdout": "",
            "stderr": huge,
            "interpretation": "fail",
        }
    )
    result = ToolResult(success=False, output=payload, error=_short_exit_error(2))
    obs = _tool_observation(result)
    ui = _ui_tool_result_text(result, obs)
    assert "stderr-marker" in ui
    assert len(ui) <= UI_TOOL_RESULT_CHARS
    assert ui.endswith("…[truncated]") or len(ui) < UI_TOOL_RESULT_CHARS


def test_short_exit_error_omits_command():
    err = _short_exit_error(1)
    assert err == "Command exited with code 1"
    assert "cd " not in err


def test_codex_begin_patch_applies(tmp_path: Path):
    f = tmp_path / "cli.py"
    f.write_text("import json\n", encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "*** Begin Patch\n*** Update File: cli.py\n"
        "@@\n import json\n+import sys\n*** End Patch"
    )
    result = asyncio.run(tool.execute({"path": "cli.py", "patch": patch}))
    assert result.success, result.error
    assert "import sys" in f.read_text(encoding="utf-8")


def test_codex_path_mismatch_rejected(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "*** Begin Patch\n*** Update File: b.py\n"
        "@@\n y = 2\n+y = 3\n*** End Patch"
    )
    result = asyncio.run(tool.execute({"path": "a.py", "patch": patch}))
    assert not result.success
    assert "does not match" in (result.error or "")


def test_codex_multi_file_rejected(tmp_path: Path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n@@\n a\n+b\n"
        "*** Update File: c.py\n@@\n c\n+d\n"
        "*** End Patch"
    )
    result = asyncio.run(tool.execute({"path": "a.py", "patch": patch}))
    assert not result.success
    assert "multi-file" in (result.error or "")
