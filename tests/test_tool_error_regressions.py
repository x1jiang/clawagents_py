"""Regressions from Goal/Act review sessions: skill tools, grep globs, apply_patch."""

from __future__ import annotations

import asyncio
from pathlib import Path

from clawagents.run_context import RunContext
from clawagents.sandbox.local import LocalBackend
from clawagents.tools.apply_patch import ApplyPatchTool
from clawagents.tools.filesystem import GrepTool
from clawagents.tools.skills import SkillStore, create_skill_tools, parse_skill_file


def test_claude_allowed_tools_yaml_block_list_parses():
    content = """---
name: review
description: review skill
allowed-tools:
  - Bash
  - Read
  - Edit
  - Grep
  - Agent
  - AskUserQuestion
---
body
"""
    skill = parse_skill_file(content, "/tmp/review/SKILL.md")
    assert skill.allowed_tools == [
        "Bash",
        "Read",
        "Edit",
        "Grep",
        "Agent",
        "AskUserQuestion",
    ]


def test_use_skill_maps_claude_tool_aliases(tmp_path: Path):
    skill_dir = tmp_path / "review"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: review
description: review
allowed-tools:
  - Bash
  - Read
  - Edit
  - Grep
  - Agent
  - AskUserQuestion
  - WebSearch
---
# Review
Do the review.
""",
        encoding="utf-8",
    )
    store = SkillStore()
    store.add_directory(tmp_path)
    asyncio.run(store.load_all())

    available = {
        "execute",
        "read_file",
        "edit_file",
        "grep",
        "task",
        "ask_user",
        "web_search",
        "use_skill",
        "list_skills",
    }
    use = [
        t
        for t in create_skill_tools(store, available_tool_names=lambda: available)
        if t.name == "use_skill"
    ][0]
    result = asyncio.run(use.execute({"name": "review"}, RunContext()))
    assert result.success, result.error
    assert "execute" in (result.output or "")
    assert "read_file" in (result.output or "")
    assert "unknown allowed-tools" not in (result.error or "")


def test_grep_accepts_glob_as_path(tmp_path: Path):
    (tmp_path / "a.js").write_text("TODO fix me\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("TODO py\n", encoding="utf-8")
    tool = GrepTool(LocalBackend(str(tmp_path)))
    result = asyncio.run(
        tool.execute({"path": "*.js", "pattern": "TODO", "recursive": False})
    )
    assert result.success, result.error
    assert "a.js" in (result.output or "")
    assert not (result.error or "")


def test_apply_patch_accepts_single_file_begin_patch_envelope(tmp_path: Path):
    (tmp_path / "cli.py").write_text("import json\n", encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(str(tmp_path)))
    patch = (
        "*** Begin Patch\n*** Update File: cli.py\n"
        "@@\n import json\n+import sys\n*** End Patch"
    )
    result = asyncio.run(tool.execute({"path": "cli.py", "patch": patch}))
    assert result.success, result.error
    assert "import sys" in (tmp_path / "cli.py").read_text(encoding="utf-8")
