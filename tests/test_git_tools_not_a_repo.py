"""git_* tools should soft-fail clearly outside a repository."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from clawagents.tools.git_tools import GitCommitTool, GitDiffTool, GitStatusTool


def test_git_status_not_a_repo_is_soft_success(tmp_path: Path):
    tool = GitStatusTool(str(tmp_path))
    result = asyncio.run(tool.execute({}))
    assert result.success is True
    assert "Not a git repository" in (result.output or "")
    assert "Skip git_status" in (result.output or "")


def test_git_diff_not_a_repo_is_soft_success(tmp_path: Path):
    tool = GitDiffTool(str(tmp_path))
    result = asyncio.run(tool.execute({}))
    assert result.success is True
    assert "Not a git repository" in (result.output or "")


def test_git_commit_not_a_repo_is_hard_fail(tmp_path: Path):
    tool = GitCommitTool(str(tmp_path))
    result = asyncio.run(tool.execute({"message": "x", "all": True}))
    assert result.success is False
    assert "Not a git repository" in (result.error or "")


def test_git_status_inside_repo(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    tool = GitStatusTool(str(tmp_path))
    result = asyncio.run(tool.execute({}))
    assert result.success is True
    assert "Not a git repository" not in (result.output or "")
