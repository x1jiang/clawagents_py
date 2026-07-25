"""Subagent worktrees inherit a runnable tree.

``git worktree add`` checks out HEAD, so without replication a subagent gets
none of the parent's uncommitted work and no ``node_modules`` / ``.venv`` /
``.env`` — it often cannot build or test the change it was asked to make.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from clawagents.tools.worktree import create_worktree, worktrees_root

_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, env={**os.environ, **_ENV}
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    ws = tmp_path / "repo"
    ws.mkdir()
    _git(["init", "--quiet", "."], ws)
    (ws / ".gitignore").write_text("node_modules/\n.env\n")
    (ws / "tracked.txt").write_text("committed\n")
    _git(["add", "-A"], ws)
    _git(["commit", "-qm", "init"], ws)

    # Parent state a subagent needs but HEAD does not carry.
    (ws / "tracked.txt").write_text("committed\nUNCOMMITTED EDIT\n")
    (ws / "newfile.py").write_text("new work\n")
    (ws / "node_modules" / "pkg").mkdir(parents=True)
    (ws / "node_modules" / "pkg" / "index.js").write_text("dep\n")
    (ws / ".env").write_text("SECRET=x\n")
    return ws


def test_replication_copies_dirty_and_ignored_files(repo: Path):
    result = create_worktree(workspace=repo, name="sub", replicate=True)
    assert result["ok"], result.get("error")
    wt = Path(result["path"])

    # The uncommitted edit must win over the HEAD version — this is the case a
    # naive `git status --porcelain -z` parse silently drops, because the
    # common " M path" line begins with a space.
    assert (wt / "tracked.txt").read_text() == "committed\nUNCOMMITTED EDIT\n"
    assert (wt / "newfile.py").read_text() == "new work\n"
    assert (wt / "node_modules" / "pkg" / "index.js").read_text() == "dep\n"
    assert (wt / ".env").read_text() == "SECRET=x\n"
    assert result["replicated"]["copied"] == 4


def test_replication_is_opt_out(repo: Path):
    result = create_worktree(workspace=repo, name="plain", replicate=False)
    wt = Path(result["path"])
    assert "replicated" not in result
    assert (wt / "tracked.txt").read_text() == "committed\n"  # HEAD only
    assert not (wt / "newfile.py").exists()
    assert not (wt / ".env").exists()


def test_ignored_files_can_be_excluded(repo: Path):
    result = create_worktree(
        workspace=repo, name="nodeps", replicate=True, include_ignored=False
    )
    wt = Path(result["path"])
    assert (wt / "newfile.py").exists()  # untracked-but-not-ignored still comes
    assert not (wt / ".env").exists()
    assert not (wt / "node_modules").exists()


def test_worktrees_root_is_self_ignoring(repo: Path):
    root = worktrees_root(repo)
    assert (root / ".gitignore").read_text().strip().endswith("*")

    create_worktree(workspace=repo, name="sub", replicate=True)
    # A routine `git add .` in the parent must not stage replicated secrets.
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout
    assert "worktrees" not in tracked


def test_vcs_and_cache_dirs_are_never_replicated(repo: Path):
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    result = create_worktree(workspace=repo, name="sub", replicate=True)
    wt = Path(result["path"])
    assert not (wt / "__pycache__").exists()
    assert not (wt / ".clawagents").exists()
