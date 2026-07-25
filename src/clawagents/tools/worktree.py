"""Git worktree isolation for parallel subagents."""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


def _run(args: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def worktrees_root(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or Path.cwd()) / ".clawagents" / "worktrees"
    root.mkdir(parents=True, exist_ok=True)
    # Self-ignoring: replication copies uncommitted work, dependency trees and
    # dotenv files in here, and the parent repo sees the whole directory as
    # untracked. This sits *above* each worktree's own root, so it hides the
    # tree from the parent without affecting the worktree's own git status.
    marker = root / ".gitignore"
    if not marker.exists():
        try:
            marker.write_text(
                "# Subagent worktrees (may hold replicated .env / deps). Never commit.\n*\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    return root


# Never replicate these into a worktree: VCS metadata, our own worktree root
# (recursion), and caches that are large and trivially regenerated.
_REPLICATE_SKIP_DIRS: frozenset[str] = frozenset({
    ".git",
    ".clawagents",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
})
_REPLICATE_MAX_FILES = 2_000
_REPLICATE_MAX_BYTES = 256 * 1024 * 1024


def _skipped(rel: str) -> bool:
    return any(part in _REPLICATE_SKIP_DIRS for part in Path(rel).parts)


def _run_raw(args: list[str], cwd: Path) -> tuple[int, str]:
    """Like :func:`_run` but preserves stdout exactly.

    ``git status --porcelain -z`` encodes status in the first two columns, so
    the common ` M path` case *starts with a space*. ``_run`` strips it, which
    shifts the parse by one and silently mangles the first filename.
    """
    try:
        p = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=120
        )
        return p.returncode, p.stdout
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""


def _dirty_files(ws: Path) -> list[str]:
    """Tracked files with uncommitted modifications (not deletions)."""
    code, out = _run_raw(["git", "status", "--porcelain", "-z"], ws)
    if code != 0 or not out:
        return []
    files: list[str] = []
    # -z output: "XY path\0" (renames add a second NUL-separated original).
    for entry in out.split("\0"):
        if len(entry) < 4:
            continue
        status, rel = entry[:2], entry[3:]
        if "D" in status or not rel:
            continue
        files.append(rel)
    return files


def _ignored_files(ws: Path) -> list[str]:
    """Git-ignored files — node_modules/.venv/.env and friends."""
    code, out = _run_raw(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"], ws
    )
    if code != 0 or not out:
        return []
    return [rel for rel in out.split("\0") if rel]


def replicate_working_files(
    ws: Path,
    dest: Path,
    *,
    include_ignored: bool = True,
) -> dict[str, Any]:
    """Copy uncommitted + git-ignored files from *ws* into a fresh worktree.

    ``git worktree add`` checks out HEAD, so a subagent otherwise starts with
    none of the user's in-progress edits and no ``node_modules`` / ``.venv`` /
    ``.env`` — it frequently cannot build or test the very change it was asked
    to make. Copying both classes makes the isolated tree actually runnable.

    Bounded by file count and total bytes so a huge ignored tree cannot stall
    subagent startup; returns what was copied and whether a cap was hit.
    """
    copied = 0
    skipped_big = False
    total = 0
    rels: list[str] = _dirty_files(ws)
    if include_ignored:
        rels += _ignored_files(ws)

    seen: set[str] = set()
    for rel in rels:
        if rel in seen or _skipped(rel):
            continue
        seen.add(rel)
        src = ws / rel
        if not src.is_file() or src.is_symlink():
            continue
        try:
            size = src.stat().st_size
        except OSError:
            continue
        if copied >= _REPLICATE_MAX_FILES or total + size > _REPLICATE_MAX_BYTES:
            skipped_big = True
            break
        target = dest / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        except OSError:
            continue
        copied += 1
        total += size
    return {"copied": copied, "bytes": total, "truncated": skipped_big}


def create_worktree(
    *,
    workspace: str | Path | None = None,
    name: str | None = None,
    branch: str | None = None,
    replicate: bool = False,
    include_ignored: bool = True,
) -> dict[str, Any]:
    ws = Path(workspace or Path.cwd()).resolve()
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or uuid.uuid4().hex[:8])).strip("-")[:40]
    path = worktrees_root(ws) / slug
    if path.exists():
        return {"ok": True, "path": str(path), "branch": branch or f"claw/{slug}", "reused": True}
    br = branch or f"claw/{slug}"
    # create branch from HEAD if needed
    code, _, err = _run(["git", "worktree", "add", "-b", br, str(path)], ws)
    if code != 0:
        # branch may exist — try without -b
        code, _, err = _run(["git", "worktree", "add", str(path), br], ws)
        if code != 0:
            return {"ok": False, "error": err}
    result: dict[str, Any] = {"ok": True, "path": str(path), "branch": br, "reused": False}
    if replicate:
        result["replicated"] = replicate_working_files(
            ws, path, include_ignored=include_ignored
        )
    return result


def ensure_task_worktree(
    *,
    workspace: str | Path | None = None,
    name: str | None = None,
    replicate: bool = True,
) -> dict[str, Any]:
    """Create (or reuse) a worktree for a task subagent.

    Unique slug per call when ``name`` is omitted so parallel tasks don't collide.
    Replicates the parent's uncommitted and ignored files by default so the
    subagent inherits a tree it can actually build and test.
    """
    slug = name or f"task-{uuid.uuid4().hex[:8]}"
    return create_worktree(workspace=workspace, name=slug, replicate=replicate)


def remove_worktree(
    path: str,
    *,
    workspace: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    ws = Path(workspace or Path.cwd()).resolve()
    args = ["git", "worktree", "remove"]
    if force:
        args.append("--force")
    args.append(path)
    code, out, err = _run(args, ws)
    if code != 0:
        return {"ok": False, "error": err or out}
    return {"ok": True, "output": out}


def list_worktrees(workspace: str | Path | None = None) -> list[dict[str, str]]:
    ws = Path(workspace or Path.cwd()).resolve()
    code, out, _ = _run(["git", "worktree", "list", "--porcelain"], ws)
    if code != 0 or not out:
        return []
    rows: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            if cur:
                rows.append(cur)
            cur = {"path": line[len("worktree "):]}
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):]
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD "):]
    if cur:
        rows.append(cur)
    return rows
