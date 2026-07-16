"""Git worktree isolation for parallel subagents."""

from __future__ import annotations

import re
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
    return root


def create_worktree(
    *,
    workspace: str | Path | None = None,
    name: str | None = None,
    branch: str | None = None,
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
    return {"ok": True, "path": str(path), "branch": br, "reused": False}


def ensure_task_worktree(
    *,
    workspace: str | Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Create (or reuse) a worktree for a task subagent.

    Unique slug per call when ``name`` is omitted so parallel tasks don't collide.
    """
    slug = name or f"task-{uuid.uuid4().hex[:8]}"
    return create_worktree(workspace=workspace, name=slug)


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
