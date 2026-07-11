"""Shadow-git turn checkpoints (Cline-inspired) — separate from project git."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _run(args: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=120
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def shadow_root(workspace: str | Path | None = None) -> Path:
    ws = Path(workspace or Path.cwd()).resolve()
    digest = hashlib.sha1(str(ws).encode()).hexdigest()[:12]
    root = Path.home() / ".clawagents" / "shadow-git" / digest
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_shadow_git(workspace: str | Path | None = None) -> Path:
    ws = Path(workspace or Path.cwd()).resolve()
    root = shadow_root(ws)
    git_dir = root / ".git"
    if not git_dir.exists():
        _run(["git", "init"], root)
        _run(["git", "config", "core.worktree", str(ws)], root)
        _run(["git", "config", "user.email", "clawagents@local"], root)
        _run(["git", "config", "user.name", "ClawAgents Checkpoint"], root)
        # ignore heavy dirs inside the worktree via info/exclude
        exclude = git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(
            "\n".join(
                [
                    ".git/",
                    "node_modules/",
                    ".venv/",
                    "venv/",
                    "dist/",
                    "build/",
                    "__pycache__/",
                    ".clawagents/shadow-git/",
                    "*.pyc",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        _run(["git", "add", "-A"], root)
        _run(["git", "commit", "--allow-empty", "-m", "checkpoint:init"], root)
    else:
        # refresh worktree pointer
        _run(["git", "config", "core.worktree", str(ws)], root)
    return root


def create_checkpoint(
    label: str = "",
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    root = ensure_shadow_git(workspace)
    _run(["git", "add", "-A"], root)
    msg = f"checkpoint:{int(time.time())}:{label or 'turn'}"
    code, out, err = _run(["git", "commit", "--allow-empty", "-m", msg], root)
    if code != 0 and "nothing to commit" not in (err + out).lower():
        # still try to read HEAD
        pass
    c, sha, _ = _run(["git", "rev-parse", "HEAD"], root)
    return {
        "ok": c == 0,
        "sha": sha if c == 0 else "",
        "label": label,
        "message": msg,
        "shadow_root": str(root),
    }


def list_checkpoints(
    *,
    workspace: str | Path | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    root = ensure_shadow_git(workspace)
    code, out, err = _run(
        ["git", "log", f"-{max(1, limit)}", "--format=%H%x09%s%x09%ct"], root
    )
    if code != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            rows.append({"sha": parts[0], "message": parts[1], "ts": int(parts[2])})
    return rows


def restore_checkpoint(
    sha: str,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Hard-reset shadow git HEAD → restores workspace files via core.worktree."""
    root = ensure_shadow_git(workspace)
    sha = (sha or "").strip()
    if not sha:
        return {"ok": False, "error": "sha required"}
    code, out, err = _run(["git", "reset", "--hard", sha], root)
    if code != 0:
        return {"ok": False, "error": err or out}
    return {"ok": True, "sha": sha, "output": out}


def checkpoint_diff(
    lhs: str,
    rhs: str | None = None,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    root = ensure_shadow_git(workspace)
    args = ["git", "diff", "--name-status", lhs]
    if rhs:
        args.append(rhs)
    code, out, err = _run(args, root)
    if code != 0:
        return {"ok": False, "error": err or out, "files": []}
    files = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            files.append({"status": parts[0], "path": parts[1]})
    return {"ok": True, "files": files}
