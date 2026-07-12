"""Shadow-git turn checkpoints (Cline-inspired) — separate from project git."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

RestoreMode = Literal["files", "conversation", "both"]

# A git object name we're willing to hand to `git reset/diff/rev-parse`.
# Commit SHAs (full/abbrev) plus HEAD-relative forms. Anything starting with
# '-' (which git would parse as an option) is rejected. Defence-in-depth:
# these calls are already list-based (no shell) and the sha comes from our
# own checkpoint index, so this only guards against a malformed/hostile id.
_REF_RE = re.compile(r"^[0-9a-fA-F]{4,64}(?:[~^]\d*)*$|^HEAD(?:[~^]\d*)*$")


def _valid_ref(ref: str) -> bool:
    return bool(_REF_RE.match((ref or "").strip()))


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


def _index_path(root: Path) -> Path:
    # Keep metadata inside .git so `git reset --hard` (worktree restore) cannot
    # delete the SHA→turn binding index.
    return root / ".git" / "clawagents-checkpoint-index.json"


def _load_index(root: Path) -> dict[str, Any]:
    path = _index_path(root)
    if not path.is_file():
        return {"checkpoints": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("checkpoints"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"checkpoints": {}}


def _save_index(root: Path, data: dict[str, Any]) -> None:
    path = _index_path(root)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_shadow_git(workspace: str | Path | None = None) -> Path:
    ws = Path(workspace or Path.cwd()).resolve()
    root = shadow_root(ws)
    git_dir = root / ".git"
    if not git_dir.exists():
        _run(["git", "init"], root)
        _run(["git", "config", "core.worktree", str(ws)], root)
        _run(["git", "config", "user.email", "clawagents@local"], root)
        _run(["git", "config", "user.name", "ClawAgents Checkpoint"], root)
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
        _run(["git", "config", "core.worktree", str(ws)], root)
    return root


def create_checkpoint(
    label: str = "",
    *,
    workspace: str | Path | None = None,
    tool: str | None = None,
    turn_index: int | None = None,
    message_count: int | None = None,
    session_path: str | Path | None = None,
    chat_ui_path: str | Path | None = None,
    phase: str = "post",
) -> dict[str, Any]:
    """Commit the whole workspace into the shadow-git store.

    Metadata (tool, turn_index, message_count, session paths) is persisted in
    ``checkpoint-index.json`` beside the shadow repo for conversation restore.
    """
    root = ensure_shadow_git(workspace)
    _run(["git", "add", "-A"], root)
    ts = int(time.time())
    msg = f"checkpoint:{ts}:{label or phase or 'turn'}"
    code, out, err = _run(["git", "commit", "--allow-empty", "-m", msg], root)
    if code != 0 and "nothing to commit" not in (err + out).lower():
        pass
    c, sha, _ = _run(["git", "rev-parse", "HEAD"], root)
    meta: dict[str, Any] = {
        "ok": c == 0,
        "sha": sha if c == 0 else "",
        "label": label,
        "tool": tool or label or "",
        "turn_index": turn_index,
        "message_count": message_count,
        "session_path": str(session_path) if session_path else None,
        "chat_ui_path": str(chat_ui_path) if chat_ui_path else None,
        "phase": phase,
        "ts": ts,
        "message": msg,
        "shadow_root": str(root),
    }
    if meta["ok"] and meta["sha"]:
        index = _load_index(root)
        index["checkpoints"][meta["sha"]] = {
            k: v for k, v in meta.items() if k not in {"ok", "shadow_root"}
        }
        # Keep index bounded
        cps = index["checkpoints"]
        if len(cps) > 200:
            by_ts = sorted(cps.items(), key=lambda kv: int(kv[1].get("ts") or 0))
            for old_sha, _ in by_ts[: len(cps) - 200]:
                cps.pop(old_sha, None)
        _save_index(root, index)
    return meta


def bind_checkpoint_meta(
    sha: str,
    *,
    workspace: str | Path | None = None,
    tool: str | None = None,
    turn_index: int | None = None,
    message_count: int | None = None,
    session_path: str | Path | None = None,
    chat_ui_path: str | Path | None = None,
    label: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Update index metadata for an existing SHA without creating a new commit."""
    root = ensure_shadow_git(workspace)
    sha = (sha or "").strip()
    if not sha:
        return {"ok": False, "error": "sha required"}
    index = _load_index(root)
    # Resolve short SHA
    full = sha
    if sha not in index["checkpoints"]:
        for key in index["checkpoints"]:
            if key.startswith(sha):
                full = key
                break
    meta = dict(index["checkpoints"].get(full) or {"sha": full})
    meta["sha"] = full
    if tool is not None:
        meta["tool"] = tool
    if turn_index is not None:
        meta["turn_index"] = turn_index
    if message_count is not None:
        meta["message_count"] = message_count
    if session_path is not None:
        meta["session_path"] = str(session_path)
    if chat_ui_path is not None:
        meta["chat_ui_path"] = str(chat_ui_path)
    if label is not None:
        meta["label"] = label
    if phase is not None:
        meta["phase"] = phase
    meta.setdefault("ts", int(time.time()))
    index["checkpoints"][full] = meta
    _save_index(root, index)
    return {"ok": True, **meta}


def get_checkpoint_meta(
    sha: str,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any] | None:
    root = ensure_shadow_git(workspace)
    sha = (sha or "").strip()
    if not sha:
        return None
    index = _load_index(root)
    exact = index["checkpoints"].get(sha)
    if exact:
        return dict(exact)
    # Allow short SHA prefix
    for full, meta in index["checkpoints"].items():
        if full.startswith(sha):
            return dict(meta)
    return None


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
    index = _load_index(root)
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        sha, message, ts_s = parts[0], parts[1], parts[2]
        row: dict[str, Any] = {"sha": sha, "message": message, "ts": int(ts_s)}
        meta = index["checkpoints"].get(sha) or {}
        for key in ("label", "tool", "turn_index", "message_count", "phase"):
            if key in meta and meta[key] is not None:
                row[key] = meta[key]
        rows.append(row)
    return rows


def _truncate_jsonl(path: Path, keep: int) -> dict[str, Any]:
    if keep < 0:
        return {"ok": False, "error": "invalid message_count"}
    if not path.is_file():
        return {"ok": False, "error": f"session file missing: {path}"}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept = lines[:keep]
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {"ok": True, "path": str(path), "kept": len(kept), "dropped": len(lines) - len(kept)}


def restore_conversation(
    sha: str,
    *,
    workspace: str | Path | None = None,
    session_path: str | Path | None = None,
    chat_ui_path: str | Path | None = None,
) -> dict[str, Any]:
    """Truncate session / UI logs to the checkpoint's bound message_count."""
    meta = get_checkpoint_meta(sha, workspace=workspace) or {}
    count = meta.get("message_count")
    if count is None:
        return {
            "ok": False,
            "error": "checkpoint has no message_count binding; cannot restore conversation",
        }
    try:
        keep = int(count)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid message_count on checkpoint"}

    sess = Path(session_path) if session_path else (
        Path(meta["session_path"]) if meta.get("session_path") else None
    )
    ui = Path(chat_ui_path) if chat_ui_path else (
        Path(meta["chat_ui_path"]) if meta.get("chat_ui_path") else None
    )
    results: dict[str, Any] = {"ok": True, "sha": sha, "message_count": keep, "truncated": []}
    if sess is None and ui is None:
        return {
            "ok": False,
            "error": "no session_path or chat_ui_path bound to checkpoint",
        }
    if sess is not None:
        r = _truncate_jsonl(sess, keep)
        results["truncated"].append({"kind": "session", **r})
        if not r.get("ok"):
            results["ok"] = False
    if ui is not None:
        # UI log is event-oriented; truncate to message_count events when bound,
        # else leave untouched if file missing.
        r = _truncate_jsonl(ui, keep)
        results["truncated"].append({"kind": "chat_ui", **r})
        if not r.get("ok") and sess is None:
            results["ok"] = False
    return results


def restore_checkpoint(
    sha: str,
    *,
    workspace: str | Path | None = None,
    mode: RestoreMode = "files",
    session_path: str | Path | None = None,
    chat_ui_path: str | Path | None = None,
) -> dict[str, Any]:
    """Restore files and/or conversation from a shadow-git checkpoint SHA."""
    root = ensure_shadow_git(workspace)
    sha = (sha or "").strip()
    if not sha:
        return {"ok": False, "error": "sha required"}
    if not _valid_ref(sha):
        return {"ok": False, "error": f"invalid checkpoint ref: {sha!r}", "sha": sha}
    mode_norm: RestoreMode = mode if mode in ("files", "conversation", "both") else "files"
    out: dict[str, Any] = {"ok": True, "sha": sha, "mode": mode_norm}

    if mode_norm in ("files", "both"):
        code, stdout, err = _run(["git", "reset", "--hard", sha], root)
        if code != 0:
            return {"ok": False, "error": err or stdout, "mode": mode_norm, "sha": sha}
        out["files"] = {"ok": True, "output": stdout}

    if mode_norm in ("conversation", "both"):
        conv = restore_conversation(
            sha,
            workspace=workspace,
            session_path=session_path,
            chat_ui_path=chat_ui_path,
        )
        out["conversation"] = conv
        if not conv.get("ok"):
            out["ok"] = False
            out["error"] = conv.get("error")

    return out


def checkpoint_diff(
    lhs: str,
    rhs: str | None = None,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    root = ensure_shadow_git(workspace)
    if not _valid_ref(lhs) or (rhs is not None and not _valid_ref(rhs)):
        return {"ok": False, "error": "invalid checkpoint ref", "files": []}
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
