"""Attributed hunk tracker — accept/reject reviewable diffs without git commit.

Baseline = last-accepted file content. Current = on-disk. Hunks are computed
as unified diffs with stable UUIDs attributed to a turn/tool.
"""

from __future__ import annotations

import difflib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


HunkSource = Literal["agent", "external", "external_on_agent", "unknown"]


def agent_edit_attribution(prompt_index: int | None) -> str:
    """Typed attribution for agent-originated edits."""
    if prompt_index is None:
        return "AgentEdit"
    return f"AgentEdit{prompt_index}"


def external_edit_attribution(*, on_agent_file: bool) -> str:
    """Typed attribution for watcher-detected external edits."""
    return "ExternalEditOnAgentFile" if on_agent_file else "External"


@dataclass
class AttributedHunk:
    id: str
    path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    body: str
    source: HunkSource = "agent"
    attribution: str = "agent"
    turn_index: int | None = None
    tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HunkStore:
    """In-memory + on-disk store keyed by workspace."""

    workspace: Path
    baselines: dict[str, str] = field(default_factory=dict)
    hunks: dict[str, AttributedHunk] = field(default_factory=dict)

    @property
    def store_dir(self) -> Path:
        d = self.workspace / ".clawagents" / "hunks"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def index_path(self) -> Path:
        return self.store_dir / "index.json"

    def save(self) -> None:
        payload = {
            "baselines": self.baselines,
            "hunks": {k: v.to_dict() for k, v in self.hunks.items()},
        }
        self.index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, workspace: str | Path | None = None) -> "HunkStore":
        ws = Path(workspace or Path.cwd()).resolve()
        store = cls(workspace=ws)
        path = store.index_path
        if not path.is_file():
            return store
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return store
        store.baselines = {
            str(k): str(v) for k, v in (data.get("baselines") or {}).items()
        }
        for hid, raw in (data.get("hunks") or {}).items():
            if not isinstance(raw, dict):
                continue
            src = raw.get("source") or "agent"
            store.hunks[str(hid)] = AttributedHunk(
                id=str(raw.get("id") or hid),
                path=str(raw.get("path") or ""),
                old_start=int(raw.get("old_start") or 0),
                old_count=int(raw.get("old_count") or 0),
                new_start=int(raw.get("new_start") or 0),
                new_count=int(raw.get("new_count") or 0),
                header=str(raw.get("header") or ""),
                body=str(raw.get("body") or ""),
                source=src,  # type: ignore[arg-type]
                attribution=str(raw.get("attribution") or src or "agent"),
                turn_index=raw.get("turn_index"),
                tool=raw.get("tool"),
            )
        return store


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _safe_workspace_file(store: HunkStore, path: str) -> tuple[str, Path]:
    """Resolve ``path`` under ``store.workspace``; reject absolute/``..`` escapes."""
    raw = (path or "").strip()
    if not raw:
        raise ValueError("empty path")
    if Path(raw).is_absolute():
        raise ValueError(f"absolute paths not allowed: {path}")
    if ".." in Path(raw).parts:
        raise ValueError(f"path escapes workspace: {path}")
    ws = store.workspace.resolve()
    abs_path = (ws / raw).resolve()
    try:
        rel = abs_path.relative_to(ws).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path}") from exc
    return rel, abs_path


def _parse_hunk_header(header: str) -> tuple[int, int, int, int]:
    # @@ -a,b +c,d @@
    try:
        parts = header.split()
        old = parts[1][1:]  # -a,b
        new = parts[2][1:]  # +c,d
        o_s, _, o_rest = old.partition(",")
        n_s, _, n_rest = new.partition(",")
        return (
            int(o_s),
            int(o_rest or "1"),
            int(n_s),
            int(n_rest or "1"),
        )
    except (IndexError, ValueError):
        return (0, 0, 0, 0)


def compute_hunks(
    path: str,
    baseline: str,
    current: str,
    *,
    turn_index: int | None = None,
    tool: str | None = None,
    source: HunkSource = "agent",
    attribution: str | None = None,
) -> list[AttributedHunk]:
    """Split a unified diff into attributed hunks."""
    old_lines = baseline.splitlines(keepends=True)
    new_lines = current.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    hunks: list[AttributedHunk] = []
    header = ""
    body: list[str] = []

    def flush() -> None:
        nonlocal header, body
        if not header:
            body = []
            return
        o_s, o_c, n_s, n_c = _parse_hunk_header(header)
        hid = uuid.uuid4().hex[:12]
        hunks.append(
            AttributedHunk(
                id=hid,
                path=path,
                old_start=o_s,
                old_count=o_c,
                new_start=n_s,
                new_count=n_c,
                header=header.strip(),
                body="".join(body),
                source=source,
                attribution=attribution or source,
                turn_index=turn_index,
                tool=tool,
            )
        )
        header = ""
        body = []

    for line in diff:
        if line.startswith("@@"):
            flush()
            header = line
            body = [line]
        elif header:
            body.append(line)
    flush()
    return hunks


def refresh_file_hunks(
    rel_path: str,
    *,
    workspace: str | Path | None = None,
    turn_index: int | None = None,
    tool: str | None = None,
    source: HunkSource = "agent",
    attribution: str | None = None,
    seed_baseline_if_missing: bool = True,
) -> list[AttributedHunk]:
    """Recompute hunks for one file relative to its baseline."""
    store = HunkStore.load(workspace)
    rel_path, abs_path = _safe_workspace_file(store, rel_path)
    # Never seed secret files into .clawagents/hunks/index.json.
    try:
        from clawagents.memory.hunk_watcher import is_secret_or_ignored_path

        if is_secret_or_ignored_path(rel_path):
            return []
    except Exception:
        pass
    current = _read_text(abs_path)
    if rel_path not in store.baselines:
        if seed_baseline_if_missing:
            # First sight: baseline = current → no pending hunks
            store.baselines[rel_path] = current
            # Drop prior hunks for this path
            store.hunks = {k: v for k, v in store.hunks.items() if v.path != rel_path}
            store.save()
            return []
        store.baselines[rel_path] = ""
    baseline = store.baselines[rel_path]
    new_hunks = compute_hunks(
        rel_path,
        baseline,
        current,
        turn_index=turn_index,
        tool=tool,
        source=source,
        attribution=attribution,
    )
    store.hunks = {k: v for k, v in store.hunks.items() if v.path != rel_path}
    for h in new_hunks:
        store.hunks[h.id] = h
    store.save()
    return new_hunks


def list_hunks(
    *,
    workspace: str | Path | None = None,
    path: str | None = None,
) -> list[AttributedHunk]:
    store = HunkStore.load(workspace)
    rows = list(store.hunks.values())
    if path:
        rows = [h for h in rows if h.path == path]
    rows.sort(key=lambda h: (h.path, h.old_start, h.id))
    return rows


def _apply_single_hunk_to_baseline(baseline: str, hunk: AttributedHunk) -> str:
    """Apply a unified hunk's '+' lines onto baseline at old_start.

    Accept = move baseline forward for this hunk only.
    """
    old_lines = baseline.splitlines(keepends=True)
    # Rebuild new block from hunk body
    new_block: list[str] = []
    for line in hunk.body.splitlines(keepends=True):
        if line.startswith("@@"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            new_block.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            continue
        elif line.startswith(" "):
            new_block.append(line[1:])
        elif line.startswith("\\"):
            continue
        else:
            # context without prefix (rare)
            new_block.append(line if line.endswith("\n") else line + "\n")

    start = max(0, hunk.old_start - 1)
    end = start + hunk.old_count
    return "".join(old_lines[:start] + new_block + old_lines[end:])


def _revert_hunk_on_current(current: str, baseline: str, hunk: AttributedHunk) -> str:
    """Reject = restore the old lines for this hunk into the current file."""
    # Equivalent: apply reverse of the hunk to current by splicing baseline segment
    cur_lines = current.splitlines(keepends=True)
    base_lines = baseline.splitlines(keepends=True)
    old_start = max(0, hunk.old_start - 1)
    old_end = old_start + hunk.old_count
    new_start = max(0, hunk.new_start - 1)
    new_end = new_start + hunk.new_count
    restored = base_lines[old_start:old_end]
    return "".join(cur_lines[:new_start] + restored + cur_lines[new_end:])


def accept_hunk(
    hunk_id: str,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Accept one hunk: advance baseline to include it; leave disk unchanged."""
    store = HunkStore.load(workspace)
    hunk = store.hunks.get(hunk_id)
    if hunk is None:
        return {"ok": False, "error": f"unknown hunk: {hunk_id}"}
    baseline = store.baselines.get(hunk.path, "")
    store.baselines[hunk.path] = _apply_single_hunk_to_baseline(baseline, hunk)
    del store.hunks[hunk_id]
    # Recompute remaining hunks against new baseline
    try:
        _, abs_path = _safe_workspace_file(store, hunk.path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    current = _read_text(abs_path)
    remaining = compute_hunks(hunk.path, store.baselines[hunk.path], current)
    store.hunks = {k: v for k, v in store.hunks.items() if v.path != hunk.path}
    for h in remaining:
        store.hunks[h.id] = h
    store.save()
    return {"ok": True, "path": hunk.path, "accepted": hunk_id, "remaining": len(remaining)}


def reject_hunk(
    hunk_id: str,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Reject one hunk: restore that region on disk toward baseline."""
    store = HunkStore.load(workspace)
    hunk = store.hunks.get(hunk_id)
    if hunk is None:
        return {"ok": False, "error": f"unknown hunk: {hunk_id}"}
    try:
        _, abs_path = _safe_workspace_file(store, hunk.path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    baseline = store.baselines.get(hunk.path, "")
    current = _read_text(abs_path)
    restored = _revert_hunk_on_current(current, baseline, hunk)
    _write_text(abs_path, restored)
    del store.hunks[hunk_id]
    remaining = compute_hunks(hunk.path, baseline, restored)
    store.hunks = {k: v for k, v in store.hunks.items() if v.path != hunk.path}
    for h in remaining:
        store.hunks[h.id] = h
    store.save()
    return {"ok": True, "path": hunk.path, "rejected": hunk_id, "remaining": len(remaining)}


def accept_all(
    path: str,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Accept all pending hunks for a path (baseline := current)."""
    store = HunkStore.load(workspace)
    try:
        path, abs_path = _safe_workspace_file(store, path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    store.baselines[path] = _read_text(abs_path)
    removed = [hid for hid, h in store.hunks.items() if h.path == path]
    for hid in removed:
        del store.hunks[hid]
    store.save()
    return {"ok": True, "path": path, "accepted": removed}


__all__ = [
    "AttributedHunk",
    "HunkStore",
    "HunkSource",
    "agent_edit_attribution",
    "external_edit_attribution",
    "compute_hunks",
    "refresh_file_hunks",
    "list_hunks",
    "accept_hunk",
    "reject_hunk",
    "accept_all",
]
