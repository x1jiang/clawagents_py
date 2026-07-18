"""Hunk file watcher + session rewind deltas.

Grok Build parity (xai-hunk-tracker / xai-chat-state): mtime-based external
edit attribution and per-prompt rewind snapshots for files + conversation.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class FileBaseline:
    path: str
    mtime: float
    content: str
    agent_touched: bool = False
    prompt_index: int | None = None


@dataclass
class HunkTurnDelta:
    prompt_index: int
    file_states: dict[str, str] = field(default_factory=dict)  # relpath → content
    note: str = ""


@dataclass
class RewindSnapshot:
    prompt_index: int
    user_text: str
    file_states: dict[str, str] = field(default_factory=dict)
    message_count: int | None = None
    conversation_marker: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


def is_secret_or_ignored_path(rel: str) -> bool:
    """True for VCS/build noise or secret-like paths that must never be snapshotted."""
    from clawagents.security.secret_paths import (
        is_secret_or_ignored_path as _central,
    )

    return _central(rel)


class HunkWatcher:
    """gitignore-light mtime watcher that refreshes attributed hunks."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        interval_s: float = 1.0,
        watch_budget: int = 500,
    ):
        self.workspace = Path(workspace).resolve()
        self.interval_s = interval_s
        self.watch_budget = watch_budget
        self._files: dict[str, FileBaseline] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prompt_index = 0
        self._deltas: list[HunkTurnDelta] = []
        self._store_dir = self.workspace / ".clawagents" / "rewind"
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def record_agent_write(self, rel_path: str, content: str, prompt_index: int | None = None) -> None:
        # Do NOT use str.lstrip("./") — that strips every leading '.' and turns
        # ".env" into "env", defeating secret filters.
        rel = rel_path.replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        # Never baseline/snapshot secrets into world-readable rewind/hunk JSON.
        if self._should_ignore(rel):
            return
        abs_path = self.workspace / rel
        try:
            mtime = abs_path.stat().st_mtime if abs_path.exists() else time.time()
        except OSError:
            mtime = time.time()
        idx = self._prompt_index if prompt_index is None else prompt_index
        self._files[rel] = FileBaseline(
            path=rel,
            mtime=mtime,
            content=content,
            agent_touched=True,
            prompt_index=idx,
        )
        # Seed hunk baseline
        try:
            from clawagents.memory.attributed_hunks import (
                agent_edit_attribution,
                refresh_file_hunks,
            )

            refresh_file_hunks(
                rel,
                workspace=self.workspace,
                turn_index=idx,
                tool="agent_write",
                source="agent",
                attribution=agent_edit_attribution(idx),
                seed_baseline_if_missing=True,
            )
        except Exception:
            pass

    def snapshot_turn(
        self,
        prompt_index: int,
        user_text: str = "",
        *,
        message_count: int | None = None,
        conversation_marker: list[dict[str, str]] | None = None,
    ) -> RewindSnapshot:
        states = {
            p: b.content
            for p, b in self._files.items()
            if not self._should_ignore(p)
        }
        snap = RewindSnapshot(
            prompt_index=prompt_index,
            user_text=user_text,
            file_states=dict(states),
            message_count=message_count,
            conversation_marker=list(conversation_marker or []),
        )
        delta = HunkTurnDelta(prompt_index=prompt_index, file_states=dict(states))
        self._deltas.append(delta)
        path = self._store_dir / f"prompt_{prompt_index:04d}.json"
        path.write_text(json.dumps(asdict(snap), indent=2) + "\n", encoding="utf-8")
        self._prompt_index = max(self._prompt_index, prompt_index)
        return snap

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = []
        for p in sorted(self._store_dir.glob("prompt_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "prompt_index": data.get("prompt_index"),
                        "user_text": (data.get("user_text") or "")[:200],
                        "message_count": data.get("message_count"),
                        "files": len(data.get("file_states") or {}),
                        "path": str(p),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
        return rows

    def rewind_to_prompt(self, prompt_index: int) -> dict[str, Any]:
        """Restore file contents to snapshot at prompt_index (last write wins)."""
        from clawagents.config.features import is_enabled

        if not is_enabled("session_rewind"):
            return {"ok": False, "error": "session_rewind feature disabled"}

        # Compose deltas ascending up to N
        composed: dict[str, str] = {}
        for d in sorted(self._deltas, key=lambda x: x.prompt_index):
            if d.prompt_index > prompt_index:
                break
            composed.update(d.file_states)

        # Prefer on-disk snapshot if present
        snap_path = self._store_dir / f"prompt_{prompt_index:04d}.json"
        snap_user_text = ""
        snap_message_count: int | None = None
        snap_marker: list[dict[str, str]] = []
        if snap_path.is_file():
            try:
                data = json.loads(snap_path.read_text(encoding="utf-8"))
                composed = dict(data.get("file_states") or composed)
                snap_user_text = str(data.get("user_text") or "")
                raw_mc = data.get("message_count")
                snap_message_count = int(raw_mc) if raw_mc is not None else None
                raw_marker = data.get("conversation_marker")
                if isinstance(raw_marker, list):
                    snap_marker = [
                        {"role": str(m.get("role") or ""), "preview": str(m.get("preview") or "")[:120]}
                        for m in raw_marker
                        if isinstance(m, dict)
                    ]
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        restored = []
        for rel, content in composed.items():
            if ".." in Path(rel).parts or Path(rel).is_absolute():
                continue
            abs_path = (self.workspace / rel).resolve()
            try:
                abs_path.relative_to(self.workspace)
            except ValueError:
                continue
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            restored.append(rel)
            self._files[rel] = FileBaseline(
                path=rel,
                mtime=time.time(),
                content=content,
                agent_touched=True,
                prompt_index=prompt_index,
            )
        # Drop later deltas
        self._deltas = [d for d in self._deltas if d.prompt_index <= prompt_index]
        self._prompt_index = prompt_index
        return {
            "ok": True,
            "prompt_index": prompt_index,
            "restored": restored,
            "user_text": snap_user_text,
            "truncate_to_user_text": snap_user_text,
            "message_count": snap_message_count,
            "conversation_marker": snap_marker,
        }

    def _should_ignore(self, rel: str) -> bool:
        return is_secret_or_ignored_path(rel)

    def poll_once(self) -> list[str]:
        """Scan watched files for external mtime changes; refresh hunks."""
        from clawagents.config.features import is_enabled

        if not is_enabled("hunk_watcher"):
            return []
        changed: list[str] = []
        # Expand watch set: known files + limited walk of workspace text files
        candidates = list(self._files.keys())
        if len(candidates) < self.watch_budget:
            for root, dirs, files in os.walk(self.workspace):
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", ".clawagents", "node_modules", ".venv", "venv", "__pycache__"}
                ]
                for name in files:
                    rel = str((Path(root) / name).relative_to(self.workspace))
                    if self._should_ignore(rel):
                        continue
                    if rel not in self._files and len(candidates) < self.watch_budget:
                        candidates.append(rel)
                if len(candidates) >= self.watch_budget:
                    break

        for rel in candidates:
            abs_path = self.workspace / rel
            if not abs_path.is_file():
                continue
            try:
                st = abs_path.stat()
                text = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            prev = self._files.get(rel)
            if prev is None:
                self._files[rel] = FileBaseline(path=rel, mtime=st.st_mtime, content=text)
                continue
            if st.st_mtime <= prev.mtime and text == prev.content:
                continue
            # External change
            on_agent_file = bool(prev.agent_touched)
            source = "external_on_agent" if on_agent_file else "external"
            tool = "external_on_agent" if on_agent_file else "external"
            try:
                from clawagents.memory.attributed_hunks import (
                    external_edit_attribution,
                    refresh_file_hunks,
                )

                refresh_file_hunks(
                    rel,
                    workspace=self.workspace,
                    turn_index=prev.prompt_index,
                    tool=tool,
                    source=source,  # type: ignore[arg-type]
                    attribution=external_edit_attribution(on_agent_file=on_agent_file),
                    seed_baseline_if_missing=False,
                )
            except Exception:
                pass
            self._files[rel] = FileBaseline(
                path=rel,
                mtime=st.st_mtime,
                content=text,
                agent_touched=prev.agent_touched,
                prompt_index=prev.prompt_index,
            )
            changed.append(rel)
        return changed

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.poll_once()
                except Exception:
                    pass
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True, name="hunk-watcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_WATCHERS: dict[str, HunkWatcher] = {}


def get_watcher(workspace: str | Path | None = None) -> HunkWatcher:
    ws = str(Path(workspace or os.getcwd()).resolve())
    if ws not in _WATCHERS:
        _WATCHERS[ws] = HunkWatcher(ws)
    return _WATCHERS[ws]


def create_rewind_tools():
    from clawagents.tools.registry import ToolResult

    class RewindListTool:
        name = "rewind_list"
        description = "List prompt rewind snapshots (files + conversation markers)."
        parameters: dict = {}

        async def execute(self, args: dict) -> ToolResult:
            from clawagents.config.features import is_enabled

            if not is_enabled("session_rewind"):
                return ToolResult(success=False, output="", error="session_rewind disabled")
            w = get_watcher()
            rows = w.list_snapshots()
            return ToolResult(success=True, output=json.dumps(rows, indent=2))

    class RewindToTool:
        name = "rewind_to"
        description = (
            "Rewind workspace files to prompt N snapshot (Grok-style rewind). "
            "Conversation truncation is host-managed."
        )
        parameters = {
            "prompt_index": {
                "type": "number",
                "required": True,
                "description": "Prompt index to restore",
            }
        }

        async def execute(self, args: dict) -> ToolResult:
            from clawagents.config.features import is_enabled

            if not is_enabled("session_rewind"):
                return ToolResult(success=False, output="", error="session_rewind disabled")
            idx = int(args.get("prompt_index") or -1)
            if idx < 0:
                return ToolResult(success=False, output="", error="prompt_index required")
            result = get_watcher().rewind_to_prompt(idx)
            return ToolResult(
                success=bool(result.get("ok")),
                output=json.dumps(result, indent=2),
                error=result.get("error"),
            )

    return [RewindListTool(), RewindToTool()]


__all__ = [
    "FileBaseline",
    "HunkTurnDelta",
    "RewindSnapshot",
    "is_secret_or_ignored_path",
    "HunkWatcher",
    "get_watcher",
    "create_rewind_tools",
]
