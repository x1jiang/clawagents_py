"""Tools: context ledger, core memory, repo map, checkpoints, plan handoff."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from clawagents.tools.registry import Tool, ToolResult


class RehydrateLedgerTool:
    name = "rehydrate_ledger"
    description = (
        "Recover exact bytes for a context-ledger commit via git show. "
        "Pass sha from the Context Ledger section; optional path for one file."
    )
    parameters = {
        "sha": {"type": "string", "description": "Commit SHA", "required": True},
        "path": {"type": "string", "description": "Optional repo-relative file path"},
        "max_chars": {"type": "integer", "description": "Cap returned chars"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.context_ledger import rehydrate_from_git

        try:
            max_chars = int(args.get("max_chars") or 80_000)
        except (TypeError, ValueError):
            max_chars = 80_000
        ok, text = rehydrate_from_git(
            str(args.get("sha") or ""),
            workspace=self._workspace,
            path=str(args.get("path") or "") or None,
            max_chars=max_chars,
        )
        if not ok:
            return ToolResult(success=False, output="", error=text)
        return ToolResult(success=True, output=text)


class RecordLedgerTool:
    name = "record_ledger"
    description = "Manually record HEAD (or sha) into the context ledger."
    parameters = {
        "sha": {"type": "string", "description": "Optional commit SHA (default HEAD)"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.context_ledger import record_commit_ledger

        entry = record_commit_ledger(
            workspace=self._workspace,
            sha=str(args.get("sha") or "") or None,
        )
        if entry is None:
            return ToolResult(success=True, output="No new ledger entry (missing git or already recorded)")
        return ToolResult(success=True, output=entry.to_markdown())


class CoreMemoryViewTool:
    name = "memory_view"
    description = "View the editable core memory blocks (.clawagents/core-memory.md)."
    parameters: dict[str, Any] = {}

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.core_memory import load_core_memory

        return ToolResult(success=True, output=load_core_memory(workspace=self._workspace))


class CoreMemoryReplaceTool:
    name = "memory_replace"
    description = "Replace text inside a core memory block (persona|human|project|…)."
    parameters = {
        "label": {"type": "string", "description": "Block label", "required": True},
        "old_str": {"type": "string", "description": "Exact text to find", "required": True},
        "new_str": {"type": "string", "description": "Replacement", "required": True},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.core_memory import core_memory_replace

        ok, msg = core_memory_replace(
            str(args.get("label") or "project"),
            str(args.get("old_str") or ""),
            str(args.get("new_str") or ""),
            workspace=self._workspace,
        )
        if not ok:
            return ToolResult(success=False, output="", error=msg)
        return ToolResult(success=True, output=msg)


class CoreMemoryAppendTool:
    name = "memory_append"
    description = "Append a line/paragraph to a core memory block."
    parameters = {
        "label": {"type": "string", "description": "Block label", "required": True},
        "content": {"type": "string", "description": "Text to append", "required": True},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.core_memory import core_memory_append

        ok, msg = core_memory_append(
            str(args.get("label") or "project"),
            str(args.get("content") or ""),
            workspace=self._workspace,
        )
        if not ok:
            return ToolResult(success=False, output="", error=msg)
        return ToolResult(success=True, output=msg)


class RepoMapTool:
    name = "repo_map"
    description = (
        "Build a ranked symbol map of the workspace (Aider-style). "
        "Use when orienting in a large codebase before deep reads."
    )
    parameters = {
        "max_chars": {"type": "integer", "description": "Token-ish char budget (default 4000)"},
        "mentioned": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Identifiers/paths to boost",
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.scope_graph import build_repo_map_incremental

        try:
            max_chars = int(args.get("max_chars") or 4_000)
        except (TypeError, ValueError):
            max_chars = 4_000
        mentioned = set()
        raw = args.get("mentioned") or []
        if isinstance(raw, list):
            mentioned = {str(x) for x in raw}
        text = build_repo_map_incremental(
            self._workspace, max_chars=max_chars, mentioned=mentioned
        )
        return ToolResult(success=True, output=text or "(no symbols found)")


class CheckpointCreateTool:
    name = "checkpoint_create"
    description = "Snapshot the workspace into the shadow-git checkpoint store."
    parameters = {
        "label": {"type": "string", "description": "Optional label"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.shadow_checkpoint import create_checkpoint

        info = create_checkpoint(str(args.get("label") or ""), workspace=self._workspace)
        if not info.get("ok"):
            return ToolResult(success=False, output="", error="checkpoint failed")
        return ToolResult(success=True, output=str(info))


class CheckpointRestoreTool:
    name = "checkpoint_restore"
    description = (
        "Restore from a shadow-git checkpoint SHA. "
        "mode=files|conversation|both (default files)."
    )
    parameters = {
        "sha": {"type": "string", "description": "Checkpoint SHA", "required": True},
        "mode": {
            "type": "string",
            "description": "files | conversation | both",
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.shadow_checkpoint import restore_checkpoint

        mode = str(args.get("mode") or "files").strip().lower()
        if mode not in ("files", "conversation", "both"):
            mode = "files"
        info = restore_checkpoint(
            str(args.get("sha") or ""),
            workspace=self._workspace,
            mode=mode,  # type: ignore[arg-type]
        )
        if not info.get("ok"):
            return ToolResult(success=False, output="", error=str(info.get("error")))
        return ToolResult(success=True, output=str(info))


class CheckpointListTool:
    name = "checkpoint_list"
    description = "List recent shadow-git checkpoints."
    parameters = {
        "limit": {"type": "integer", "description": "Max rows (default 20)"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.shadow_checkpoint import list_checkpoints
        import json

        try:
            limit = int(args.get("limit") or 20)
        except (TypeError, ValueError):
            limit = 20
        rows = list_checkpoints(workspace=self._workspace, limit=limit)
        return ToolResult(success=True, output=json.dumps(rows, indent=2))


class CheckpointDiffTool:
    name = "checkpoint_diff"
    description = "List file changes between two shadow-git checkpoints (name-status)."
    parameters = {
        "lhs": {"type": "string", "description": "Left SHA", "required": True},
        "rhs": {"type": "string", "description": "Right SHA (default: working tree)"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        from clawagents.memory.shadow_checkpoint import checkpoint_diff
        import json

        info = checkpoint_diff(
            str(args.get("lhs") or ""),
            str(args.get("rhs") or "") or None,
            workspace=self._workspace,
        )
        if not info.get("ok"):
            return ToolResult(success=False, output="", error=str(info.get("error")))
        return ToolResult(success=True, output=json.dumps(info.get("files") or [], indent=2))

class WritePlanTool:
    name = "write_plan"
    description = (
        "Write/update .clawagents/plan.md for Plan→Act handoff "
        "(goals, steps, risks, files)."
    )
    parameters = {
        "content": {
            "type": "string",
            "description": "Full markdown plan body",
            "required": True,
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        root = Path(self._workspace) / ".clawagents"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "plan.md"
        body = str(args.get("content") or "").strip()
        if not body:
            return ToolResult(success=False, output="", error="content is required")
        if not body.startswith("#"):
            body = "# Plan\n\n" + body
        path.write_text(body + "\n", encoding="utf-8")
        return ToolResult(success=True, output=f"Wrote {path}")


def load_plan_preamble(workspace: str | Path | None = None, max_chars: int = 3_000) -> str:
    path = Path(workspace or Path.cwd()) / ".clawagents" / "plan.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n…"
    return f"## Active Plan\n\n{text}\n"


def create_context_tools(workspace: str | None = None) -> list[Tool]:
    ws = workspace or os.getcwd()
    return [
        RehydrateLedgerTool(ws),
        RecordLedgerTool(ws),
        CoreMemoryViewTool(ws),
        CoreMemoryReplaceTool(ws),
        CoreMemoryAppendTool(ws),
        RepoMapTool(ws),
        CheckpointCreateTool(ws),
        CheckpointRestoreTool(ws),
        CheckpointListTool(ws),
        CheckpointDiffTool(ws),
        WritePlanTool(ws),
    ]
