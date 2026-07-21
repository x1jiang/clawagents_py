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

class SnapshotDiffTool:
    name = "snapshot_diff"
    description = (
        "Diff the working tree against a pre-edit file snapshot under "
        ".clawagents/snapshots/ (git-free review). Use when is_git_repo is false "
        "or before claiming edits are correct. Optional path limits to one file; "
        "optional snapshot id (directory name); default is the oldest snapshot "
        "still on disk (session-start baseline when available)."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Optional workspace-relative file to diff",
        },
        "snapshot": {
            "type": "string",
            "description": "Snapshot directory name under .clawagents/snapshots/",
        },
        "max_chars": {
            "type": "integer",
            "description": "Cap returned diff chars (default 24000)",
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        import difflib

        root = Path(self._workspace).resolve()
        snap_root = root / ".clawagents" / "snapshots"
        if not snap_root.is_dir():
            return ToolResult(
                success=True,
                output=(
                    "No .clawagents/snapshots/ yet — edit a file with apply_patch / "
                    "write_file / hashline_edit first (snapshots are taken pre-write)."
                ),
            )
        dirs = sorted(
            [p for p in snap_root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        if not dirs:
            return ToolResult(success=True, output="No snapshot directories present.")
        wanted = str(args.get("snapshot") or "").strip()
        if wanted:
            snap_dir = snap_root / wanted
            if not snap_dir.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"snapshot not found: {wanted}. Available: {[d.name for d in dirs[-8:]]}",
                )
        else:
            snap_dir = dirs[0]  # oldest ≈ session-start baseline

        rel = str(args.get("path") or "").strip()
        try:
            max_chars = int(args.get("max_chars") or 24_000)
        except (TypeError, ValueError):
            max_chars = 24_000

        files: list[Path]
        if rel:
            cand = (snap_dir / rel).resolve()
            try:
                cand.relative_to(snap_dir.resolve())
            except ValueError:
                return ToolResult(success=False, output="", error="path escapes snapshot")
            files = [cand] if cand.is_file() else []
            if not files:
                return ToolResult(
                    success=True,
                    output=f"No snapshot of {rel!r} in {snap_dir.name}",
                )
        else:
            files = [p for p in snap_dir.rglob("*") if p.is_file()]

        total_files = len(files)
        file_cap = 40
        shown_files = files[:file_cap]
        header = f"Snapshot baseline: {snap_dir.name} ({total_files} file(s))"
        if total_files > file_cap:
            header += (
                f" — showing {file_cap} of {total_files}; "
                "pass path= to focus on a specific file"
            )
        parts: list[str] = [header]
        for snap_file in shown_files:
            try:
                rel_path = snap_file.relative_to(snap_dir)
            except ValueError:
                continue
            cur = root / rel_path
            try:
                before = snap_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                parts.append(f"\n## {rel_path}\n(read snapshot failed: {exc})")
                continue
            if not cur.is_file():
                parts.append(f"\n## {rel_path}\n(deleted in working tree)")
                continue
            try:
                after = cur.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                parts.append(f"\n## {rel_path}\n(read working tree failed: {exc})")
                continue
            if before == after:
                continue
            diff = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"snapshot/{rel_path}",
                    tofile=f"worktree/{rel_path}",
                    n=2,
                )
            )
            parts.append(f"\n## {rel_path}\n{diff or '(changed but empty diff)'}")

        if len(parts) == 1:
            parts.append("\n(no textual differences vs snapshot)")
        text = parts[0] + "".join(parts[1:])
        if len(text) > max_chars:
            text = text[: max_chars - 40] + "\n… [snapshot_diff truncated] …\n"
        return ToolResult(success=True, output=text)


class WritePlanTool:
    name = "write_plan"
    description = (
        "Write/update .clawagents/plan.md for Plan→Act handoff "
        "(goals, invariants, steps, risks, files). Before a publish/deploy-style "
        "side effect, add exact pre-action shell commands as backticked bullets "
        "under 'Verification gates' and exact remote-state/count/marker checks "
        "under 'Post-action reconciliation'. Act mode requires both phases. "
        "For production pipelines, cover retry/rollback and partial-failure "
        "behavior, observable evidence of resulting external state, and any "
        "domain-specific safety constraints discovered from the task."
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

    async def execute(self, args: dict[str, Any], run_context: Any = None) -> ToolResult:
        root = Path(self._workspace) / ".clawagents"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "plan.md"
        body = str(args.get("content") or "").strip()
        if not body:
            return ToolResult(success=False, output="", error="content is required")
        if not body.startswith("#"):
            body = "# Plan\n\n" + body
        path.write_text(body + "\n", encoding="utf-8")
        if run_context is not None and isinstance(
            getattr(run_context, "_metadata", None), dict
        ):
            run_context._metadata["pending_plan_text"] = body
        from clawagents.config.features import is_enabled
        from clawagents.permissions.act_invariants import (
            clear_contract,
            mark_plan_pending,
        )
        from clawagents.permissions.mode import PermissionMode

        if is_enabled("act_invariant_gate"):
            if (
                run_context is not None
                and getattr(run_context, "permission_mode", None) == PermissionMode.PLAN
            ):
                mark_plan_pending(run_context, body, workspace=self._workspace)
            else:
                clear_contract(run_context, workspace=self._workspace)
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
    preamble = f"## Active Plan\n\n{text}\n"
    from clawagents.config.features import is_enabled

    if is_enabled("act_invariant_gate"):
        from clawagents.permissions.act_invariants import contract_preamble

        gate = contract_preamble(workspace=workspace)
        if gate:
            preamble += "\n" + gate
    return preamble


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
        SnapshotDiffTool(ws),
        WritePlanTool(ws),
    ]
