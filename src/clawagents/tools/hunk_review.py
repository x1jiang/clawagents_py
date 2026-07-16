"""Tools for attributed hunk review (accept / reject)."""

from __future__ import annotations

import json
from typing import Any, Dict

from clawagents.config.features import is_enabled
from clawagents.memory.attributed_hunks import (
    accept_all,
    accept_hunk,
    list_hunks,
    refresh_file_hunks,
    reject_hunk,
)
from clawagents.tools.registry import Tool, ToolResult


class HunkListTool:
    name = "hunk_list"
    description = (
        "List pending attributed file hunks (baseline vs on-disk). "
        "Optionally refresh a path first."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "path": {
            "type": "string",
            "description": "Optional relative path to filter or refresh",
        },
        "refresh": {
            "type": "boolean",
            "description": "Recompute hunks for path from disk before listing",
        },
    }

    def __init__(self, workspace: str | None = None):
        self._workspace = workspace

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not is_enabled("hunk_review"):
            return ToolResult(success=False, output="", error="hunk_review feature disabled")
        path = args.get("path")
        if path and args.get("refresh", True):
            refresh_file_hunks(str(path), workspace=self._workspace, seed_baseline_if_missing=False)
            # If no baseline yet, seed from empty so all content shows as added
            from clawagents.memory.attributed_hunks import HunkStore

            store = HunkStore.load(self._workspace)
            if str(path) not in store.baselines:
                store.baselines[str(path)] = ""
                store.save()
                refresh_file_hunks(str(path), workspace=self._workspace, seed_baseline_if_missing=False)
        rows = list_hunks(workspace=self._workspace, path=str(path) if path else None)
        payload = [h.to_dict() for h in rows]
        return ToolResult(success=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


class HunkAcceptTool:
    name = "hunk_accept"
    description = (
        "Accept a pending hunk by id (advances baseline; disk unchanged) "
        "or accept all hunks for a path."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "hunk_id": {"type": "string", "description": "Hunk id to accept"},
        "path": {"type": "string", "description": "Accept all hunks for this path"},
    }

    def __init__(self, workspace: str | None = None):
        self._workspace = workspace

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not is_enabled("hunk_review"):
            return ToolResult(success=False, output="", error="hunk_review feature disabled")
        if args.get("path") and not args.get("hunk_id"):
            result = accept_all(str(args["path"]), workspace=self._workspace)
        elif args.get("hunk_id"):
            result = accept_hunk(str(args["hunk_id"]), workspace=self._workspace)
        else:
            return ToolResult(success=False, output="", error="provide hunk_id or path")
        ok = bool(result.get("ok"))
        return ToolResult(
            success=ok,
            output=json.dumps(result, ensure_ascii=False),
            error=None if ok else str(result.get("error") or "accept failed"),
        )


class HunkRejectTool:
    name = "hunk_reject"
    description = "Reject a pending hunk by id — restores that region on disk toward baseline."
    parameters: Dict[str, Dict[str, Any]] = {
        "hunk_id": {
            "type": "string",
            "description": "Hunk id to reject",
            "required": True,
        },
    }

    def __init__(self, workspace: str | None = None):
        self._workspace = workspace

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not is_enabled("hunk_review"):
            return ToolResult(success=False, output="", error="hunk_review feature disabled")
        hid = args.get("hunk_id")
        if not hid:
            return ToolResult(success=False, output="", error="hunk_id required")
        result = reject_hunk(str(hid), workspace=self._workspace)
        ok = bool(result.get("ok"))
        return ToolResult(
            success=ok,
            output=json.dumps(result, ensure_ascii=False),
            error=None if ok else str(result.get("error") or "reject failed"),
        )


def create_hunk_review_tools(workspace: str | None = None) -> list[Tool]:
    return [
        HunkListTool(workspace),
        HunkAcceptTool(workspace),
        HunkRejectTool(workspace),
    ]
