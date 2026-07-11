"""Worktree isolation tools for parallel agents."""

from __future__ import annotations

import json
import os
from typing import Any

from clawagents.tools.registry import Tool, ToolResult
from clawagents.tools.worktree import create_worktree, list_worktrees, remove_worktree


class WorktreeCreateTool:
    name = "worktree_create"
    description = (
        "Create a git worktree under .clawagents/worktrees/ for isolated parallel work. "
        "Point a subagent at the returned path."
    )
    parameters = {
        "name": {"type": "string", "description": "Optional worktree slug"},
        "branch": {"type": "string", "description": "Optional branch name"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        info = create_worktree(
            workspace=self._workspace,
            name=str(args.get("name") or "") or None,
            branch=str(args.get("branch") or "") or None,
        )
        if not info.get("ok"):
            return ToolResult(success=False, output="", error=str(info.get("error")))
        return ToolResult(success=True, output=json.dumps(info, indent=2))


class WorktreeListTool:
    name = "worktree_list"
    description = "List git worktrees for this repository."
    parameters: dict[str, Any] = {}

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        rows = list_worktrees(self._workspace)
        return ToolResult(success=True, output=json.dumps(rows, indent=2))


class WorktreeRemoveTool:
    name = "worktree_remove"
    description = "Remove a git worktree by path."
    parameters = {
        "path": {"type": "string", "description": "Worktree path", "required": True},
        "force": {"type": "boolean", "description": "Force remove"},
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        info = remove_worktree(
            str(args.get("path") or ""),
            workspace=self._workspace,
            force=bool(args.get("force")),
        )
        if not info.get("ok"):
            return ToolResult(success=False, output="", error=str(info.get("error")))
        return ToolResult(success=True, output=str(info))


def create_worktree_tools(workspace: str | None = None) -> list[Tool]:
    ws = workspace or os.getcwd()
    return [WorktreeCreateTool(ws), WorktreeListTool(ws), WorktreeRemoveTool(ws)]
