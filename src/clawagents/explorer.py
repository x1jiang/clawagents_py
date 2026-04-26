from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clawagents.tools.registry import ToolRegistry, ToolResult


def _safe_path(root: Path, user_path: str) -> Path:
    p = Path(user_path or ".")
    if not p.is_absolute():
        p = root / p
    resolved = p.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path traversal blocked: {user_path}")
    return resolved


class _ExplorerListTools:
    name = "explorer_list_tools"
    description = "List the ClawAgents tools registered in the supplied registry."
    parameters: dict[str, dict[str, Any]] = {}
    parallel_safe = True

    def __init__(self, tools: ToolRegistry | None):
        self._tools = tools

    async def execute(self, args):
        return ToolResult(True, json.dumps(self._tools.inspect_tools() if self._tools else []))


class _ExplorerReadSource:
    name = "explorer_read_source"
    description = "Read a source file under the explorer root."
    parameters = {"path": {"type": "string", "description": "Path relative to the explorer root.", "required": True}}
    parallel_safe = True

    def __init__(self, root: Path):
        self._root = root

    async def execute(self, args):
        try:
            p = _safe_path(self._root, str(args.get("path") or ""))
            if not p.is_file():
                return ToolResult(False, "", "Not a file")
            return ToolResult(True, p.read_text(encoding="utf-8"))
        except Exception as err:
            return ToolResult(False, "", str(err))


class _ExplorerListDirectory:
    name = "explorer_list_directory"
    description = "List files and directories under the explorer root."
    parameters = {"path": {"type": "string", "description": "Directory path relative to the explorer root."}}
    parallel_safe = True

    def __init__(self, root: Path):
        self._root = root

    async def execute(self, args):
        try:
            p = _safe_path(self._root, str(args.get("path") or "."))
            rows = [
                {
                    "name": child.name,
                    "path": str(child.relative_to(self._root)),
                    "is_directory": child.is_dir(),
                    "is_file": child.is_file(),
                }
                for child in p.iterdir()
            ]
            return ToolResult(True, json.dumps(rows))
        except Exception as err:
            return ToolResult(False, "", str(err))


class _ExplorerArchitecture:
    name = "explorer_architecture"
    description = "Return a compact architecture summary for the current ClawAgents package."
    parameters: dict[str, dict[str, Any]] = {}
    parallel_safe = True

    def __init__(self, root: Path):
        self._root = root

    async def execute(self, args):
        return ToolResult(True, json.dumps({
            "root": str(self._root),
            "modules": ["agent", "graph", "tools", "sandbox", "session", "trajectory", "rl", "mcp", "gateway"],
        }))


def create_explorer_tools(
    *,
    root: str | Path | None = None,
    tools: ToolRegistry | None = None,
):
    root_path = Path(root or Path.cwd()).resolve()
    return [
        _ExplorerListTools(tools),
        _ExplorerReadSource(root_path),
        _ExplorerListDirectory(root_path),
        _ExplorerArchitecture(root_path),
    ]
