"""Advanced Filesystem Tools — tree, diff, insert_lines

Backed by a pluggable SandboxBackend (same as filesystem.py).
"""

from __future__ import annotations

from typing import Any, Dict, List

from clawagents.tools.registry import Tool, ToolResult

IGNORE_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env",
    "__pycache__", "dist", "build", ".next", ".cache",
    ".idea", ".vscode", "coverage", ".tox", ".mypy_cache",
}


class TreeTool:
    name = "tree"
    description = (
        "Show a recursive directory tree. Much faster than ls for getting a project overview. "
        "Automatically skips node_modules, .git, __pycache__, etc."
    )
    parameters = {
        "path": {"type": "string", "description": "Root directory. Default: current directory"},
        "max_depth": {"type": "number", "description": "Max depth to recurse. Default: 4"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        root = sb.safe_path(str(args.get("path", ".")))
        try:
            max_depth = max(1, min(10, int(args.get("max_depth", 4))))
        except (TypeError, ValueError):
            max_depth = 4

        try:
            root_stat = await sb.stat(root)
        except (FileNotFoundError, OSError):
            return ToolResult(success=False, output="", error=f"Not a directory: {root}")

        if not root_stat.is_directory:
            return ToolResult(success=False, output="", error=f"Not a directory: {root}")

        rel = sb.relative(sb.cwd, root)
        lines = [rel or "."]
        counts = {"files": 0, "dirs": 0}
        MAX_ENTRIES = 500

        async def walk(d: str, prefix: str, depth: int):
            if counts["files"] + counts["dirs"] >= MAX_ENTRIES:
                return

            try:
                entries = await sb.read_dir(d)
                entries.sort(key=lambda e: (not e.is_directory, e.name.lower()))
            except (PermissionError, OSError, FileNotFoundError):
                return

            for i, entry in enumerate(entries):
                if counts["files"] + counts["dirs"] >= MAX_ENTRIES:
                    lines.append(f"{prefix}... (truncated at {MAX_ENTRIES} entries)")
                    return

                is_last = i == len(entries) - 1
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")
                full_path = sb.resolve(d, entry.name)

                if entry.is_directory:
                    counts["dirs"] += 1
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    if depth < max_depth and entry.name not in IGNORE_DIRS:
                        await walk(full_path, child_prefix, depth + 1)
                else:
                    counts["files"] += 1
                    lines.append(f"{prefix}{connector}{entry.name}")

        await walk(root, "", 1)
        lines.append(f"\n{counts['dirs']} directories, {counts['files']} files")
        return ToolResult(success=True, output="\n".join(lines))


class DiffTool:
    name = "diff"
    description = (
        "Compare two files and show their differences in unified diff format. "
        "Useful for reviewing changes before or after edits."
    )
    parameters = {
        "file_a": {"type": "string", "description": "Path to the first file", "required": True},
        "file_b": {"type": "string", "description": "Path to the second file", "required": True},
        "context_lines": {"type": "number", "description": "Lines of context around changes. Default: 3"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        path_a = sb.safe_path(str(args.get("file_a", "")))
        path_b = sb.safe_path(str(args.get("file_b", "")))
        try:
            ctx = max(0, min(20, int(args.get("context_lines", 3))))
        except (TypeError, ValueError):
            ctx = 3

        if not await sb.exists(path_a):
            return ToolResult(success=False, output="", error=f"File not found: {path_a}")
        if not await sb.exists(path_b):
            return ToolResult(success=False, output="", error=f"File not found: {path_b}")

        try:
            import difflib
            lines_a = (await sb.read_file(path_a)).splitlines(keepends=True)
            lines_b = (await sb.read_file(path_b)).splitlines(keepends=True)

            name_a = sb.relative(sb.cwd, path_a)
            name_b = sb.relative(sb.cwd, path_b)

            diff = list(difflib.unified_diff(lines_a, lines_b, fromfile=name_a, tofile=name_b, n=ctx))
            if not diff:
                return ToolResult(success=True, output="Files are identical.")

            return ToolResult(success=True, output="".join(diff))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"diff failed: {str(e)}")


class InsertLinesTool:
    name = "insert_lines"
    description = (
        "Insert text at a specific line number in a file. Line 0 inserts at the top; "
        "a line beyond the file length appends at the end. More precise than edit_file for adding new code."
    )
    parameters = {
        "path": {"type": "string", "description": "Path to the file", "required": True},
        "line": {"type": "number", "description": "Line number to insert before (1-indexed). 0 = top of file.", "required": True},
        "content": {"type": "string", "description": "The text to insert", "required": True},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        file_path = sb.safe_path(str(args.get("path", "")))
        try:
            line_num = max(0, int(args.get("line", 0)))
        except (TypeError, ValueError):
            line_num = 0
        content = str(args.get("content", ""))

        if not content:
            return ToolResult(success=False, output="", error="No content to insert")

        try:
            if not await sb.exists(file_path):
                return ToolResult(success=False, output="", error=f"File not found: {file_path}")

            existing = await sb.read_file(file_path)
            lines = existing.split("\n")
            insert_idx = min(line_num, len(lines))

            new_lines = content.split("\n")
            for i, nl in enumerate(new_lines):
                lines.insert(insert_idx + i, nl)

            await sb.write_file(file_path, "\n".join(lines))

            return ToolResult(
                success=True,
                output=f"Inserted {len(new_lines)} line(s) at line {insert_idx} in {file_path} (now {len(lines)} lines total)",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"insert_lines failed: {str(e)}")


# ─── Public API ──────────────────────────────────────────────────────────────

def create_advanced_fs_tools(backend: Any) -> List[Tool]:
    """Create advanced filesystem tools backed by a specific SandboxBackend."""
    return [TreeTool(backend), DiffTool(backend), InsertLinesTool(backend)]


def _default_backend() -> Any:
    from clawagents.sandbox.local import LocalBackend
    return LocalBackend()


advanced_fs_tools: List[Tool] = create_advanced_fs_tools(_default_backend())
