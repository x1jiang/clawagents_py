"""Filesystem Tools — backed by a pluggable SandboxBackend.

Provides: ls (with metadata), read_file, write_file, edit_file, grep, glob

Default export uses LocalBackend (real filesystem). Call
``create_filesystem_tools(backend)`` to plug in InMemoryBackend or DockerBackend.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from clawagents.tools.registry import Tool, ToolResult

IGNORE_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env",
    "__pycache__", "dist", "build", ".next", ".cache",
    ".idea", ".vscode", "coverage",
}


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def _matches_glob(name: str, pattern: str) -> bool:
    if pattern == "*":
        return True
    if pattern.startswith("*."):
        return name.endswith(pattern[1:])
    return name == pattern


# ─── Tool classes that accept a SandboxBackend ──────────────────────────────

class LsTool:
    name = "ls"
    description = "List files and directories with metadata (size, modified time)."
    parameters = {
        "path": {"type": "string", "description": "Path to list. Default: current directory", "required": True}
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        target_path = sb.safe_path(str(args.get("path", ".")))
        try:
            entries = await sb.read_dir(target_path)
            entries.sort(key=lambda e: (not e.is_directory, e.name.lower()))

            lines: list[str] = []
            for e in entries:
                try:
                    full_path = sb.resolve(target_path, e.name)
                    s = await sb.stat(full_path)
                    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.mtime_ms / 1000))
                    if e.is_directory:
                        lines.append(f"[DIR]  {e.name}/")
                    else:
                        lines.append(f"[FILE] {e.name} ({_format_size(s.size)}, {mtime})")
                except OSError:
                    lines.append(f"[????] {e.name}")

            return ToolResult(success=True, output="\n".join(lines) or "(empty directory)")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"ls failed: {str(e)}")


class ReadFileTool:
    name = "read_file"
    description = "Read file contents with line numbers. Supports offset/limit for pagination."
    parameters = {
        "path": {"type": "string", "description": "Path to the file to read", "required": True},
        "offset": {"type": "number", "description": "Line number to start from (0-indexed). Default: 0"},
        "limit": {"type": "number", "description": "Max lines to return. Default: 100"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        file_path = sb.safe_path(str(args.get("path", "")))
        try:
            offset = max(0, int(args.get("offset", 0)))
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = max(1, int(args.get("limit", 100)))
        except (TypeError, ValueError):
            limit = 100

        try:
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            if ext in {"png", "jpg", "jpeg", "webp"}:
                import base64
                raw = await sb.read_file_bytes(file_path)
                if len(raw) > 15 * 1024 * 1024:
                    return ToolResult(success=False, output="", error="Image too large (max 15MB)")
                b64_img = base64.b64encode(raw).decode("utf-8")
                mime = "jpeg" if ext == "jpg" else ext
                return ToolResult(
                    success=True,
                    output=[
                        {"type": "text", "text": f"Image loaded from {file_path}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_img}"}},
                    ],
                )

            content = await sb.read_file(file_path)
            lines = content.splitlines()
            slice_lines = lines[offset:offset + limit]
            numbered = [f"{str(offset + i + 1).rjust(4)}: {line}" for i, line in enumerate(slice_lines)]
            header = f"File: {file_path} ({len(lines)} lines total, showing {offset + 1}-{offset + len(slice_lines)})"
            return ToolResult(success=True, output=header + "\n" + "\n".join(numbered))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"read_file failed: {str(e)}")


class WriteFileTool:
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    parameters = {
        "path": {"type": "string", "description": "Path to write the file", "required": True},
        "content": {"type": "string", "description": "Content to write to the file", "required": True},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        file_path = sb.safe_path(str(args.get("path", "")))
        content = str(args.get("content", ""))

        try:
            parent = sb.dirname(file_path)
            if not await sb.exists(parent):
                await sb.mkdir(parent, recursive=True)
            await sb.write_file(file_path, content)
            return ToolResult(success=True, output=f"Wrote {len(content)} bytes to {file_path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"write_file failed: {str(e)}")


class EditFileTool:
    name = "edit_file"
    description = "Edit a file by replacing a specific block of text. The target must exactly match existing content."
    parameters = {
        "path": {"type": "string", "description": "Path to the file to edit", "required": True},
        "target": {"type": "string", "description": "The exact block of text to replace", "required": True},
        "replacement": {"type": "string", "description": "The new text", "required": True},
        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false, requires unique match)"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        file_path = sb.safe_path(str(args.get("path", "")))
        target = str(args.get("target", ""))
        replacement = str(args.get("replacement", ""))
        replace_all = bool(args.get("replace_all", False))

        try:
            if not await sb.exists(file_path):
                return ToolResult(success=False, output="", error=f"edit_file failed: File does not exist at {file_path}")

            content = await sb.read_file(file_path)

            if target not in content:
                return ToolResult(success=False, output="", error=f"edit_file failed: Could not find exact target text in {file_path}. Check whitespace and line endings.")

            count = content.count(target)
            if count > 1 and not replace_all:
                return ToolResult(success=False, output="", error=f"edit_file failed: Target text appears {count} times. Use replace_all=true or provide a more specific target.")

            if replace_all:
                new_content = content.replace(target, replacement)
            else:
                new_content = content.replace(target, replacement, 1)

            await sb.write_file(file_path, new_content)
            return ToolResult(
                success=True,
                output=f"Edited {file_path}: replaced {count if replace_all else 1} occurrence(s) ({len(target)} → {len(replacement)} bytes)",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"edit_file failed: {str(e)}")


class GrepTool:
    name = "grep"
    description = "Search for a text pattern in files. Supports recursive multi-file search with glob filtering."
    parameters = {
        "path": {"type": "string", "description": "File or directory to search", "required": True},
        "pattern": {"type": "string", "description": "Text pattern to search for", "required": True},
        "glob_filter": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py'). Only used when path is a directory."},
        "recursive": {"type": "boolean", "description": "Search recursively in subdirectories. Default: false"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        target = sb.safe_path(str(args.get("path", "")))
        pattern = str(args.get("pattern", ""))
        glob_filter = str(args.get("glob_filter", "*"))
        recursive = bool(args.get("recursive", False))

        if not pattern:
            return ToolResult(success=False, output="", error="No pattern provided")

        try:
            try:
                target_stat = await sb.stat(target)
            except (FileNotFoundError, OSError):
                return ToolResult(success=False, output="", error=f"Path does not exist: {target}")

            if target_stat.is_file:
                return await self._search_file(sb, target, pattern)

            if not target_stat.is_directory:
                return ToolResult(success=False, output="", error=f"Path does not exist: {target}")

            all_matches: list[str] = []
            files_searched = 0
            max_matches = 100

            async for file_path in _walk_dir(sb, target, glob_filter, recursive):
                files_searched += 1
                try:
                    content = await sb.read_file(file_path)
                except (UnicodeDecodeError, OSError, FileNotFoundError):
                    continue

                rel = sb.relative(target, file_path)
                for i, line in enumerate(content.splitlines()):
                    if pattern in line:
                        all_matches.append(f"{rel}:{i + 1}: {line.strip()}")
                        if len(all_matches) >= max_matches:
                            break
                if len(all_matches) >= max_matches:
                    break

            if not all_matches:
                return ToolResult(success=True, output=f'No matches for "{pattern}" in {files_searched} files under {target}')

            truncated = f" (truncated at {max_matches})" if len(all_matches) >= max_matches else ""
            return ToolResult(success=True, output=f"{len(all_matches)} match(es) in {files_searched} files{truncated}:\n" + "\n".join(all_matches))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"grep failed: {str(e)}")

    @staticmethod
    async def _search_file(sb: Any, file_path: str, pattern: str) -> ToolResult:
        try:
            content = await sb.read_file(file_path)
            lines = content.splitlines()
            matches = [{"line": line, "num": i + 1} for i, line in enumerate(lines) if pattern in line]

            if not matches:
                return ToolResult(success=True, output=f'No matches for "{pattern}" in {file_path}')

            output = "\n".join([f"{str(m['num']).rjust(4)}: {m['line']}" for m in matches])
            return ToolResult(success=True, output=f"{len(matches)} match(es) in {file_path}:\n{output}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"grep failed: {str(e)}")


class GlobTool:
    name = "glob"
    description = "Find files matching a glob pattern. Use '**/*.py' for recursive search."
    parameters = {
        "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')", "required": True},
        "path": {"type": "string", "description": "Root directory to search from. Default: current directory"},
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sb = self._sb
        root = sb.safe_path(str(args.get("path", ".")))
        pattern = str(args.get("pattern", ""))

        if not pattern:
            return ToolResult(success=False, output="", error="No glob pattern provided")

        try:
            try:
                root_stat = await sb.stat(root)
            except (FileNotFoundError, OSError):
                return ToolResult(success=False, output="", error=f"Directory does not exist: {root}")

            if not root_stat.is_directory:
                return ToolResult(success=False, output="", error=f"Directory does not exist: {root}")

            results: list[str] = []
            max_results = 200
            is_recursive = "**" in pattern
            ext = pattern.split("*.")[-1] if "*." in pattern else ""

            async for file_path in _walk_dir(sb, root, f"*.{ext}" if ext else "*", is_recursive):
                try:
                    rel = sb.relative(root, file_path)
                    s = await sb.stat(file_path)
                    results.append(f"{rel} ({_format_size(s.size)})")
                except (OSError, FileNotFoundError):
                    continue
                if len(results) >= max_results:
                    break

            if not results:
                return ToolResult(success=True, output=f"No files matching '{pattern}' in {root}")

            truncated = f" (showing first {max_results})" if len(results) >= max_results else ""
            return ToolResult(success=True, output=f"{len(results)} file(s) matching '{pattern}'{truncated}:\n" + "\n".join(results))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"glob failed: {str(e)}")


async def _walk_dir(sb: Any, directory: str, glob_filter: str, recursive: bool):
    """Async generator that walks a directory tree via the sandbox backend."""
    try:
        entries = await sb.read_dir(directory)
    except (OSError, FileNotFoundError):
        return

    for entry in sorted(entries, key=lambda e: e.name):
        full_path = sb.resolve(directory, entry.name)
        if entry.is_directory:
            if recursive and entry.name not in IGNORE_DIRS:
                async for fp in _walk_dir(sb, full_path, glob_filter, recursive):
                    yield fp
        elif entry.is_file:
            if _matches_glob(entry.name, glob_filter):
                yield full_path


# ─── Public API ──────────────────────────────────────────────────────────────

def create_filesystem_tools(backend: Any) -> List[Tool]:
    """Create filesystem tools backed by a specific SandboxBackend."""
    return [
        LsTool(backend),
        ReadFileTool(backend),
        WriteFileTool(backend),
        EditFileTool(backend),
        GrepTool(backend),
        GlobTool(backend),
    ]


def _default_backend() -> Any:
    from clawagents.sandbox.local import LocalBackend
    return LocalBackend()


filesystem_tools: List[Tool] = create_filesystem_tools(_default_backend())
