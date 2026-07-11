"""apply_patch — Aider/Codex-style surgical edits via SEARCH/REPLACE or unified diff."""

from __future__ import annotations

import re
from typing import Any

from clawagents.tools.registry import Tool, ToolResult

_FENCE_RE = re.compile(
    r"<<<<<<< SEARCH\n(?P<search>.*?)\n=======\n(?P<replace>.*?)\n>>>>>>> REPLACE",
    re.S,
)
_UNIFIED_HUNK = re.compile(r"(?m)^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")


def _apply_search_replace(content: str, search: str, replace: str) -> tuple[bool, str, str]:
    if search == "":
        return False, content, "empty SEARCH block"
    if search not in content:
        # fuzzy: normalize trailing whitespace per line
        def norm(s: str) -> str:
            return "\n".join(ln.rstrip() for ln in s.splitlines())

        n_content = norm(content)
        n_search = norm(search)
        if n_search not in n_content:
            return False, content, "SEARCH block not found (even after whitespace normalize)"
        # apply on normalized then rebuild is lossy — fall back to exact fail message
        # Try replace on original by locating first line
        first = search.splitlines()[0].rstrip() if search.splitlines() else ""
        idx = next((i for i, ln in enumerate(content.splitlines()) if ln.rstrip() == first), -1)
        if idx < 0:
            return False, content, "SEARCH block not found"
        return False, content, "SEARCH whitespace mismatch — provide exact file text"
    if content.count(search) > 1:
        return False, content, "SEARCH block matches multiple locations — make it unique"
    return True, content.replace(search, replace, 1), "ok"


def _apply_unified_diff(content: str, patch: str) -> tuple[bool, str, str]:
    """Minimal single-file unified diff applier (context-aware)."""
    # Strip file headers
    body_lines = []
    started = False
    for ln in patch.splitlines(keepends=True):
        if ln.startswith("@@"):
            started = True
        if started:
            body_lines.append(ln)
    if not body_lines:
        return False, content, "no hunks found"

    # Work on list without keepends for easier indexing
    src = content.splitlines()
    out: list[str] = []
    src_i = 0
    hunks = _UNIFIED_HUNK.split(patch)
    # split gives [pre, old_start, new_start, body, old_start, new_start, body, ...]
    if len(hunks) < 4:
        return False, content, "failed to parse hunks"

    # Simpler approach: sequential apply of each hunk body
    for m in _UNIFIED_HUNK.finditer(patch):
        old_start = int(m.group(1)) - 1
        # body until next @@ or end
        start = m.end()
        nxt = _UNIFIED_HUNK.search(patch, start)
        hunk_body = patch[start : nxt.start() if nxt else len(patch)]
        # copy untouched lines
        if old_start < src_i:
            return False, content, f"hunk overlap at line {old_start + 1}"
        out.extend(src[src_i:old_start])
        cursor = old_start
        for raw in hunk_body.splitlines():
            if raw.startswith("\\"):  # "\ No newline"
                continue
            if not raw:
                continue
            tag, text = raw[0], raw[1:]
            if tag == " ":
                if cursor >= len(src) or src[cursor] != text:
                    # allow soft mismatch on trailing ws
                    if cursor >= len(src) or src[cursor].rstrip() != text.rstrip():
                        return False, content, f"context mismatch at line {cursor + 1}"
                out.append(src[cursor])
                cursor += 1
            elif tag == "-":
                if cursor >= len(src) or (
                    src[cursor] != text and src[cursor].rstrip() != text.rstrip()
                ):
                    return False, content, f"delete mismatch at line {cursor + 1}"
                cursor += 1
            elif tag == "+":
                out.append(text)
            else:
                continue
        src_i = cursor
    out.extend(src[src_i:])
    # Preserve final newline convention
    result = "\n".join(out)
    if content.endswith("\n"):
        result += "\n"
    return True, result, "ok"


class ApplyPatchTool:
    name = "apply_patch"
    keywords = ["patch", "diff", "search replace", "surgical edit"]
    description = (
        "Apply a surgical edit to a file using either Aider-style "
        "<<<<<<< SEARCH / ======= / >>>>>>> REPLACE fences, or a unified diff hunk. "
        "Prefer this over write_file for localized changes."
    )
    parameters = {
        "path": {"type": "string", "description": "File to patch", "required": True},
        "patch": {
            "type": "string",
            "description": "SEARCH/REPLACE fences or unified diff for this file",
            "required": True,
        },
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        sb = self._sb
        file_path = sb.safe_path(str(args.get("path", "")))
        patch = str(args.get("patch", ""))
        if not patch.strip():
            return ToolResult(success=False, output="", error="patch is required")
        try:
            if not await sb.exists(file_path):
                return ToolResult(success=False, output="", error=f"file not found: {file_path}")
            content = await sb.read_file(file_path)
            if "<<<<<<< SEARCH" in patch:
                matches = list(_FENCE_RE.finditer(patch))
                if not matches:
                    return ToolResult(
                        success=False, output="", error="malformed SEARCH/REPLACE fences"
                    )
                new_content = content
                applied = 0
                for m in matches:
                    ok, new_content, msg = _apply_search_replace(
                        new_content, m.group("search"), m.group("replace")
                    )
                    if not ok:
                        return ToolResult(success=False, output="", error=msg)
                    applied += 1
                await sb.write_file(file_path, new_content)
                return ToolResult(
                    success=True,
                    output=f"Applied {applied} SEARCH/REPLACE hunk(s) to {file_path}",
                )
            ok, new_content, msg = _apply_unified_diff(content, patch)
            if not ok:
                return ToolResult(success=False, output="", error=msg)
            await sb.write_file(file_path, new_content)
            return ToolResult(success=True, output=f"Applied unified diff to {file_path}")
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"apply_patch failed: {exc}")


def create_apply_patch_tool(sb: Any) -> Tool:
    return ApplyPatchTool(sb)
