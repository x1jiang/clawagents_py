"""apply_patch — Aider/Codex-style surgical edits via SEARCH/REPLACE or unified diff.

SEARCH/REPLACE uses a **line-based** fence parser so empty REPLACE (deletion)
hunks are representable and fence markers can never be swallowed into content.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from clawagents.tools.registry import Tool, ToolResult

_SEARCH_MARK = "<<<<<<< SEARCH"
_DIVIDER_MARK = "======="
_REPLACE_MARK = ">>>>>>> REPLACE"
_FENCE_MARKS = frozenset({_SEARCH_MARK, _DIVIDER_MARK, _REPLACE_MARK})
_UNIFIED_HUNK = re.compile(r"(?m)^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")


def _fence_kind(line: str) -> str | None:
    """Recognize fence markers even with trailing spaces on the marker line."""
    stripped = (line or "").rstrip()
    if stripped == _SEARCH_MARK:
        return "search"
    if stripped == _DIVIDER_MARK:
        return "divider"
    if stripped == _REPLACE_MARK:
        return "replace"
    return None


def _has_fence_markers(text: str) -> bool:
    for ln in text.splitlines():
        kind = _fence_kind(ln)
        if kind in ("search", "replace"):
            return True
        # Bare ======= alone is too common in markdown; only flag with SEARCH/REPLACE.
    return False


def _ws_collapse(s: str) -> str:
    """Collapse leading/trailing/internal whitespace per line (tabs↔spaces)."""
    return "\n".join(" ".join(ln.split()) for ln in s.splitlines())


def _locate_collapsed_span(content: str, search: str) -> str | None:
    """Return the original file span that uniquely matches collapsed SEARCH."""
    needle = _ws_collapse(search)
    if not needle.strip():
        return None
    content_lines = content.splitlines()
    needle_lines = needle.split("\n")
    n = len(needle_lines)
    if n == 0 or n > len(content_lines):
        return None
    hits: list[int] = []
    for i in range(0, len(content_lines) - n + 1):
        window = "\n".join(content_lines[i : i + n])
        if _ws_collapse(window) == needle:
            hits.append(i)
    if len(hits) != 1:
        return None
    i = hits[0]
    matched = "\n".join(content_lines[i : i + n])
    # Prefer newline-terminated span when the file continues after it.
    with_nl = matched + "\n"
    if with_nl in content:
        return with_nl
    if matched in content:
        return matched
    return None


def _nearest_search_hint(content: str, search: str) -> str:
    """Port of edit_file's nearest-line hint for SEARCH misses."""
    needle = (search or "").strip()
    if not needle:
        return ""
    first = needle.splitlines()[0].strip()
    if not first:
        return ""
    best: tuple[float, int, str] | None = None
    for i, line in enumerate(content.splitlines()):
        ratio = difflib.SequenceMatcher(None, first, line.strip()).ratio()
        if best is None or ratio > best[0]:
            best = (ratio, i + 1, line)
    if best is None or best[0] < 0.55:
        return ""
    score, lineno, line = best
    preview = line if len(line) <= 120 else line[:117] + "..."
    return (
        f" Nearest similar line ~{lineno} (similarity {score:.0%}): {preview!r}."
    )


def _parse_search_replace_hunks(patch: str) -> tuple[list[tuple[str, str]] | None, str]:
    """Parse Aider fences into (search, replace) pairs.

    Fence marker lines may carry trailing spaces. Empty REPLACE (deletion) is
    valid: ``=======`` immediately followed by ``>>>>>>> REPLACE``.
    """
    lines = patch.splitlines()
    hunks: list[tuple[str, str]] = []
    i = 0
    # Skip leading non-fence preamble
    while i < len(lines) and _fence_kind(lines[i]) != "search":
        i += 1
    if i >= len(lines):
        return None, "malformed SEARCH/REPLACE fences (no <<<<<<< SEARCH)"

    while i < len(lines):
        if _fence_kind(lines[i]) != "search":
            return None, (
                f"unexpected content at line {i + 1} while expecting {_SEARCH_MARK!r} "
                f"(unconsumed fence or garbage between hunks)"
            )
        i += 1
        search_lines: list[str] = []
        while i < len(lines) and _fence_kind(lines[i]) != "divider":
            if _fence_kind(lines[i]) in ("search", "replace"):
                return None, (
                    f"unexpected fence marker {lines[i]!r} inside SEARCH block "
                    f"(line {i + 1})"
                )
            search_lines.append(lines[i])
            i += 1
        if i >= len(lines):
            return None, "SEARCH block not closed with ======="
        i += 1  # skip =======
        replace_lines: list[str] = []
        while i < len(lines) and _fence_kind(lines[i]) != "replace":
            if _fence_kind(lines[i]) in ("search", "divider"):
                return None, (
                    f"unexpected fence marker {lines[i]!r} inside REPLACE block "
                    f"(line {i + 1})"
                )
            replace_lines.append(lines[i])
            i += 1
        if i >= len(lines):
            return None, "REPLACE block not closed with >>>>>>> REPLACE"
        i += 1  # skip >>>>>>> REPLACE
        search = "\n".join(search_lines)
        replace = "\n".join(replace_lines)
        hunks.append((search, replace))
        while i < len(lines) and lines[i].strip() == "":
            i += 1

    if not hunks:
        return None, "no SEARCH/REPLACE hunks found"
    leftover = [ln for ln in lines[i:] if ln.strip()]
    if leftover:
        return None, (
            "unconsumed patch content after last hunk — refuse apply "
            f"(starts with {leftover[0]!r})"
        )
    return hunks, "ok"


def _apply_search_replace(content: str, search: str, replace: str) -> tuple[bool, str, str]:
    if search == "":
        return False, content, "empty SEARCH block"

    # Line-based hunks usually omit a trailing newline on the last SEARCH line.
    # Prefer matching ``search + "\n"`` when the file uses newline-terminated lines.
    matched: str | None = None
    for cand in (search + "\n", search):
        if cand in content:
            matched = cand
            break

    if matched is None:
        # Soft match: collapse whitespace (leading/tabs/internal) and apply
        # when the collapsed span is unique in the file.
        soft = _locate_collapsed_span(content, search)
        if soft is not None:
            matched = soft
        else:
            hint = _nearest_search_hint(content, search)
            return (
                False,
                content,
                "SEARCH block not found (even after whitespace normalize). "
                "Re-read the file (or use hashline_grep → hashline_edit) and "
                "copy the exact current text into SEARCH."
                + hint,
            )

    if content.count(matched) > 1:
        return False, content, "SEARCH block matches multiple locations — make it unique"

    if replace == "":
        new_content = content.replace(matched, "", 1)
    elif matched.endswith("\n") and not replace.endswith("\n"):
        new_content = content.replace(matched, replace + "\n", 1)
    else:
        new_content = content.replace(matched, replace, 1)
    return True, new_content, "ok"


def _unified_diff_summary(before: str, after: str, path: str, *, max_lines: int = 80) -> str:
    diff = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=2,
        )
    )
    if not diff:
        return "(no textual change)"
    if len(diff) > max_lines:
        head = "".join(diff[:max_lines])
        return head + f"\n… [{len(diff) - max_lines} more diff lines omitted] …\n"
    return "".join(diff)


def _apply_unified_diff(content: str, patch: str) -> tuple[bool, str, str]:
    """Minimal single-file unified diff applier (context-aware)."""
    src = content.splitlines()
    out: list[str] = []
    src_i = 0
    matches = list(_UNIFIED_HUNK.finditer(patch))
    if not matches:
        return False, content, "no hunks found"

    for m in matches:
        old_start = int(m.group(1)) - 1
        start = m.end()
        nxt = _UNIFIED_HUNK.search(patch, start)
        hunk_body = patch[start : nxt.start() if nxt else len(patch)].lstrip("\n")
        if old_start < src_i:
            return False, content, f"hunk overlap at line {old_start + 1}"
        out.extend(src[src_i:old_start])
        cursor = old_start
        for raw in hunk_body.splitlines():
            if raw.startswith("\\"):  # "\ No newline"
                continue
            if raw == "":
                # Mid-hunk blank lines (models emit these) — refuse, don't silently skip.
                return False, content, (
                    f"blank line in hunk body near file line {cursor + 1} — "
                    "unified diffs must prefix every body line with ' ', '-', or '+'"
                )
            tag, text = raw[0], raw[1:]
            if tag not in " +-":
                return False, content, f"invalid hunk line prefix {tag!r} at file line {cursor + 1}"
            if tag == " ":
                if cursor >= len(src) or (
                    src[cursor] != text and src[cursor].rstrip() != text.rstrip()
                ):
                    return False, content, f"context mismatch at line {cursor + 1}"
                out.append(src[cursor])
                cursor += 1
            elif tag == "-":
                if cursor >= len(src) or (
                    src[cursor] != text and src[cursor].rstrip() != text.rstrip()
                ):
                    return False, content, f"delete mismatch at line {cursor + 1}"
                cursor += 1
            else:  # +
                out.append(text)
        src_i = cursor
    out.extend(src[src_i:])
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
        "Empty REPLACE deletes the SEARCH block. Prefer this over write_file for "
        "localized changes. Returns a unified diff of what changed. "
        "Do not send Cursor/Codex '*** Begin Patch' envelopes — use SEARCH/REPLACE "
        "or a plain unified diff body with @@ hunks."
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
            had_fences = _has_fence_markers(content)
            display_path = str(args.get("path", file_path))

            if _SEARCH_MARK in patch:
                hunks, parse_msg = _parse_search_replace_hunks(patch)
                if hunks is None:
                    return ToolResult(success=False, output="", error=parse_msg)
                new_content = content
                applied = 0
                for search, replace in hunks:
                    ok, new_content, msg = _apply_search_replace(new_content, search, replace)
                    if not ok:
                        return ToolResult(success=False, output="", error=msg)
                    applied += 1
                if not had_fences and _has_fence_markers(new_content):
                    return ToolResult(
                        success=False,
                        output="",
                        error=(
                            "refuse write: apply_patch would introduce SEARCH/REPLACE "
                            "fence markers into the file (parser/corruption guard). "
                            "Fix the patch and retry."
                        ),
                    )
                await sb.write_file(file_path, new_content)
                diff = _unified_diff_summary(content, new_content, display_path)
                return ToolResult(
                    success=True,
                    output=(
                        f"Applied {applied} SEARCH/REPLACE hunk(s) to {display_path}\n\n"
                        f"{diff}"
                    ),
                )
            if "*** Begin Patch" in patch or "*** Update File:" in patch:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "unsupported patch envelope (*** Begin Patch). "
                        "Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE fences "
                        "or a unified diff with @@ hunks, or call edit_file instead."
                    ),
                )
            ok, new_content, msg = _apply_unified_diff(content, patch)
            if not ok:
                return ToolResult(success=False, output="", error=msg)
            if not had_fences and _has_fence_markers(new_content):
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "refuse write: apply_patch would introduce SEARCH/REPLACE "
                        "fence markers into the file (corruption guard)."
                    ),
                )
            await sb.write_file(file_path, new_content)
            diff = _unified_diff_summary(content, new_content, display_path)
            return ToolResult(
                success=True,
                output=f"Applied unified diff to {display_path}\n\n{diff}",
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=f"apply_patch failed: {exc}")


def create_apply_patch_tool(sb: Any) -> Tool:
    return ApplyPatchTool(sb)
