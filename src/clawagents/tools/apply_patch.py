"""apply_patch — Aider/Codex-style surgical edits via SEARCH/REPLACE or unified diff.

SEARCH/REPLACE uses a **line-based** fence parser so empty REPLACE (deletion)
hunks are representable and fence markers can never be swallowed into content.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from clawagents.tools.registry import Tool, ToolResult

_SEARCH_MARK = "<<<<<<< SEARCH"
_DIVIDER_MARK = "======="
_REPLACE_MARK = ">>>>>>> REPLACE"
_FENCE_MARKS = frozenset({_SEARCH_MARK, _DIVIDER_MARK, _REPLACE_MARK})
_UNIFIED_HUNK = re.compile(r"(?m)^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")
_CODEX_UPDATE = re.compile(r"(?m)^\*\*\*\s*Update\s+File:\s*(.+?)\s*$")
_CODEX_END = re.compile(r"(?m)^\*\*\*\s*End\s+Patch\s*$")
_CODEX_HUNK = re.compile(r"(?m)^@@.*$")


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
    actual = line.strip()
    common = 0
    for expected_char, actual_char in zip(first, actual):
        if expected_char != actual_char:
            break
        common += 1
    if common == min(len(first), len(actual)) and len(first) == len(actual):
        difference = ""
    else:
        expected_tail = first[common : common + 24]
        actual_tail = actual[common : common + 24]
        difference = (
            f" First difference at column {common + 1}: SEARCH has "
            f"{expected_tail!r}, file has {actual_tail!r}."
        )
    structure = ""
    if first.startswith("- ") and actual.startswith("| "):
        structure = (
            " SEARCH uses a list marker (`- `), but the file uses a Markdown "
            "table row (`| ... |`); copy the complete table row including pipes."
        )
    elif first.startswith("| ") and actual.startswith("- "):
        structure = (
            " SEARCH uses a Markdown table row, but the file uses a list marker "
            "(`- `); copy the exact current line."
        )
    preview = line if len(line) <= 120 else line[:117] + "..."
    return (
        f" Nearest similar line ~{lineno} (similarity {score:.1%}): {preview!r}."
        f"{difference}{structure}"
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

    occurrences = content.count(matched)
    if occurrences > 1:
        return (
            False,
            content,
            f"SEARCH block matches {occurrences} locations. Do not retry the same "
            "patch. Add unique neighboring context, or use "
            "hashline_grep → hashline_edit for an anchored edit.",
        )

    if replace == "":
        new_content = content.replace(matched, "", 1)
    elif matched.endswith("\n") and not replace.endswith("\n"):
        new_content = content.replace(matched, replace + "\n", 1)
    else:
        new_content = content.replace(matched, replace, 1)
    return True, new_content, "ok"


def _hunk_failure_message(
    message: str,
    *,
    index: int,
    total: int,
    search: str,
    patch: str,
) -> str:
    previous = index - 1
    staged = ""
    if previous:
        noun = "hunk" if previous == 1 else "hunks"
        staged = f" after {previous} earlier {noun} matched in memory"
    first = next((line.strip() for line in search.splitlines() if line.strip()), "")
    excerpt = first if len(first) <= 120 else first[:117] + "..."
    escape_hint = ""
    if "\\n" in patch or '\\"' in patch:
        escape_hint = (
            " Patch contains literal escape sequences such as `\\n` or `\\\"`; "
            "do not copy JSON-escaped tool arguments into patch text unless those "
            "backslashes belong in the file."
        )
    recovery = (
        " Do not resend this patch unchanged. Refresh the target and retry only "
        "the failed logical edit with unique context; for large or repetitive "
        "files prefer hashline_grep → hashline_edit."
    )
    if total > 1:
        recovery += " Split the retry into one localized hunk per call."
    return (
        f"Hunk {index}/{total} failed{staged}. No changes written (atomic). "
        f"SEARCH starts with {excerpt!r}. {message}{escape_hint}{recovery}"
    )


def _prewrite_validation_error(path: str, content: str, patch: str) -> str:
    """Reject structurally invalid formats before the sandbox writes them."""
    if not path.lower().endswith(".json"):
        return ""
    candidate = content[1:] if content.startswith("\ufeff") else content
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        escape_hint = ""
        if "\\n" in patch or '\\"' in patch:
            escape_hint = (
                " The patch contains literal escape sequences; provide ordinary "
                "unescaped JSON lines in SEARCH/REPLACE fences."
            )
        return (
            "refuse write: apply_patch would produce invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}.{escape_hint}"
        )
    return ""


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


def _codex_paths_match(expected: str, header: str) -> bool:
    """True when Codex ``Update File`` path refers to the tool ``path`` arg."""
    e = (expected or "").replace("\\", "/").lstrip("./")
    h = (header or "").replace("\\", "/").lstrip("./")
    if not e or not h:
        return False
    if e == h:
        return True
    if e.endswith("/" + h) or h.endswith("/" + e):
        return True
    # Basename match only when the header is a bare filename.
    return "/" not in h and e.rsplit("/", 1)[-1] == h


def _normalize_codex_envelope(
    patch: str, expected_path: str
) -> tuple[str | None, str]:
    """Convert a single-file Codex ``*** Begin Patch`` envelope to SEARCH/REPLACE.

    Rejects multi-file envelopes and Add/Delete/Move ops. Returns
    ``(fences, "ok")`` or ``(None, error)``.
    """
    if "*** Begin Patch" not in patch and "*** Update File:" not in patch:
        return None, "not codex"
    lowered = patch
    if re.search(r"(?m)^\*\*\*\s*Add\s+File:", lowered):
        return None, (
            "Codex *** Add File not supported here — use write_file, or a "
            "single-file *** Update File envelope"
        )
    if re.search(r"(?m)^\*\*\*\s*Delete\s+File:", lowered):
        return None, "Codex *** Delete File not supported — use delete/edit_file"
    if re.search(r"(?m)^\*\*\*\s*Move\s+(File:|to:)", lowered):
        return None, "Codex move ops not supported — use a dedicated rename tool"
    updates = list(_CODEX_UPDATE.finditer(patch))
    if not updates:
        return None, "Codex envelope missing *** Update File:"
    if len(updates) > 1:
        return None, (
            "multi-file Codex envelope rejected; call apply_patch once per file "
            "with matching path"
        )
    header_path = updates[0].group(1).strip()
    if not _codex_paths_match(expected_path, header_path):
        return None, (
            f"Codex Update File path {header_path!r} does not match tool path "
            f"{expected_path!r}"
        )
    start = updates[0].end()
    end_m = _CODEX_END.search(patch, start)
    body = patch[start : end_m.start() if end_m else len(patch)]
    # Split on @@ hunk headers (optional context hint after @@).
    chunks = _CODEX_HUNK.split(body)
    fence_parts: list[str] = []
    for chunk in chunks:
        body_lines: list[str] = []
        for ln in chunk.splitlines():
            if ln.startswith("***"):
                continue
            if not ln and not body_lines:
                continue
            if ln[:1] in " +-":
                body_lines.append(ln)
            elif not ln.strip():
                body_lines.append(" ")
            else:
                # Unprefixed context (some models omit the leading space).
                body_lines.append(" " + ln)
        if not body_lines:
            continue
        search_lines: list[str] = []
        replace_lines: list[str] = []
        for ln in body_lines:
            tag, text = ln[0], ln[1:]
            if tag == " ":
                search_lines.append(text)
                replace_lines.append(text)
            elif tag == "-":
                search_lines.append(text)
            elif tag == "+":
                replace_lines.append(text)
        if not search_lines and not replace_lines:
            continue
        if not search_lines:
            return None, (
                "addition-only Codex hunk needs context lines (space-prefixed); "
                "re-send with surrounding context or use edit_file"
            )
        fence_parts.append(
            f"{_SEARCH_MARK}\n"
            + "\n".join(search_lines)
            + "\n"
            + f"{_DIVIDER_MARK}\n"
            + (("\n".join(replace_lines) + "\n") if replace_lines else "")
            + f"{_REPLACE_MARK}\n"
        )
    if not fence_parts:
        return None, "Codex envelope had no applicable hunks"
    return "".join(fence_parts), "ok"


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
        "<<<<<<< SEARCH / ======= / >>>>>>> REPLACE fences, a unified diff hunk, "
        "or a single-file Codex *** Begin Patch / *** Update File envelope "
        "(path must match; multi-file envelopes are rejected). "
        "Empty REPLACE deletes the SEARCH block. Prefer this over write_file for "
        "localized changes. Use unique surrounding context and keep each call to "
        "one logical region when possible. After a failure, refresh the file or "
        "switch to hashline_grep/hashline_edit; never resend the same patch "
        "unchanged. Returns a unified diff of what changed."
    )
    parameters = {
        "path": {"type": "string", "description": "File to patch", "required": True},
        "patch": {
            "type": "string",
            "description": (
                "SEARCH/REPLACE fences, unified diff, or single-file Codex "
                "*** Begin Patch envelope for this file"
            ),
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
                for index, (search, replace) in enumerate(hunks, start=1):
                    ok, new_content, msg = _apply_search_replace(new_content, search, replace)
                    if not ok:
                        return ToolResult(
                            success=False,
                            output="",
                            error=_hunk_failure_message(
                                msg,
                                index=index,
                                total=len(hunks),
                                search=search,
                                patch=patch,
                            ),
                        )
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
                validation_error = _prewrite_validation_error(
                    display_path, new_content, patch
                )
                if validation_error:
                    return ToolResult(success=False, output="", error=validation_error)
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
                fences, nmsg = _normalize_codex_envelope(patch, display_path)
                if fences is None:
                    return ToolResult(success=False, output="", error=nmsg)
                hunks, parse_msg = _parse_search_replace_hunks(fences)
                if hunks is None:
                    return ToolResult(success=False, output="", error=parse_msg)
                new_content = content
                applied = 0
                for index, (search, replace) in enumerate(hunks, start=1):
                    ok, new_content, msg = _apply_search_replace(
                        new_content, search, replace
                    )
                    if not ok:
                        return ToolResult(
                            success=False,
                            output="",
                            error=_hunk_failure_message(
                                msg,
                                index=index,
                                total=len(hunks),
                                search=search,
                                patch=patch,
                            ),
                        )
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
                validation_error = _prewrite_validation_error(
                    display_path, new_content, patch
                )
                if validation_error:
                    return ToolResult(success=False, output="", error=validation_error)
                await sb.write_file(file_path, new_content)
                diff = _unified_diff_summary(content, new_content, display_path)
                return ToolResult(
                    success=True,
                    output=(
                        f"Applied {applied} Codex hunk(s) to {display_path}\n\n"
                        f"{diff}"
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
            validation_error = _prewrite_validation_error(
                display_path, new_content, patch
            )
            if validation_error:
                return ToolResult(success=False, output="", error=validation_error)
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
