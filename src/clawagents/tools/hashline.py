"""Grok Build hashline read/edit tools (feature-gated).

Port of ``xai-grok-tools`` ``grok_build_hashline`` (chunk_v1 scheme):
whitespace-normalized FNV-1a anchors, atomic multi-edit batches, stale-anchor
recovery with fresh snippets.

Additive to ``read_file`` / ``edit_file`` — does not replace them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clawagents.tools.registry import Tool, ToolResult

# ─── Hash (Grok util/hash.rs) ─────────────────────────────────────────────────

_FNV_OFFSET = 2_166_136_261
_FNV_PRIME = 16_777_619
DEFAULT_HASH_LEN = 3
DEFAULT_CHUNK_SIZE = 16
DEFAULT_SEARCH_RADIUS = 15
SNIPPET_CONTEXT = 3
MAX_CONTIGUOUS_SNIPPET = 80
ARROW = "\u2192"


def fnv1a_32(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def line_hash(line: str) -> int:
    """Whitespace-normalized FNV-1a (matches Grok ``util/hash.rs``)."""
    h = _FNV_OFFSET
    prev_ws = False
    for byte in line.strip().encode("utf-8"):
        if byte in (9, 10, 11, 12, 13, 32):  # ASCII whitespace
            if not prev_ws:
                h ^= ord(" ")
                h = (h * _FNV_PRIME) & 0xFFFFFFFF
                prev_ws = True
        else:
            h ^= byte
            h = (h * _FNV_PRIME) & 0xFFFFFFFF
            prev_ws = False
    return h


def encode_hash(hash_val: int, length: int = DEFAULT_HASH_LEN) -> str:
    if not 1 <= length <= 4:
        raise ValueError("encode_hash: len must be 1..=4")
    return "".join(chr(((hash_val >> (i * 8)) % 26) + ord("a")) for i in range(length))


def split_lines(content: str) -> List[str]:
    """Split like Grok: preserve trailing empty line after final newline."""
    if content == "":
        return []
    # splitlines() drops a final empty segment; Grok keeps synthetic trailing "".
    parts = content.split("\n")
    if content.endswith("\n"):
        return parts  # last element is ""
    return parts


# ─── Scheme ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Anchor:
    line: int  # 1-based
    local: str
    context: Optional[str] = None

    def render(self) -> str:
        if self.context:
            return f"{self.line}:{self.local}:{self.context}"
        return f"{self.line}:{self.local}"

    def suffix(self) -> str:
        if self.context:
            return f"{self.local}:{self.context}"
        return self.local


@dataclass(frozen=True)
class ParsedAnchor:
    line: int
    local: str
    context: Optional[str] = None

    @classmethod
    def parse(cls, s: str) -> Optional["ParsedAnchor"]:
        parts = s.split(":", 2)
        if len(parts) < 2:
            return None
        line_str, local = parts[0], parts[1]
        if not line_str or not local:
            return None
        try:
            line = int(line_str)
        except ValueError:
            return None
        if line == 0:
            return None
        if not local.isascii() or not local.islower() or not local.isalpha():
            return None
        context = parts[2] if len(parts) == 3 else None
        if context is not None:
            if not context or not context.isascii() or not context.islower() or not context.isalpha():
                return None
        return cls(line=line, local=local, context=context)

    def render(self) -> str:
        if self.context:
            return f"{self.line}:{self.local}:{self.context}"
        return f"{self.line}:{self.local}"


class ValidationResult(str, Enum):
    VALID = "valid"
    STALE = "stale"
    OUT_OF_RANGE = "out_of_range"


class ShiftResult:
    def __init__(
        self,
        *,
        found: Optional[int] = None,
        ambiguous: Optional[List[int]] = None,
    ) -> None:
        self.found = found
        self.ambiguous = ambiguous or []


class ChunkFingerprint:
    """Grok Candidate B — ``chunk_v1``."""

    name = "chunk_v1"

    def __init__(self, hash_len: int = DEFAULT_HASH_LEN, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if not 1 <= hash_len <= 4:
            raise ValueError("hash_len must be 1..=4")
        if chunk_size < 1:
            raise ValueError("chunk_size must be > 0")
        self.hash_len = hash_len
        self.chunk_size = chunk_size

    def _chunk_fp(self, lines: Sequence[str], line_idx: int) -> str:
        start = (line_idx // self.chunk_size) * self.chunk_size
        end = min(start + self.chunk_size, len(lines))
        combined = fnv1a_32(b"chunk")
        for line in lines[start:end]:
            lh = line_hash(line)
            combined ^= lh
            combined = (combined * 16_777_619) & 0xFFFFFFFF
        return encode_hash(combined, self.hash_len)

    def generate_anchors(self, lines: Sequence[str]) -> List[Anchor]:
        if not lines:
            return []
        n_chunks = (len(lines) + self.chunk_size - 1) // self.chunk_size
        chunk_fps: List[str] = []
        for ci in range(n_chunks):
            start = ci * self.chunk_size
            end = min(start + self.chunk_size, len(lines))
            combined = fnv1a_32(b"chunk")
            for line in lines[start:end]:
                lh = line_hash(line)
                combined ^= lh
                combined = (combined * 16_777_619) & 0xFFFFFFFF
            chunk_fps.append(encode_hash(combined, self.hash_len))
        out: List[Anchor] = []
        for i, line in enumerate(lines):
            out.append(
                Anchor(
                    line=i + 1,
                    local=encode_hash(line_hash(line), self.hash_len),
                    context=chunk_fps[i // self.chunk_size],
                )
            )
        return out

    def validate(self, anchor: ParsedAnchor, lines: Sequence[str]) -> ValidationResult:
        idx = anchor.line - 1
        if idx < 0 or idx >= len(lines):
            return ValidationResult.OUT_OF_RANGE
        expected_local = encode_hash(line_hash(lines[idx]), self.hash_len)
        if anchor.local != expected_local:
            return ValidationResult.STALE
        if anchor.context is None:
            return ValidationResult.STALE
        if anchor.context != self._chunk_fp(lines, idx):
            return ValidationResult.STALE
        return ValidationResult.VALID

    def find_shifted(
        self, anchor: ParsedAnchor, lines: Sequence[str], search_radius: int = DEFAULT_SEARCH_RADIUS
    ) -> ShiftResult:
        orig_idx = max(0, anchor.line - 1)
        start = max(0, orig_idx - search_radius)
        end = min(len(lines), orig_idx + search_radius + 1)
        candidates: List[int] = []
        for idx in range(start, end):
            if idx == orig_idx:
                continue
            local = encode_hash(line_hash(lines[idx]), self.hash_len)
            if local != anchor.local:
                continue
            if anchor.context is not None:
                probe = ParsedAnchor(line=idx + 1, local=local, context=anchor.context)
                if self.validate(probe, lines) != ValidationResult.VALID:
                    continue
            candidates.append(idx + 1)
        if not candidates:
            return ShiftResult()
        if len(candidates) == 1:
            return ShiftResult(found=candidates[0])
        return ShiftResult(ambiguous=candidates)


DEFAULT_SCHEME = ChunkFingerprint()


# ─── Format / apply ───────────────────────────────────────────────────────────

def format_hashline_content(
    file_content: str,
    *,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    scheme: ChunkFingerprint = DEFAULT_SCHEME,
) -> Tuple[str, str]:
    """Return (anchored_output, raw_window). offset is 1-based like Grok."""
    all_lines = split_lines(file_content)
    anchors = scheme.generate_anchors(all_lines)
    skip = max(0, (offset or 1) - 1)
    take = limit if limit is not None else len(all_lines)
    out_parts: List[str] = []
    raw_parts: List[str] = []
    for i in range(skip, min(len(all_lines), skip + take)):
        a = anchors[i]
        out_parts.append(f"{a.line}:{a.suffix()}{ARROW}{all_lines[i]}")
        raw_parts.append(all_lines[i])
    return "\n".join(out_parts), "\n".join(raw_parts)


def _detect_anchor_prefix(content: str) -> Optional[int]:
    for idx, line in enumerate(content.splitlines()):
        s = line.lstrip()
        for sep in (ARROW, "->"):
            if sep in s:
                before, _ = s.split(sep, 1)
                if len(before) <= 25 and ":" in before and " " not in before:
                    return idx + 1
    return None


@dataclass
class _ResolvedOp:
    original_idx: int
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive; insert when start == end
    new_lines: List[str]


@dataclass
class ApplyError:
    message: str
    kind: str = "error"
    requested_anchor: Optional[str] = None
    current: Optional[str] = None
    context: Optional[str] = None
    context_start_line: Optional[int] = None
    shifted_to: Optional[int] = None
    shifted_anchor: Optional[str] = None
    ambiguous_candidates: List[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"status": "error", "error": self.kind, "message": self.message}
        if self.requested_anchor is not None:
            d["requested_anchor"] = self.requested_anchor
        if self.current is not None:
            d["current"] = self.current
        if self.context is not None:
            d["context"] = self.context
        if self.context_start_line is not None:
            d["context_start_line"] = self.context_start_line
        if self.shifted_to is not None:
            d["shifted_to"] = self.shifted_to
        if self.shifted_anchor is not None:
            d["shifted_anchor"] = self.shifted_anchor
        if self.ambiguous_candidates:
            d["ambiguous_candidates"] = self.ambiguous_candidates
        return d


@dataclass
class ApplyOk:
    applied: int
    scheme: str
    snippet_start_line: int
    snippet: str
    path: str
    warnings: List[str] = field(default_factory=list)
    new_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": "ok",
            "applied": self.applied,
            "scheme": self.scheme,
            "snippet_start_line": self.snippet_start_line,
            "snippet": self.snippet,
            "path": self.path,
        }
        if self.warnings:
            d["warnings"] = self.warnings
        return d


def _strip_arrow(anchor_str: str) -> str:
    for sep in (ARROW, "->"):
        if sep in anchor_str:
            return anchor_str.split(sep, 1)[0]
    return anchor_str


def _recover_by_suffix(
    suffix: str, lines: Sequence[str], scheme: ChunkFingerprint
) -> Optional[ParsedAnchor]:
    anchors = scheme.generate_anchors(lines)
    matches = [a for a in anchors if a.suffix() == suffix]
    if len(matches) != 1:
        return None
    a = matches[0]
    return ParsedAnchor(line=a.line, local=a.local, context=a.context)


def _render_anchored(a: Anchor, content: str) -> str:
    return f"{a.line}:{a.suffix()}{ARROW}{content}"


def _sample_anchors(
    lines: Sequence[str], scheme: ChunkFingerprint, *, limit: int = 4
) -> list[str]:
    """Fresh ``LINE:HASH1:HASH2`` samples for malformed-anchor recovery hints."""
    if not lines:
        return []
    anchors = scheme.generate_anchors(lines)
    out: list[str] = []
    for a in anchors[: max(0, limit)]:
        out.append(a.render())
    return out


def _validate_anchor(
    anchor_str: str, lines: Sequence[str], scheme: ChunkFingerprint
) -> Tuple[Optional[int], Optional[ApplyError]]:
    cleaned = _strip_arrow(anchor_str)
    parsed = ParsedAnchor.parse(cleaned)
    if parsed is None:
        recovered = _recover_by_suffix(cleaned, lines, scheme)
        if recovered is None:
            samples = _sample_anchors(lines, scheme)
            hint = ""
            if samples:
                shown = ", ".join(f'"{s}"' for s in samples)
                hint = (
                    f" Valid anchors from this file (copy exactly, including the "
                    f"line number): {shown}. Prefer hashline_read / hashline_grep, "
                    f"then paste the ANCHOR before the arrow."
                )
            return None, ApplyError(
                kind="invalid_input",
                message=(
                    f'Malformed anchor: "{cleaned}". '
                    'Expected format: "LINE:HASH1:HASH2" (e.g. "22:abc:rst"). '
                    "Do not pass bare HASH1:HASH2 without the line number unless "
                    "it uniquely matches a suffix in the file."
                    f"{hint}"
                ),
                requested_anchor=cleaned,
                context="\n".join(
                    _render_anchored(a, lines[a.line - 1])
                    for a in scheme.generate_anchors(lines)[:4]
                    if 0 < a.line <= len(lines)
                )
                or None,
                context_start_line=1 if lines else None,
            )
        parsed = recovered

    result = scheme.validate(parsed, lines)
    if result == ValidationResult.VALID:
        return parsed.line - 1, None
    if result == ValidationResult.OUT_OF_RANGE:
        return None, ApplyError(
            kind="anchor_not_found",
            message=f"Line {parsed.line} is out of range (file has {len(lines)} lines).",
            requested_anchor=cleaned,
        )

    # Stale — recovery
    shift = scheme.find_shifted(parsed, lines)
    anchors = scheme.generate_anchors(lines)
    recovery_ctx = 5
    ctx_start = max(0, parsed.line - 1 - recovery_ctx)
    ctx_end = min(len(lines), parsed.line + recovery_ctx)
    context = "\n".join(
        _render_anchored(anchors[i], lines[i]) for i in range(ctx_start, ctx_end)
    )
    idx = parsed.line - 1
    current = _render_anchored(anchors[idx], lines[idx]) if 0 <= idx < len(lines) else None

    if shift.found is not None:
        fresh = f"{shift.found}:{anchors[shift.found - 1].suffix()}"
        return None, ApplyError(
            kind="anchor_stale",
            message=(
                f"Anchor stale at line {parsed.line}. Content appears to have shifted "
                f'to line {shift.found}. Retry with anchor "{fresh}".'
            ),
            requested_anchor=cleaned,
            current=current,
            context=context,
            context_start_line=ctx_start + 1,
            shifted_to=shift.found,
            shifted_anchor=fresh,
        )
    if shift.ambiguous:
        return None, ApplyError(
            kind="ambiguous_anchor",
            message=(
                f"Anchor stale at line {parsed.line}. Multiple candidates at lines "
                f"{shift.ambiguous}. Use the fresh anchors from the context below."
            ),
            requested_anchor=cleaned,
            current=current,
            context=context,
            context_start_line=ctx_start + 1,
            ambiguous_candidates=list(shift.ambiguous),
        )
    return None, ApplyError(
        kind="anchor_stale",
        message=(
            f"Anchor stale at line {parsed.line}. Use the fresh anchors from the "
            "context below to retry your edit."
        ),
        requested_anchor=cleaned,
        current=current,
        context=context,
        context_start_line=ctx_start + 1,
    )


def _check_overlaps(ops: Sequence[_ResolvedOp]) -> Optional[ApplyError]:
    ranges = [(op.start, op.end, op.original_idx) for op in ops if op.start != op.end]
    ranges.sort(key=lambda r: r[0])
    for i in range(len(ranges) - 1):
        if ranges[i][1] > ranges[i + 1][0]:
            a, b = ranges[i], ranges[i + 1]
            return ApplyError(
                kind="overlapping_edits",
                message=(
                    f"Overlapping edits: edit #{a[2] + 1} (lines {a[0] + 1}-{a[1]}) "
                    f"and edit #{b[2] + 1} (lines {b[0] + 1}-{b[1]})."
                ),
            )
    for op in ops:
        if op.start != op.end:
            continue
        insert_at = op.start
        for rs, re, r_idx in ranges:
            if rs <= insert_at < re:
                return ApplyError(
                    kind="overlapping_edits",
                    message=(
                        f"Overlapping edits: edit #{r_idx + 1} (lines {rs + 1}-{re}) "
                        f"and edit #{op.original_idx + 1} (insertion at line {insert_at + 1})."
                    ),
                )
    return None


def _build_snippet(
    new_content: str,
    edit_regions: List[Tuple[int, int]],
    scheme: ChunkFingerprint,
) -> Tuple[str, int]:
    if not edit_regions:
        return "", 1
    total = len(split_lines(new_content))
    global_start = max(0, edit_regions[0][0] - SNIPPET_CONTEXT)
    global_end = min(total, edit_regions[-1][1] + SNIPPET_CONTEXT)
    if global_end - global_start <= MAX_CONTIGUOUS_SNIPPET:
        snippet, _ = format_hashline_content(
            new_content, offset=global_start + 1, limit=global_end - global_start, scheme=scheme
        )
        return snippet, global_start + 1
    # Compact per-region
    parts: List[str] = []
    prev_end = 0
    merged: List[Tuple[int, int]] = []
    for start, end in edit_regions:
        ctx_start = max(0, start - SNIPPET_CONTEXT)
        ctx_end = min(total, end + SNIPPET_CONTEXT)
        if merged and ctx_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ctx_end))
        else:
            merged.append((ctx_start, ctx_end))
    for i, (start, end) in enumerate(merged):
        if i > 0:
            parts.append(f"... {start - prev_end} lines not shown ...")
        elif start > 0:
            parts.append(f"... {start} lines not shown ...")
        region, _ = format_hashline_content(
            new_content, offset=start + 1, limit=end - start, scheme=scheme
        )
        parts.append(region)
        prev_end = end
    if prev_end < total:
        parts.append(f"... {total - prev_end} lines not shown ...")
    return "\n".join(parts), merged[0][0] + 1


def _parse_ops(raw: Any) -> Tuple[Optional[List[dict[str, Any]]], Optional[str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"edits was a JSON string but could not be parsed: {exc}"
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        return None, "edits must be a non-empty array of edit operations"
    ops: List[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError as exc:
                return None, f"edit #{idx + 1} was a JSON string but could not be parsed: {exc}"
        if not isinstance(item, dict):
            return None, f"edit #{idx + 1} must be an object"
        ops.append(item)
    return ops, None


def _fresh_partial_anchor(
    anchor_str: Any,
    lines: Sequence[str],
    scheme: ChunkFingerprint,
) -> Optional[str]:
    """Return the full current anchor for an incomplete ``LINE:HASH`` input.

    This is only a recovery hint for anchors missing the context component;
    complete-but-stale anchors retain the stricter shifted/ambiguous workflow.
    """
    parsed = ParsedAnchor.parse(_strip_arrow(str(anchor_str or "")))
    if parsed is None or parsed.context is not None:
        return None
    idx = parsed.line - 1
    if idx < 0 or idx >= len(lines):
        return None
    return scheme.generate_anchors(lines)[idx].render()


def apply_edits(
    content: str,
    ops_raw: Any,
    *,
    path: str,
    scheme: ChunkFingerprint = DEFAULT_SCHEME,
) -> Tuple[Optional[str], dict[str, Any]]:
    """Validate + apply. Returns (new_content|None, result_dict)."""
    ops, err = _parse_ops(ops_raw)
    if err or ops is None:
        return None, ApplyError(kind="invalid_input", message=err or "invalid edits").to_dict()

    # Sole write op
    if len(ops) == 1 and ops[0].get("op") == "write":
        new_content = str(ops[0].get("content", ""))
        bad = _detect_anchor_prefix(new_content)
        if bad is not None:
            return None, ApplyError(
                kind="invalid_input",
                message=(
                    f"write content contains anchor prefixes copied from hashline_read "
                    f"(first offending line {bad}). Strip anchors and {ARROW} separators."
                ),
            ).to_dict()
        snippet, _ = format_hashline_content(
            new_content, offset=1, limit=min(SNIPPET_CONTEXT * 2, len(split_lines(new_content)) or 1),
            scheme=scheme,
        )
        ok = ApplyOk(
            applied=1,
            scheme=scheme.name,
            snippet_start_line=1,
            snippet=snippet,
            path=path,
            new_content=new_content,
        )
        return new_content, ok.to_dict()

    lines = split_lines(content)
    resolved: List[_ResolvedOp] = []

    for idx, op in enumerate(ops):
        kind = str(op.get("op") or "")
        try:
            if kind == "write":
                return None, ApplyError(
                    kind="invalid_input",
                    message=(
                        "Write op must be the only operation in a batch. "
                        "Either use write alone or use replace/insert_after without write."
                    ),
                ).to_dict()
            if kind == "replace":
                anchor = str(op.get("anchor") or "")
                end_anchor = op.get("end_anchor")
                body = str(op.get("content", ""))
                start, aerr = _validate_anchor(anchor, lines, scheme)
                if aerr:
                    if len(ops) > 1:
                        aerr.message = (
                            f"Edit {idx + 1}/{len(ops)} (replace): {aerr.message}\n\n"
                            f"This batch contained {len(ops)} edits. Because this anchor "
                            f"failed validation, none of the edits were applied. "
                            f"Retry all {len(ops)} edits with fresh anchors."
                        )
                    error = aerr.to_dict()
                    fresh_anchors: dict[str, str] = {}
                    fresh_start = _fresh_partial_anchor(anchor, lines, scheme)
                    if fresh_start:
                        fresh_anchors["anchor"] = fresh_start
                    if end_anchor:
                        fresh_end = _fresh_partial_anchor(end_anchor, lines, scheme)
                        if fresh_end:
                            fresh_anchors["end_anchor"] = fresh_end
                    if fresh_anchors:
                        error["fresh_anchors"] = fresh_anchors
                        rendered = json.dumps(fresh_anchors, separators=(",", ":"))
                        error["message"] += (
                            f"\nRetry with these exact anchors: {rendered}. "
                            "Do not shorten or reorder their hash components."
                        )
                    return None, error
                assert start is not None
                if end_anchor:
                    end_i, eerr = _validate_anchor(str(end_anchor), lines, scheme)
                    if eerr:
                        return None, eerr.to_dict()
                    assert end_i is not None
                    if end_i < start:
                        return None, ApplyError(
                            kind="invalid_input",
                            message=(
                                f"end_anchor line {end_i + 1} is before start "
                                f"anchor line {start + 1}."
                            ),
                            requested_anchor=str(end_anchor),
                        ).to_dict()
                    end = end_i + 1
                else:
                    end = start + 1
                bad = _detect_anchor_prefix(body)
                if bad is not None:
                    return None, ApplyError(
                        kind="invalid_input",
                        message=(
                            f"replace content contains anchor prefixes "
                            f"(first offending line {bad}). Strip them before retrying."
                        ),
                    ).to_dict()
                new_lines = [] if body == "" else body.splitlines()
                # Preserve intentional trailing newline as empty last line? Grok uses .lines()
                # which drops final empty — match that.
                resolved.append(_ResolvedOp(idx, start, end, new_lines))
            elif kind == "insert_after":
                anchor = str(op.get("anchor") or "")
                body = str(op.get("content", ""))
                if anchor == "0:":
                    insert_at = 0
                elif anchor == "EOF":
                    n = len(lines)
                    insert_at = (n - 1) if n > 1 and lines[-1] == "" else n
                else:
                    line_i, aerr = _validate_anchor(anchor, lines, scheme)
                    if aerr:
                        if len(ops) > 1:
                            aerr.message = (
                                f"Edit {idx + 1}/{len(ops)} (insert_after): {aerr.message}\n\n"
                                f"This batch contained {len(ops)} edits. Because this anchor "
                                f"failed validation, none of the edits were applied."
                            )
                        return None, aerr.to_dict()
                    assert line_i is not None
                    insert_at = line_i + 1
                bad = _detect_anchor_prefix(body)
                if bad is not None:
                    return None, ApplyError(
                        kind="invalid_input",
                        message=(
                            f"insert_after content contains anchor prefixes "
                            f"(first offending line {bad})."
                        ),
                    ).to_dict()
                new_lines = [""] if body == "" else body.splitlines()
                resolved.append(_ResolvedOp(idx, insert_at, insert_at, new_lines))
            else:
                return None, ApplyError(
                    kind="invalid_input",
                    message=f'Unknown op "{kind}". Use replace, insert_after, or write.',
                ).to_dict()
        except Exception as exc:  # pragma: no cover
            return None, ApplyError(kind="invalid_input", message=str(exc)).to_dict()

    overlap = _check_overlaps(resolved)
    if overlap:
        if len(ops) > 1:
            overlap.message += (
                f"\n\nThis batch contained {len(ops)} edits. Because of the overlap, "
                "none were applied."
            )
        return None, overlap.to_dict()

    warnings: List[str] = []
    for op in resolved:
        span = op.end - op.start
        if 6 <= span <= 20:
            warnings.append(f"Medium replacement range ({span} lines) at line {op.start + 1}.")
        elif span > 20:
            warnings.append(f"Large replacement range ({span} lines) at line {op.start + 1}.")

    resolved.sort(key=lambda o: (-o.start, -o.original_idx))
    result_lines = list(lines)
    edit_regions: List[Tuple[int, int]] = []
    cumulative_shift = 0
    for op in reversed(resolved):
        shifted_start = op.start + cumulative_shift
        inserted = len(op.new_lines)
        replaced = op.end - op.start
        edit_regions.append((shifted_start, shifted_start + inserted))
        cumulative_shift += inserted - replaced
    edit_regions.sort(key=lambda r: r[0])

    for op in resolved:
        result_lines[op.start : op.end] = op.new_lines

    new_content = "\n".join(result_lines)
    # If original ended with newline and we still have content, Grok join doesn't
    # re-add a trailing newline unless a trailing "" line remains.
    snippet, snippet_start = _build_snippet(new_content, edit_regions, scheme)
    ok = ApplyOk(
        applied=len(ops),
        scheme=scheme.name,
        snippet_start_line=snippet_start,
        snippet=snippet,
        path=path,
        warnings=warnings,
        new_content=new_content,
    )
    return new_content, ok.to_dict()


# ─── Tools ────────────────────────────────────────────────────────────────────

def inject_hashline_anchors(
    content: str,
    match_line_numbers: Sequence[int],
    *,
    context: int = 0,
    scheme: ChunkFingerprint = DEFAULT_SCHEME,
) -> str:
    """Render file lines with hashline anchors; highlight match lines.

    ``match_line_numbers`` are 1-based. When ``context`` > 0, include
    surrounding lines (like grep -C).
    """
    lines = split_lines(content)
    if not lines:
        return ""
    anchors = scheme.generate_anchors(lines)
    wanted: set[int] = set()
    for ln in match_line_numbers:
        for i in range(max(1, ln - context), min(len(lines), ln + context) + 1):
            wanted.add(i)
    if not wanted:
        return ""
    out: list[str] = []
    for ln in sorted(wanted):
        idx = ln - 1
        prefix = ":" if ln in match_line_numbers else "-"
        body = lines[idx]
        out.append(f"{anchors[idx].render()}{ARROW}{prefix}{body}")
    return "\n".join(out)


class HashlineReadTool:
    name = "hashline_read"
    cacheable = True
    keywords = ["hashline", "anchor read", "read with anchors"]
    description = (
        "Read a file with line-anchored output for use with hashline_edit. "
        f"Each line is ANCHOR{ARROW}CONTENT (e.g. 22:abc:rst{ARROW}  let x = 1;). "
        "Pass the ANCHOR (before the arrow) to hashline_edit. Prefer hashline_grep "
        "to find match sites with anchors, then hashline_edit. Anchors are valid "
        "only for the file state at read time — after any edit, use fresh anchors "
        "from hashline_edit or re-read."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "path": {"type": "string", "description": "Path to the file to read", "required": True},
        "offset": {
            "type": "number",
            "description": "1-based line to start from. Default: 1",
        },
        "limit": {
            "type": "number",
            "description": "Max lines to return. Default: 100",
        },
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        from clawagents.config.features import is_enabled

        if not is_enabled("hashline_tools"):
            return ToolResult(False, "", "hashline_tools feature disabled")
        sb = self._sb
        file_path = sb.safe_path(str(args.get("path", "")))
        try:
            offset = max(1, int(args.get("offset", 1)))
        except (TypeError, ValueError):
            offset = 1
        try:
            limit = max(1, int(args.get("limit", 100)))
        except (TypeError, ValueError):
            limit = 100
        try:
            if not await sb.exists(file_path):
                return ToolResult(False, "", f"hashline_read failed: File not found: {file_path}")
            content = await sb.read_file(file_path)
            anchored, _ = format_hashline_content(content, offset=offset, limit=limit)
            total = len(split_lines(content))
            header = (
                f"File: {file_path} ({total} lines, hashline chunk_v1, "
                f"showing from line {offset}, limit={limit})"
            )
            return ToolResult(True, header + "\n" + anchored)
        except Exception as exc:
            return ToolResult(False, "", f"hashline_read failed: {exc}")


_HASHLINE_GREP_MAX_FILE_BYTES = 1_048_576
_HASHLINE_GREP_MAX_FILES = 200
_HASHLINE_GREP_MAX_HEAD = 200
_HASHLINE_GREP_MAX_CONTEXT = 10
_HASHLINE_GREP_MAX_PATTERN_LEN = 512


class HashlineGrepTool:
    name = "hashline_grep"
    cacheable = True
    keywords = ["hashline", "search anchors", "grep anchors"]
    description = (
        "Search file contents and return matches with hashline anchors for "
        "hashline_edit. Workflow: hashline_grep → hashline_edit (prefer this over "
        "plain grep + edit_file for multi-hunk edits). Pattern is a Python regex."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "pattern": {"type": "string", "description": "Regex pattern to search", "required": True},
        "path": {"type": "string", "description": "File or directory to search", "required": True},
        "glob_filter": {
            "type": "string",
            "description": "Glob filter when path is a directory (e.g. '*.py')",
        },
        "recursive": {
            "type": "boolean",
            "description": "Search subdirectories when path is a directory. Default: true",
        },
        "context": {
            "type": "number",
            "description": "Context lines around each match (like grep -C). Default: 0",
        },
        "case_insensitive": {
            "type": "boolean",
            "description": "Case-insensitive match. Default: false",
        },
        "head_limit": {
            "type": "number",
            "description": "Max match lines to return across files. Default: 50",
        },
    }

    def __init__(self, sb: Any):
        self._sb = sb

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        import re

        from clawagents.config.features import is_enabled
        from clawagents.tools.filesystem import _walk_dir

        if not is_enabled("hashline_tools"):
            return ToolResult(False, "", "hashline_tools feature disabled")

        sb = self._sb
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return ToolResult(False, "", "hashline_grep failed: pattern required")
        if len(pattern) > _HASHLINE_GREP_MAX_PATTERN_LEN:
            return ToolResult(
                False,
                "",
                f"hashline_grep failed: pattern longer than {_HASHLINE_GREP_MAX_PATTERN_LEN} chars",
            )
        raw_path = str(args.get("path") or "").strip() or "."
        file_path = sb.safe_path(raw_path)
        glob_filter = str(args.get("glob_filter") or "*")
        recursive = self._truthy(args.get("recursive", True))
        try:
            ctx = max(0, min(_HASHLINE_GREP_MAX_CONTEXT, int(args.get("context", 0))))
        except (TypeError, ValueError):
            ctx = 0
        case_i = self._truthy(args.get("case_insensitive", False))
        try:
            head_limit = max(1, min(_HASHLINE_GREP_MAX_HEAD, int(args.get("head_limit", 50))))
        except (TypeError, ValueError):
            head_limit = 50

        flags = re.MULTILINE
        if case_i:
            flags |= re.IGNORECASE
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            return ToolResult(False, "", f"hashline_grep failed: invalid regex: {exc}")

        try:
            st = await sb.stat(file_path)
        except (FileNotFoundError, OSError):
            return ToolResult(False, "", f"hashline_grep failed: path not found: {file_path}")

        paths: List[str] = []
        truncated_files = False
        if st.is_file:
            paths = [file_path]
        elif st.is_directory:
            async for fp in _walk_dir(sb, file_path, glob_filter, recursive):
                paths.append(fp)
                if len(paths) >= _HASHLINE_GREP_MAX_FILES:
                    truncated_files = True
                    break
        else:
            return ToolResult(False, "", f"hashline_grep failed: not a file or directory: {file_path}")

        if not paths:
            return ToolResult(False, "", f"hashline_grep failed: no files under {file_path}")

        blocks: List[str] = []
        matches = 0
        skipped_binary = 0
        skipped_large = 0
        for path in paths:
            if matches >= head_limit:
                break
            try:
                content = await sb.read_file(path)
            except Exception:
                continue
            if "\x00" in content:
                skipped_binary += 1
                continue
            if len(content.encode("utf-8", errors="replace")) > _HASHLINE_GREP_MAX_FILE_BYTES:
                skipped_large += 1
                continue
            lines = split_lines(content)
            hit_lines: List[int] = []
            for i, line in enumerate(lines):
                try:
                    hit = rx.search(line) is not None
                except re.error:
                    return ToolResult(False, "", "hashline_grep failed: regex runtime error")
                if hit:
                    hit_lines.append(i + 1)
                    matches += 1
                    if matches >= head_limit:
                        break
            if not hit_lines:
                continue
            anchored = inject_hashline_anchors(content, hit_lines, context=ctx)
            blocks.append(f"File: {path}\n{anchored}")

        if not blocks:
            return ToolResult(True, f"No matches for {pattern!r} under {file_path}")
        notes: List[str] = [f"{matches} match line(s); use anchors with hashline_edit"]
        if truncated_files:
            notes.append(f"file walk capped at {_HASHLINE_GREP_MAX_FILES}")
        if skipped_binary:
            notes.append(f"skipped {skipped_binary} binary file(s)")
        if skipped_large:
            notes.append(f"skipped {skipped_large} oversized file(s)")
        footer = "\n\n(" + "; ".join(notes) + ")"
        return ToolResult(True, "\n\n".join(blocks) + footer)


class HashlineEditTool:
    name = "hashline_edit"
    keywords = ["hashline", "anchor edit", "replace by anchor"]
    description = (
        "Edit a file using anchors from hashline_read or hashline_grep. Ops: "
        'replace {op, anchor, end_anchor?, content}, '
        'insert_after {op, anchor|"0:"|"EOF", content}, '
        "write {op, content} (sole op). Batch is atomic — all validate or none apply. "
        "Anchors must be full LINE:HASH1:HASH2 (e.g. 22:abc:rst) copied from "
        "hashline_read/grep BEFORE the arrow — never invent short hashes. "
        "On success returns a fresh-anchor snippet; on stale/malformed anchors "
        "returns recovery context with valid samples. "
        "Prefer: hashline_grep → hashline_edit for multi-hunk work."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "path": {"type": "string", "description": "Path of the file to edit", "required": True},
        "edits": {
            "type": "array",
            "description": (
                "Array of edit objects. Pass objects directly, never JSON-encoded strings."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["replace", "insert_after", "write"],
                        "description": "Edit operation.",
                    },
                    "anchor": {
                        "type": "string",
                        "description": (
                            "Exact LINE:HASH1:HASH2 anchor copied before the arrow; "
                            "insert_after also accepts 0: or EOF."
                        ),
                    },
                    "end_anchor": {
                        "type": "string",
                        "description": "Optional inclusive end anchor for replace.",
                    },
                    "content": {"type": "string", "description": "Replacement text."},
                },
                "required": ["op", "content"],
                "additionalProperties": False,
            },
            "required": True,
        },
    }

    def __init__(self, sb: Any):
        self._sb = sb

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        from clawagents.config.features import is_enabled

        if not is_enabled("hashline_tools"):
            return ToolResult(False, "", "hashline_tools feature disabled")
        sb = self._sb
        file_path = sb.safe_path(str(args.get("path") or args.get("file_path") or ""))
        edits = args.get("edits")
        try:
            if not await sb.exists(file_path):
                # Allow write-only create
                ops, perr = _parse_ops(edits)
                if perr or not ops or not (len(ops) == 1 and ops[0].get("op") == "write"):
                    return ToolResult(False, "", f"hashline_edit failed: File not found: {file_path}")
                content = ""
            else:
                content = await sb.read_file(file_path)

            new_content, result = apply_edits(content, edits, path=file_path)
            if new_content is None:
                return ToolResult(False, json.dumps(result, indent=2), result.get("message", "edit failed"))
            parent = sb.dirname(file_path)
            if not await sb.exists(parent):
                await sb.mkdir(parent, recursive=True)
            await sb.write_file(file_path, new_content)
            return ToolResult(True, json.dumps(result, indent=2))
        except Exception as exc:
            return ToolResult(False, "", f"hashline_edit failed: {exc}")


def create_hashline_tools(backend: Any) -> List[Tool]:
    return [
        HashlineReadTool(backend),
        HashlineGrepTool(backend),
        HashlineEditTool(backend),
    ]


__all__ = [
    "HashlineReadTool",
    "HashlineGrepTool",
    "HashlineEditTool",
    "create_hashline_tools",
    "apply_edits",
    "format_hashline_content",
    "inject_hashline_anchors",
    "line_hash",
    "encode_hash",
    "ChunkFingerprint",
    "ParsedAnchor",
    "split_lines",
]
