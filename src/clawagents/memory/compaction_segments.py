"""Greppable compaction segments — segment_NNN.md + INDEX.md.

Grok Build parity (xai-chat-state compaction segments): compaction stops being
purely lossy — the model can grep segment files to recover detail.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


INDEX_HEADER = (
    "| Segment | File | Turns | Approx bytes | Keywords |\n"
    "|---------|------|-------|--------------|----------|\n"
)


@dataclass
class CompactionSegment:
    index: int
    turns: int
    content: str
    keywords: list[str] = field(default_factory=list)
    approx_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.approx_bytes:
            self.approx_bytes = len(self.content.encode("utf-8"))


def segment_filename(index: int) -> str:
    return f"segment_{int(index):03d}.md"


def parse_segment_index(name: str) -> int | None:
    m = re.match(r"segment_(\d+)\.md$", name or "")
    return int(m.group(1)) if m else None


def extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", text or "")
    freq: dict[str, int] = {}
    for t in tokens:
        low = t.lower()
        if low in {"the", "and", "for", "with", "this", "that", "from", "are"}:
            continue
        freq[low] = freq.get(low, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:limit]]


def render_segment_md(segment: CompactionSegment) -> str:
    kws = ", ".join(segment.keywords) if segment.keywords else "(none)"
    return (
        f"# Compaction segment {segment.index:03d}\n\n"
        f"- turns: {segment.turns}\n"
        f"- bytes: {segment.approx_bytes}\n"
        f"- keywords: {kws}\n\n"
        f"{segment.content.rstrip()}\n"
    )


def render_index_row(segment: CompactionSegment) -> str:
    fname = segment_filename(segment.index)
    kws = ", ".join(segment.keywords[:6])
    return (
        f"| {segment.index:03d} | `{fname}` | {segment.turns} | "
        f"{segment.approx_bytes} | {kws} |\n"
    )


def compaction_dir(workspace: Path) -> Path:
    d = workspace / ".clawagents" / "compaction"
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_segment_index(directory: Path) -> int:
    best = 0
    for p in directory.glob("segment_*.md"):
        idx = parse_segment_index(p.name)
        if idx is not None:
            best = max(best, idx)
    return best + 1


def write_segment(
    content: str,
    *,
    workspace: str | Path,
    turns: int = 1,
    index: int | None = None,
) -> CompactionSegment:
    from clawagents.config.features import is_enabled

    ws = Path(workspace).resolve()
    directory = compaction_dir(ws)
    idx = index if index is not None else next_segment_index(directory)
    seg = CompactionSegment(
        index=idx,
        turns=turns,
        content=content,
        keywords=extract_keywords(content),
    )
    if not is_enabled("compaction_segments"):
        return seg
    path = directory / segment_filename(idx)
    path.write_text(render_segment_md(seg), encoding="utf-8")
    _upsert_index(directory, seg)
    return seg


def _upsert_index(directory: Path, segment: CompactionSegment) -> None:
    index_path = directory / "INDEX.md"
    row = render_index_row(segment)
    if not index_path.is_file():
        index_path.write_text(
            "# Compaction segments\n\n"
            "Greppable archives of compacted turns. Use grep/read on "
            "`segment_*.md` + this INDEX to recover detail.\n\n"
            + INDEX_HEADER
            + row,
            encoding="utf-8",
        )
        return
    text = index_path.read_text(encoding="utf-8")
    # Replace existing row for same index or append
    pat = re.compile(rf"^\| {segment.index:03d} \|.*\n", re.M)
    if pat.search(text):
        text = pat.sub(row, text, count=1)
    else:
        if INDEX_HEADER not in text:
            text = text.rstrip() + "\n\n" + INDEX_HEADER
        text = text.rstrip() + "\n" + row
    index_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def segment_recovery_hint() -> str:
    return (
        "Compacted history is archived under `.clawagents/compaction/` "
        "(`segment_NNN.md` + `INDEX.md`). Grep those files to recover detail; "
        "do not rewrite segment archives."
    )


# ── History-then-steps mode helpers ──────────────────────────────────────


@dataclass
class CompactionFailureKind:
    kind: str  # deterministic | transient
    detail: str = ""


def classify_compaction_failure(exc: BaseException | str) -> CompactionFailureKind:
    """Map compaction LLM errors to deterministic vs transient."""
    msg = str(exc).lower()
    # Deterministic: 4xx (≠408,≠429), context length, invalid request
    if any(
        s in msg
        for s in (
            "context length",
            "context_length",
            "maximum context",
            "invalid_request",
            "invalid request",
            "400",
            "401",
            "403",
            "404",
            "422",
        )
    ) and not any(s in msg for s in ("408", "429", "rate")):
        return CompactionFailureKind(kind="deterministic", detail=msg[:200])
    if any(s in msg for s in ("timeout", "429", "408", "500", "502", "503", "529", "empty response")):
        return CompactionFailureKind(kind="transient", detail=msg[:200])
    return CompactionFailureKind(kind="transient", detail=msg[:200])


def should_compact_steps_after_history(
    history_tokens: int,
    steps_tokens: int,
    *,
    steps_trigger_ratio: float = 0.30,
) -> bool:
    """HistoryThenSteps: compact recent tool steps only if they dominate."""
    if history_tokens <= 0:
        return steps_tokens > 0
    return steps_tokens > int(history_tokens * steps_trigger_ratio)


def separate_prior_user_queries(text: str) -> tuple[str, str]:
    """Split prior `<grok_user_queries>` / `<user_query>` blocks from body."""
    if not text:
        return "", ""
    prior_parts: list[str] = []
    body = text
    for tag in ("grok_user_queries", "user_query"):
        pat = re.compile(rf"<{tag}>(.*?)</{tag}>", re.S | re.I)
        for m in pat.finditer(body):
            prior_parts.append(m.group(1).strip())
        body = pat.sub("", body)
    return "\n\n".join(p for p in prior_parts if p), body.strip()


def wrap_user_query(text: str) -> str:
    return f"<user_query>\n{(text or '').strip()}\n</user_query>"


__all__ = [
    "CompactionSegment",
    "INDEX_HEADER",
    "segment_filename",
    "parse_segment_index",
    "extract_keywords",
    "render_segment_md",
    "render_index_row",
    "write_segment",
    "segment_recovery_hint",
    "CompactionFailureKind",
    "classify_compaction_failure",
    "should_compact_steps_after_history",
    "separate_prior_user_queries",
    "wrap_user_query",
]
