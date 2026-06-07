"""Shared search snippet formatting for session and history search."""

from __future__ import annotations


def snippet_from_content(content: str, query: str, width: int = 80) -> str:
    """Return a short excerpt around the first case-insensitive query match."""
    lower = content.lower()
    q = query.lower()
    idx = lower.find(q)
    if idx < 0:
        return content[:width]
    start = max(0, idx - 24)
    end = min(len(content), idx + len(query) + 24)
    out = content[start:end]
    if start > 0:
        out = "…" + out
    if end < len(content):
        out = out + "…"
    return out.replace(query, f"[{query}]", 1) if query in out else out
