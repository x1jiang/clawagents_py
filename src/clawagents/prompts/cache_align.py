"""Stabilize static prompt prefixes so provider KV caches hit more often."""

from __future__ import annotations

import re

_MULTI_NL = re.compile(r"\n{3,}")
_TRAIL_WS = re.compile(r"[ \t]+\n")


def normalize_stable_prefix(text: str) -> str:
    """Canonical whitespace for the cacheable system-prompt prefix.

    Does not reorder semantic sections — only collapses noisy whitespace so
    identical logical prompts hash the same across runs.
    """
    if not text:
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _TRAIL_WS.sub("\n", out)
    out = _MULTI_NL.sub("\n\n", out)
    return out.strip() + ("\n" if out.strip() else "")


def sort_tool_names(names: list[str]) -> list[str]:
    """Stable alphabetical tool order for schema / description blocks."""
    return sorted(names, key=lambda n: n.lower())
