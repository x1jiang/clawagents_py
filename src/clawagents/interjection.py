"""Mid-turn interjection formatting + pending buffer (Grok interjection-core).

- Each pending entry drains as its **own** synthetic user message (never merged).
- Format matches Grok: mid-turn note + ``<user_query>`` envelope.
- Stranded entries (run ended / cancelled before drain) become queued prompts.
"""

from __future__ import annotations

import threading
from typing import Any

LARGE_PROMPT_THRESHOLD = 25_000

_META_KEY = "pending_interjects"  # list[str]
_LEGACY_KEY = "pending_interject"  # str (compat)

_BUF_LOCK = threading.Lock()


def user_query(user_message: str) -> str:
    return f"<user_query>\n{user_message}\n</user_query>"


def format_interjection(text: str) -> str:
    """Wrap interjection text as a standalone synthetic user turn."""
    body = (text or "").strip()
    if len(body) > LARGE_PROMPT_THRESHOLD:
        # Truncate on a UTF-8-safe character boundary (Grok parity).
        end = LARGE_PROMPT_THRESHOLD
        while end > 0 and (ord(body[end - 1]) & 0xC0) == 0x80:
            end -= 1
        # Prefer slicing by unicode chars
        cut = body[:LARGE_PROMPT_THRESHOLD]
        body = f"{cut}... [truncated]"
    return (
        "The user sent a message while you were working:\n"
        f"{user_query(body)}"
    )


def _meta(run_context: Any) -> dict | None:
    if run_context is None:
        return None
    meta = getattr(run_context, "_metadata", None)
    return meta if isinstance(meta, dict) else None


def enqueue_interject(run_context: Any, text: str) -> bool:
    """Append one interjection entry (never merge into prior text).

    Thread-safe for host threads enqueueing while the agent loop drains.
    """
    msg = (text or "").strip()
    if not msg:
        return False
    meta = _meta(run_context)
    if meta is None:
        return False
    with _BUF_LOCK:
        # Migrate legacy single-string key once
        legacy = meta.pop(_LEGACY_KEY, None)
        buf = meta.get(_META_KEY)
        if not isinstance(buf, list):
            buf = []
            meta[_META_KEY] = buf
        if isinstance(legacy, str) and legacy.strip():
            buf.append(legacy.strip())
        buf.append(msg)
        return True


def drain_interjects(run_context: Any) -> list[str]:
    """Pop all pending entries and return **formatted** synthetic user texts.

    One formatted string per entry (FIFO). Empty if nothing pending.
    """
    meta = _meta(run_context)
    if meta is None:
        return []
    with _BUF_LOCK:
        legacy = meta.pop(_LEGACY_KEY, None)
        raw = meta.pop(_META_KEY, None)
    entries: list[str] = []
    if isinstance(raw, list):
        entries.extend(str(x).strip() for x in raw if str(x).strip())
    elif isinstance(raw, str) and raw.strip():
        entries.append(raw.strip())
    if isinstance(legacy, str) and legacy.strip():
        entries.insert(0, legacy.strip())
    return [format_interjection(t) for t in entries]


def take_stranded_interjects(run_context: Any) -> list[str]:
    """Pop pending **raw** texts for promotion to queued prompts (no format)."""
    meta = _meta(run_context)
    if meta is None:
        return []
    with _BUF_LOCK:
        legacy = meta.pop(_LEGACY_KEY, None)
        raw = meta.pop(_META_KEY, None)
    entries: list[str] = []
    if isinstance(raw, list):
        entries.extend(str(x).strip() for x in raw if str(x).strip())
    elif isinstance(raw, str) and raw.strip():
        entries.append(raw.strip())
    if isinstance(legacy, str) and legacy.strip():
        entries.insert(0, legacy.strip())
    return entries


def peek_interject_count(run_context: Any) -> int:
    meta = _meta(run_context)
    if meta is None:
        return 0
    n = 0
    raw = meta.get(_META_KEY)
    if isinstance(raw, list):
        n += sum(1 for x in raw if str(x).strip())
    elif isinstance(raw, str) and raw.strip():
        n += 1
    legacy = meta.get(_LEGACY_KEY)
    if isinstance(legacy, str) and legacy.strip():
        n += 1
    return n


__all__ = [
    "LARGE_PROMPT_THRESHOLD",
    "user_query",
    "format_interjection",
    "enqueue_interject",
    "drain_interjects",
    "take_stranded_interjects",
    "peek_interject_count",
]
