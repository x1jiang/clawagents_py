"""Pre-compaction memory flush — extract durable facts before context is lost.

Grok Build parity (memory_flush.rs): flush near compact threshold, exact
blake2 dedup, optional semantic abstain with NO_REPLY.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


LLMComplete = Callable[[str], Awaitable[str]]


@dataclass
class FlushConfig:
    enabled: bool = True
    soft_threshold_tokens: int = 4_000
    compact_pct: float = 0.85
    max_flush_write_chars: int = 8_000
    recent_messages: int = 24


@dataclass
class FlushOutcome:
    status: str  # skipped | nothing | accepted | rejected | error
    detail: str = ""
    stored: bool = False


_LAST_FLUSH_CYCLE: dict[str, int] = {}


def should_flush(
    total_tokens: int,
    context_window: int,
    *,
    compaction_cycle: int = 0,
    workspace: str | Path | None = None,
    config: FlushConfig | None = None,
) -> bool:
    from clawagents.config.features import is_enabled

    if not is_enabled("memory_flush"):
        return False
    cfg = config or FlushConfig()
    if not cfg.enabled or context_window <= 0:
        return False
    key = str(Path(workspace or os.getcwd()).resolve())
    # Skip if we already flushed for this compaction cycle (incl. cycle 0).
    if _LAST_FLUSH_CYCLE.get(key) == compaction_cycle:
        return False
    threshold = int(context_window * cfg.compact_pct) - cfg.soft_threshold_tokens
    return total_tokens >= max(1, threshold)


def select_flush_window(messages: list[Any], recent_n: int = 24) -> list[Any]:
    """Drop system; take last N; expand left to a user boundary."""
    non_system = [m for m in messages if getattr(m, "role", None) != "system"]
    if not non_system:
        return []
    window = non_system[-max(1, recent_n) :]
    # Expand left to include leading user turn if truncated mid-exchange
    start = len(non_system) - len(window)
    while start > 0 and getattr(non_system[start], "role", None) != "user":
        start -= 1
        window = non_system[start:]
    return window


def _format_window(messages: list[Any]) -> str:
    lines: list[str] = []
    for m in messages:
        role = getattr(m, "role", "?")
        content = getattr(m, "content", "")
        if isinstance(content, list):
            texts = [
                str(b.get("text", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(t for t in texts if t)
        text = str(content or "").strip()
        if not text:
            continue
        lines.append(f"[{role}]\n{text[:2000]}")
    return "\n\n".join(lines)


def process_flush_response(response: str, *, max_chars: int = 8_000) -> str | None:
    text = (response or "").strip()
    if not text or text.upper() == "NO_REPLY":
        return None
    if "##" not in text and "- " not in text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


async def run_memory_flush(
    messages: list[Any],
    llm_complete: LLMComplete,
    *,
    workspace: str | Path | None = None,
    compaction_cycle: int = 0,
    config: FlushConfig | None = None,
) -> FlushOutcome:
    from clawagents.config.features import is_enabled

    if not is_enabled("memory_flush"):
        return FlushOutcome(status="skipped", detail="feature_disabled")

    cfg = config or FlushConfig()
    ws = Path(workspace or os.getcwd()).resolve()
    window = select_flush_window(messages, cfg.recent_messages)
    if not window:
        return FlushOutcome(status="nothing", detail="empty_window")

    prompt = (
        "Extract durable project memories from this recent conversation window.\n"
        "Write markdown with ## headers (facts, decisions, open questions).\n"
        "Only include NEW durable knowledge. If nothing new, reply NO_REPLY.\n\n"
        + _format_window(window)
    )
    try:
        raw = await llm_complete(prompt)
    except Exception as exc:  # noqa: BLE001
        return FlushOutcome(status="error", detail=str(exc))

    body = process_flush_response(raw, max_chars=cfg.max_flush_write_chars)
    if not body:
        return FlushOutcome(status="nothing", detail="NO_REPLY")

    # Exact dedup via smart store
    try:
        from clawagents.memory.smart_store import SmartMemoryStore, MemoryChunk, ingest_text
        import uuid

        store = SmartMemoryStore(ws)
        try:
            if store.is_duplicate_exact(body):
                return FlushOutcome(status="rejected", detail="exact_duplicate")
        finally:
            store.close()

        ok = ingest_text(
            body,
            path=f"flush/{int(time.time())}.md",
            source="session",
            workspace=ws,
            chunk_id=f"flush_{uuid.uuid4().hex[:10]}",
        )
        # Daily log
        day = time.strftime("%Y-%m-%d")
        log_dir = ws / ".clawagents" / "memory-sessions"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"flush_{day}.md"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n## Flush {time.strftime('%H:%M:%S')}\n{body}\n")

        _LAST_FLUSH_CYCLE[str(ws)] = compaction_cycle
        return FlushOutcome(
            status="accepted" if ok else "rejected",
            detail="stored" if ok else "duplicate",
            stored=ok,
        )
    except Exception as exc:  # noqa: BLE001
        return FlushOutcome(status="error", detail=str(exc))


__all__ = [
    "FlushConfig",
    "FlushOutcome",
    "should_flush",
    "select_flush_window",
    "process_flush_response",
    "run_memory_flush",
]
