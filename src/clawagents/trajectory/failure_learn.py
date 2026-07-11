"""Append durable failure lessons into AGENTS.md (local Headroom-learn style)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_BULLET = re.compile(r"^\s*[-*]\s+(.+)$", re.M)


def extract_lesson_bullets(lessons_text: str, *, limit: int = 8) -> list[str]:
    bullets = [m.group(1).strip() for m in _BULLET.finditer(lessons_text or "")]
    # Deduplicate case-insensitively while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for b in bullets:
        key = b.lower()
        if len(b) < 12 or key in seen:
            continue
        seen.add(key)
        out.append(b)
        if len(out) >= limit:
            break
    return out


def append_failure_lessons_to_agents_md(
    lessons_text: str,
    *,
    workspace: str | Path | None = None,
    filename: str = "AGENTS.md",
    min_bullets: int = 1,
) -> list[str]:
    """Append new lesson bullets under a dated ``## Failure lessons`` section.

    Returns the bullets that were newly written (empty if nothing changed).
    """
    bullets = extract_lesson_bullets(lessons_text)
    if len(bullets) < min_bullets:
        return []

    root = Path(workspace or Path.cwd())
    path = root / filename
    existing = ""
    if path.is_file():
        existing = path.read_text(encoding="utf-8", errors="replace")

    existing_lower = existing.lower()
    fresh = [b for b in bullets if b.lower() not in existing_lower]
    if not fresh:
        return []

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    block_lines = [f"\n## Failure lessons ({ts})\n"]
    for b in fresh:
        block_lines.append(f"- {b}")
    block_lines.append("")
    block = "\n".join(block_lines)

    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + block, encoding="utf-8")
    return fresh
