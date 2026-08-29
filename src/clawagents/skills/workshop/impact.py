"""Append-only skill-impact ledger (WikiSkill skill-impact.md).

Written by the workshop harness, never by an LLM. Rollback must not delete it.
"""

from __future__ import annotations

import difflib
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

IMPACT_FILENAME = "skill-impact.md"
IMPACT_RELATIVE_PATH = f".clawagents/skill-workshop/{IMPACT_FILENAME}"
IMPACT_PREVIEW_ENTRIES = 8
MAX_DIFF_CHARS = 12_000

_LOCK = threading.Lock()
_ENTRY_HEAD = re.compile(r"(?m)^## \d{4}-\d{2}-\d{2}T")

_HEADER = (
    "# Skill impact ledger\n"
    "\n"
    "Append-only audit of skill_workshop apply / reject / quarantine / rollback.\n"
    "Never delete entries when rolling a skill back. Written by the workshop "
    "harness, not by an LLM.\n"
    "\n"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def unified_skill_diff(old: str, new: str, relpath: str = "SKILL.md") -> str:
    """Unified diff of one skill file. Empty string if the texts match."""
    old_lines = _diff_lines(old)
    new_lines = _diff_lines(new)
    if old_lines == new_lines:
        return ""
    rendered = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{relpath}",
            tofile=f"b/{relpath}",
            lineterm="\n",
        )
    )
    if len(rendered) > MAX_DIFF_CHARS:
        return rendered[:MAX_DIFF_CHARS].rstrip() + "\n... [diff truncated]\n"
    return rendered


def format_impact_entry(
    *,
    outcome: str,
    skill_name: str,
    action: str = "",
    proposal_id: str = "",
    rollback_id: str = "",
    reason: str = "",
    scan_findings: Iterable[str] | None = None,
    diff: str = "",
    when: str | None = None,
) -> str:
    stamp = when or utc_now_iso()
    label = outcome.upper().replace("_", " ")
    skill = skill_name or "(unknown)"
    lines = [
        f"## {stamp} · {label} · `{skill}`",
        "",
        f"- Outcome: `{outcome}`",
        f"- Target Skill: `{skill}`",
    ]
    if action:
        lines.append(f"- Action: `{action}`")
    if proposal_id:
        lines.append(f"- Proposal: `{proposal_id}`")
    if rollback_id:
        lines.append(f"- Rollback: `{rollback_id}`")
    lines.append(f"- Reason: {_one_line(reason) or '(none)'}")
    findings = [item for item in (scan_findings or []) if item]
    if findings:
        lines.append("- Scan findings:")
        lines.extend(f"  - {_one_line(item)}" for item in findings)
    else:
        lines.append("- Scan findings: (none)")
    lines.append("")
    lines.append("### Diff")
    lines.append("")
    body = (diff or "").rstrip()
    if not body:
        lines.append("(no SKILL.md change)")
        lines.append("")
    else:
        fence = "````" if "```" in body else "```"
        lines.append(f"{fence}diff")
        lines.append(body)
        lines.append(fence)
        lines.append("")
    return "\n".join(lines) + "\n"


def append_impact_entry(path: Path, markdown: str) -> None:
    """Append one formatted entry. Creates the file with a header if needed."""
    text = markdown if markdown.endswith("\n") else markdown + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size == 0:
            path.write_text(_HEADER + text, encoding="utf-8")
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def read_impact_text(path: Path, limit: Optional[int] = None) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if limit is None or limit <= 0:
        return text
    return _tail_entries(text, limit)


def _tail_entries(text: str, max_entries: int) -> str:
    starts = [match.start() for match in _ENTRY_HEAD.finditer(text)]
    if not starts:
        return text
    start = starts[max(0, len(starts) - max_entries)]
    return text[start:]


def _diff_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _one_line(value: str) -> str:
    return " ".join((value or "").split())
