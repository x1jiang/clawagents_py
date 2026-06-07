"""Lightweight proposal scanner before apply."""

from __future__ import annotations

import re
from pathlib import Path

MAX_SKILL_BYTES = 40_000
MAX_DESCRIPTION_BYTES = 160
MAX_SUPPORT_FILES = 64
MAX_SUPPORT_FILE_BYTES = 256 * 1024

_SUSPICIOUS = re.compile(
    r"(rm\s+-rf|curl\s+.*\|\s*(ba)?sh|eval\s*\(|exec\s*\(|__import__|subprocess\.|os\.system)",
    re.IGNORECASE,
)


def scan_proposal_content(name: str, description: str, body: str, support_files: list[tuple[str, str]]) -> list[str]:
    findings: list[str] = []
    if not name or not re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", name):
        findings.append("skill name must be lowercase alphanumeric with hyphens/underscores")
    if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        findings.append(f"description exceeds {MAX_DESCRIPTION_BYTES} bytes")
    if len(body.encode("utf-8")) > MAX_SKILL_BYTES:
        findings.append(f"proposal body exceeds {MAX_SKILL_BYTES} bytes")
    if len(support_files) > MAX_SUPPORT_FILES:
        findings.append(f"too many support files (max {MAX_SUPPORT_FILES})")
    for path, content in support_files:
        if _SUSPICIOUS.search(content):
            findings.append(f"suspicious pattern in support file {path}")
        if len(content.encode("utf-8")) > MAX_SUPPORT_FILE_BYTES:
            findings.append(f"support file too large: {path}")
        parts = Path(path).parts
        if parts[0] not in {"assets", "examples", "references", "scripts", "templates"}:
            findings.append(f"support file must live under standard folders: {path}")
        if ".." in parts or path.startswith("/"):
            findings.append(f"invalid support path: {path}")
    if _SUSPICIOUS.search(body):
        findings.append("suspicious pattern in proposal body")
    return findings
