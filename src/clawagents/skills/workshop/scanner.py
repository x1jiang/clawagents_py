"""Lightweight proposal scanner before apply."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from clawagents.skills.workshop.types import SUPPORT_FOLDERS

MAX_SKILL_BYTES = 40_000
MAX_DESCRIPTION_BYTES = 160
MAX_SUPPORT_FILES = 64
MAX_SUPPORT_FILE_BYTES = 256 * 1024

_SUSPICIOUS = re.compile(
    r"(rm\s+-rf|curl\s+.*\|\s*(ba)?sh|eval\s*\(|exec\s*\(|__import__|subprocess\.|os\.system)",
    re.IGNORECASE,
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def support_path_findings(path: str, support_root: Path | None = None) -> list[str]:
    """Validate a support path before it is joined to a writable root."""
    if not path or "\x00" in path:
        return [f"invalid support path: {path}"]

    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    windows = PureWindowsPath(path)
    parts = pure.parts
    findings: list[str] = []
    if pure.is_absolute() or windows.is_absolute() or bool(windows.drive):
        findings.append(f"invalid absolute support path: {path}")
    if len(parts) < 2 or parts[0] not in SUPPORT_FOLDERS:
        findings.append(f"support file must live under standard folders: {path}")
    if any(part in {".", "..", ""} for part in parts):
        findings.append(f"invalid support path traversal: {path}")

    if support_root is not None and not findings:
        root = support_root.resolve(strict=False)
        parent = support_root.parent.resolve(strict=False)
        destination = support_root.joinpath(*parts).resolve(strict=False)
        if not _is_within(root, parent) or not _is_within(destination, root):
            findings.append(f"support path escapes proposal support root: {path}")
    return findings


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
        findings.extend(support_path_findings(path))
    if _SUSPICIOUS.search(body):
        findings.append("suspicious pattern in proposal body")
    return findings
