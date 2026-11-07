"""Lightweight proposal scanner before apply."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from clawagents.skills.workshop.types import SUPPORT_FOLDERS
from clawagents.trajectory.harness_surface import (
    GRADER_VERDICT_MARKERS,
    check_python_compiles,
    check_script_references,
    parse_frontmatter_fields,
)

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


def scan_proposal_content(
    name: str,
    description: str,
    body: str,
    support_files: list[tuple[str, str]],
    *,
    existing_support_paths: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    findings: list[str] = []
    if not name or not re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", name):
        findings.append("skill name must be lowercase alphanumeric with hyphens/underscores")
    if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        findings.append(f"description exceeds {MAX_DESCRIPTION_BYTES} bytes")
    if len(body.encode("utf-8")) > MAX_SKILL_BYTES:
        findings.append(f"proposal body exceeds {MAX_SKILL_BYTES} bytes")
    if len(support_files) > MAX_SUPPORT_FILES:
        findings.append(f"too many support files (max {MAX_SUPPORT_FILES})")

    # Harness² frontmatter rule: if YAML frontmatter defines `name:`, it must match
    # the skill directory name, otherwise it overrides the directory name at load
    # and breaks trigger lines.
    fm = parse_frontmatter_fields(body)
    fm_name = fm.get("name")
    if fm_name and name and fm_name != name:
        findings.append(
            f"frontmatter name '{fm_name}' does not match skill name '{name}' "
            "(overrides directory name at load)"
        )

    available_scripts: set[str] = set(existing_support_paths)
    for path, content in support_files:
        if _SUSPICIOUS.search(content):
            findings.append(f"suspicious pattern in support file {path}")
        if len(content.encode("utf-8")) > MAX_SUPPORT_FILE_BYTES:
            findings.append(f"support file too large: {path}")
        path_errs = support_path_findings(path)
        findings.extend(path_errs)
        normalized_rel = path.replace("\\", "/").strip("/")
        available_scripts.add(normalized_rel)
        if not path_errs and normalized_rel.endswith(".py"):
            findings.extend(check_python_compiles(normalized_rel, content))
        for marker in GRADER_VERDICT_MARKERS:
            if marker in content:
                findings.append(f"grader verdict marker {marker!r} in support file {path}")

    # Harness² script-reference rule: every `python scripts/<file>.py` invoked by
    # SKILL.md must ship with the skill (or already exist in the target skill).
    findings.extend(
        check_script_references(
            f"skills/{name}/SKILL.md",
            body,
            available_scripts,
            skill_name=name,
        )
    )

    if _SUSPICIOUS.search(body):
        findings.append("suspicious pattern in proposal body")
    for marker in GRADER_VERDICT_MARKERS:
        if marker in body:
            findings.append(f"grader verdict marker {marker!r} in proposal body")
    return findings
