"""Letta-inspired editable core memory blocks + optional memory-bank files."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_CORE_BUDGET = 2_400  # chars ≈ 600–800 tokens
_BLOCK_SPLIT = re.compile(r"(?m)^##\s+(\w+)\s*$")


def core_memory_path(workspace: str | Path | None = None) -> Path:
    root = Path(workspace or Path.cwd()) / ".clawagents"
    root.mkdir(parents=True, exist_ok=True)
    return root / "core-memory.md"


def memory_bank_dir(workspace: str | Path | None = None) -> Path:
    d = Path(workspace or Path.cwd()) / ".clawagents" / "memory-bank"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_core_memory(workspace: str | Path | None = None) -> Path:
    path = core_memory_path(workspace)
    if not path.is_file():
        path.write_text(
            "# Core Memory\n\n"
            "## persona\nHelpful coding agent. Prefer surgical edits and tests.\n\n"
            "## human\n(unknown)\n\n"
            "## project\n(fill with durable project facts)\n",
            encoding="utf-8",
        )
    return path


def load_core_memory(
    *,
    workspace: str | Path | None = None,
    max_chars: int = DEFAULT_CORE_BUDGET,
) -> str:
    path = ensure_core_memory(workspace)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 30] + "\n… [core memory truncated] …"
    return f"## Core Memory\n\n{text}\n"


def _parse_blocks(text: str) -> dict[str, str]:
    parts = _BLOCK_SPLIT.split(text)
    # parts: [preamble, label1, body1, label2, body2, ...]
    blocks: dict[str, str] = {}
    if len(parts) < 3:
        blocks["project"] = text.strip()
        return blocks
    for i in range(1, len(parts), 2):
        label = parts[i].strip().lower()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        blocks[label] = body
    return blocks


def _serialize_blocks(blocks: dict[str, str]) -> str:
    lines = ["# Core Memory", ""]
    for label, body in blocks.items():
        lines.append(f"## {label}")
        lines.append(body.strip() or "(empty)")
        lines.append("")
    return "\n".join(lines)


def core_memory_replace(
    label: str,
    old_str: str,
    new_str: str,
    *,
    workspace: str | Path | None = None,
    max_chars: int = DEFAULT_CORE_BUDGET,
) -> tuple[bool, str]:
    path = ensure_core_memory(workspace)
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _parse_blocks(text)
    key = (label or "project").strip().lower()
    body = blocks.get(key, "")
    if old_str not in body:
        return False, f"old_str not found in block '{key}'"
    body2 = body.replace(old_str, new_str, 1)
    blocks[key] = body2
    out = _serialize_blocks(blocks)
    if len(out) > max_chars * 2:
        return False, f"core memory would exceed budget ({len(out)} chars)"
    path.write_text(out, encoding="utf-8")
    return True, f"updated block '{key}' ({len(out)} chars)"


def core_memory_append(
    label: str,
    content: str,
    *,
    workspace: str | Path | None = None,
    max_chars: int = DEFAULT_CORE_BUDGET,
) -> tuple[bool, str]:
    path = ensure_core_memory(workspace)
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _parse_blocks(text)
    key = (label or "project").strip().lower()
    prev = blocks.get(key, "").rstrip()
    blocks[key] = (prev + "\n" + content.strip()).strip()
    out = _serialize_blocks(blocks)
    if len(out) > max_chars * 2:
        return False, f"core memory would exceed budget ({len(out)} chars)"
    path.write_text(out, encoding="utf-8")
    return True, f"appended to block '{key}'"


def load_memory_bank_preamble(
    *,
    workspace: str | Path | None = None,
    max_chars: int = 4_000,
) -> str:
    d = memory_bank_dir(workspace)
    preferred = ["product.md", "tech.md", "progress.md", "decisions.md"]
    parts: list[str] = []
    used = 0
    for name in preferred:
        p = d / name
        if not p.is_file():
            continue
        body = p.read_text(encoding="utf-8", errors="replace").strip()
        if not body:
            continue
        chunk = f"### {name}\n{body}\n"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    if not parts:
        return ""
    return "## Memory Bank\n\n" + "\n".join(parts)


def ensure_memory_bank_stubs(workspace: str | Path | None = None) -> None:
    d = memory_bank_dir(workspace)
    stubs = {
        "product.md": "# Product\n\nWhat we are building and for whom.\n",
        "tech.md": "# Tech\n\nStack, constraints, conventions.\n",
        "progress.md": "# Progress\n\nCurrent focus and open work.\n",
        "decisions.md": "# Decisions\n\nDurable architectural decisions.\n",
    }
    for name, content in stubs.items():
        p = d / name
        if not p.is_file():
            p.write_text(content, encoding="utf-8")
