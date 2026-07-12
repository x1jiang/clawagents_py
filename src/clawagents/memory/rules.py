"""Always-on project rules discovery (Cline .clinerules-inspired)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Union

# Files / dirs searched relative to workspace cwd.
_RULE_ROOT_FILES = (
    "AGENTS.md",
    "CLAWAGENTS.md",
    "CLAUDE.md",
)
_RULE_NESTED_FILES = (
    ".clawagents/instructions.md",
)
_RULES_DIR = ".clawagents/rules"

DEFAULT_RULES_MAX_CHARS = 12_000


def discover_rule_paths(workspace: str | Path | None = None) -> List[Path]:
    """Return rule file paths in stable order (deduped)."""
    root = Path(workspace or os.getcwd()).resolve()
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except OSError:
            return
        if rp in seen or not rp.is_file():
            return
        seen.add(rp)
        found.append(rp)

    for name in _RULE_ROOT_FILES:
        _add(root / name)
    for rel in _RULE_NESTED_FILES:
        _add(root / rel)

    rules_dir = root / _RULES_DIR
    if rules_dir.is_dir():
        for path in sorted(rules_dir.rglob("*.md")):
            _add(path)

    return found


def load_rules_text(
    workspace: str | Path | None = None,
    *,
    max_chars: int = DEFAULT_RULES_MAX_CHARS,
    paths: Iterable[Union[str, Path]] | None = None,
) -> str | None:
    """Load and concatenate rules with a hard char budget.

    Returns tagged markdown suitable for system-prompt injection, or None.
    """
    from clawagents.memory.loader import load_memory_files

    file_paths = [Path(p) for p in paths] if paths is not None else discover_rule_paths(workspace)
    if not file_paths:
        return None

    # load_memory_files wraps each file; we then enforce a global budget.
    combined = load_memory_files(file_paths)
    if not combined:
        return None

    header = "## Project Rules (always-on)\n\n"
    body = combined
    # Prefer stripping the default "## Agent Memory" header from loader
    if body.startswith("## Agent Memory"):
        body = body.split("\n", 2)[-1].lstrip()

    text = header + body
    if max_chars > 0 and len(text) > max_chars:
        notice = f"\n\n[rules truncated to {max_chars} chars]\n"
        keep = max(0, max_chars - len(notice))
        text = text[:keep] + notice
    return text
