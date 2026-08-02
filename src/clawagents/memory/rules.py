"""Always-on project rules discovery (Cline .clinerules-inspired)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Union

# Files / dirs searched relative to workspace cwd.
#
# Short, user-pinned context that a UI can edit inline. Listed first so it
# leads the injected block: it is the smallest and most situational of the
# rules sources (which .venv to use, which skill to remember this week), and
# it is the one most likely to be lost to the char budget if it went last.
_PINNED_CONTEXT_FILE = ".clawagents/pinned-context.md"
_PINNED_HEADING = "# Pinned context (always applies)"
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

    _add(root / _PINNED_CONTEXT_FILE)
    for name in _RULE_ROOT_FILES:
        _add(root / name)
    for rel in _RULE_NESTED_FILES:
        _add(root / rel)

    rules_dir = root / _RULES_DIR
    if rules_dir.is_dir():
        for path in sorted(rules_dir.rglob("*.md")):
            _add(path)

    return found


def pinned_context_path(workspace: str | Path | None = None) -> Path:
    """Where inline pinned context lives for a workspace.

    A plain file rather than app state so it is diffable, editable outside the
    UI, and picked up by every ClawAgents front end via rules discovery.
    """
    return Path(workspace or os.getcwd()).resolve() / _PINNED_CONTEXT_FILE


def read_pinned_context(workspace: str | Path | None = None) -> str:
    """Current pinned context as the user wrote it, or ``""`` when unset.

    The stored file carries a generated heading so the model can tell this
    block apart from the other rules sources concatenated around it. That
    heading is ours, not the user's, so strip it here — otherwise an editor
    round-trip shows it as editable text and re-saving nests a second copy.
    """
    try:
        raw = pinned_context_path(workspace).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    body = raw.lstrip()
    if body.startswith(_PINNED_HEADING):
        body = body[len(_PINNED_HEADING):].lstrip("\n")
    return body.strip()


def write_pinned_context(
    text: str,
    workspace: str | Path | None = None,
    *,
    max_chars: int = 4_000,
) -> str:
    """Persist pinned context, or remove the file when cleared.

    Bounded because this text is injected on *every* LLM round — an accidental
    paste of a whole log would quietly tax each request. Returns what was
    stored so a caller can echo back the truncated value.
    """
    path = pinned_context_path(workspace)
    body = (text or "").strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
    if not body:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Heading travels with the content so the model sees why this text is here
    # even though rules injection concatenates several sources.
    payload = f"{_PINNED_HEADING}\n\n{body}\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return body


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
