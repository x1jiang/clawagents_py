"""System prompt composition analyzer.

Detects memory component boundaries in the assembled system prompt and
calculates per-component token counts. Components are identified by their
section headers (## Core Memory, ## Live Facts, etc.) as injected by the
memory modules in ``agent_loop.py`` lines 2185-2220.
"""

from __future__ import annotations

import re
from typing import Any

from clawagents.tokenizer import count_tokens

# Section headers injected by the memory pipeline. Order matters: later
# sections may be nested inside earlier ones, so we scan in reverse priority.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("core_memory", "## Core Memory"),
    ("facts", "## Live Facts"),
    ("context_ledger", "## Context Ledger"),
    ("memory_bank", "## Memory Bank"),
    ("repo_map", "## Repo Map"),
    ("rules", "## Project Rules"),
    ("carryover", "## Carryover State"),
    ("lessons", "## Lessons"),
    ("skills", "## Skills"),
    ("agent_memory", "<agent_memory"),
]


def analyze_system_prompt(
    content: str | list[Any],
    model: str | None = None,
) -> dict[str, int]:
    """Return a {component_name: token_count} breakdown of the system prompt.

    The system prompt is assembled from a base prompt + tool descriptions +
    dynamic context packs (core memory, facts, rules, repo map, etc.).
    This function detects each component by its section header and computes
    individual token counts.

    Any text that doesn't match a known section is counted under ``base_prompt``.
    """
    if not content:
        return {}

    text = content if isinstance(content, str) else str(content)
    total = count_tokens(text, model)

    breakdown: dict[str, int] = {}
    remaining = text

    # Extract each known section
    for name, marker in _SECTION_PATTERNS:
        section_text = _extract_section(remaining, marker)
        if section_text:
            tokens = count_tokens(section_text, model)
            breakdown[name] = tokens
            # Remove from remaining to avoid double-counting
            remaining = remaining.replace(section_text, "", 1)

    # Whatever is left is the base prompt + tool descriptions
    remaining_stripped = remaining.strip()
    if remaining_stripped:
        breakdown["base_prompt"] = count_tokens(remaining_stripped, model)

    # Verify: breakdown should sum close to total (rounding errors ok)
    breakdown_total = sum(breakdown.values())
    if abs(breakdown_total - total) > 20:
        # Adjust base_prompt to account for rounding / overlap
        diff = total - breakdown_total
        breakdown["base_prompt"] = breakdown.get("base_prompt", 0) + diff

    return breakdown


def _extract_section(text: str, marker: str) -> str:
    """Extract the text belonging to a section starting at ``marker``.

    A section extends from its header line until the next ``##`` header
    or ``<agent_memory`` tag or end-of-string.
    """
    idx = text.find(marker)
    if idx < 0:
        return ""

    start = idx

    # For XML-style markers, find the closing tag
    if marker.startswith("<"):
        tag_name = marker.lstrip("<").rstrip(">").split()[0]
        close_tag = f"</{tag_name}>"
        close_idx = text.find(close_tag, start)
        if close_idx >= 0:
            return text[start : close_idx + len(close_tag)]
        # No closing tag — take until next section or end
        return _find_section_end(text, start)

    return _find_section_end(text, start)


def _find_section_end(text: str, start: int) -> str:
    """Find the end of a section starting at ``start``.

    Sections end at the next ``##`` header that isn't part of the current
    section (i.e., skip the header at ``start`` itself).
    """
    # Skip past the header line itself
    header_end = text.find("\n", start)
    if header_end < 0:
        return text[start:]

    # Search for next section header
    next_section = re.search(r"\n##\s+\S", text[header_end + 1 :])
    if next_section:
        end = header_end + 1 + next_section.start()
        return text[start:end]

    return text[start:]


def compute_role_tokens(
    messages: list[Any],
    model: str | None = None,
) -> dict[str, int]:
    """Compute total tokens per message role.

    Returns: {system: N, user: N, assistant: N, tool: N}
    """
    role_tokens: dict[str, int] = {}
    for m in messages:
        role = getattr(m, "role", "unknown")
        content = getattr(m, "content", "")
        if isinstance(content, list):
            # Multimodal: estimate from text parts + image overhead
            text = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
            tokens = count_tokens(text, model)
            tokens += sum(
                500 for p in content
                if isinstance(p, dict) and p.get("type") == "image_url"
            )
        elif isinstance(content, str):
            tokens = count_tokens(content, model)
        else:
            tokens = count_tokens(str(content), model)
        role_tokens[role] = role_tokens.get(role, 0) + tokens
    return role_tokens
