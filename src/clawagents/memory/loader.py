"""AGENTS.md / CLAWAGENTS.md memory loader with typed memory support.

Reads project-specific memory files and returns their combined content
for injection into the agent's system prompt.

Typed Memory (learned from Claude Code):
  Memory files can include YAML frontmatter with type/name/description metadata.
  This enables type-based filtering and recall precision.

  Supported types:
    - user:      User preferences ("prefers pytest -x")
    - feedback:  Corrections to agent behavior ("stop summarizing diffs")
    - project:   Project-specific facts ("sprint deadline is March 15")
    - reference: Reference values ("staging URL: https://...")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re

# ─── Frontmatter Parser (learned from Claude Code: typed memory taxonomy) ──

_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

VALID_MEMORY_TYPES = frozenset({"user", "feedback", "project", "reference", "general"})


def parse_memory_frontmatter(content: str) -> Dict[str, Any]:
    """Parse YAML-like frontmatter from a memory file.

    Returns a dict with at least 'type' and 'content' keys.
    If no frontmatter is found, returns type='general' with the full content.

    Example frontmatter:
        ---
        name: user_testing_preference
        type: feedback
        description: User wants pytest -x flag always used
        ---
        When running tests, always use `pytest -x` to stop on first failure.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {"type": "general", "content": content, "name": "", "description": ""}

    meta: Dict[str, str] = {}
    for line in match.group(1).split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    # Validate type
    mem_type = meta.get("type", "general")
    if mem_type not in VALID_MEMORY_TYPES:
        mem_type = "general"

    return {
        "type": mem_type,
        "name": meta.get("name", ""),
        "description": meta.get("description", ""),
        "content": content[match.end():].strip(),
    }


def load_memory_files(
    paths: List[Union[str, Path]],
    filter_type: Optional[str] = None,
) -> Optional[str]:
    """Read memory files and return combined content wrapped in tags.

    Args:
        paths: List of file paths to AGENTS.md / CLAWAGENTS.md files.
        filter_type: If set, only include memories of this type.

    Returns:
        Combined content string or None if no files were found/readable.
    """
    from clawagents.config.features import is_enabled

    sections: list[str] = []

    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        try:
            raw = path.read_text("utf-8").strip()
            if not raw:
                continue

            source = path.name

            # Typed memory: parse frontmatter if feature is enabled
            if is_enabled("typed_memory"):
                parsed = parse_memory_frontmatter(raw)
                mem_type = parsed["type"]

                # Apply type filter
                if filter_type and mem_type != filter_type:
                    continue

                content = parsed["content"]
                type_attr = f' type="{mem_type}"' if mem_type != "general" else ""
                name_attr = f' name="{parsed["name"]}"' if parsed["name"] else ""
                sections.append(
                    f'<agent_memory source="{source}"{type_attr}{name_attr}>\n{content}\n</agent_memory>'
                )
            else:
                sections.append(
                    f'<agent_memory source="{source}">\n{raw}\n</agent_memory>'
                )
        except Exception:
            continue

    if not sections:
        return None

    return "## Agent Memory\n\n" + "\n\n".join(sections)


def load_memory_directory(
    dir_path: Union[str, Path],
    filter_type: Optional[str] = None,
) -> Optional[str]:
    """Load all .md files from a memory directory.

    Designed for Claude Code-style memory directories where each memory
    is a separate markdown file with frontmatter.
    """
    d = Path(dir_path)
    if not d.exists() or not d.is_dir():
        return None

    memory_files = sorted(d.glob("*.md"))
    if not memory_files:
        return None

    return load_memory_files([str(f) for f in memory_files], filter_type=filter_type)
