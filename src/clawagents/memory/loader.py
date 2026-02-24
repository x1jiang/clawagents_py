"""AGENTS.md / CLAWAGENTS.md memory loader.

Reads project-specific memory files and returns their combined content
for injection into the agent's system prompt.
"""

from pathlib import Path
from typing import List, Optional, Union


def load_memory_files(paths: List[Union[str, Path]]) -> Optional[str]:
    """Read memory files and return combined content wrapped in tags.

    Args:
        paths: List of file paths to AGENTS.md / CLAWAGENTS.md files.

    Returns:
        Combined content string or None if no files were found/readable.
    """
    sections: list[str] = []

    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text("utf-8").strip()
            if content:
                source = path.name
                sections.append(
                    f"<agent_memory source=\"{source}\">\n{content}\n</agent_memory>"
                )
        except Exception:
            continue

    if not sections:
        return None

    return "## Agent Memory\n\n" + "\n\n".join(sections)
