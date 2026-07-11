"""Continuous Background Memory Extraction (learned from Claude Code).

Instead of extracting lessons only at end-of-run, this module provides
a background extraction mechanism that runs every N turns during the
agent loop. It uses a lightweight LLM call to extract actionable memories
from recent conversation turns.

Integration point: Called from agent_loop.py after tool results are recorded.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_EXTRACTION_INTERVAL = 5  # Extract memories every N turns
_MEMORIES_DIR_NAME = "memories"

_MEMORY_EXTRACTION_PROMPT = """\
You are analyzing a recent segment of an AI agent's conversation to extract \
durable, reusable memories.

## Recent Conversation Segment (turns {start_turn} to {end_turn})
{conversation_segment}

## Instructions
Extract 0-3 short, typed memories from this conversation segment. Each memory \
should be a reusable fact, preference, or project detail that would help the agent \
in future runs.

Respond with a JSON array. Each entry:
```json
[
  {{
    "type": "project|user|feedback|reference",
    "content": "one-line actionable memory",
    "confidence": 0.0-1.0
  }}
]
```

Only extract HIGH-CONFIDENCE memories (>0.7). If nothing is worth remembering, \
respond with `[]`.

Types:
- project: project-specific facts (file paths, architecture, tech stack)
- user: user preferences (coding style, tool preferences)
- feedback: corrections to agent behavior from this run
- reference: reference values (URLs, config, env vars)
"""


def _get_memories_dir() -> Path:
    return Path.cwd() / ".clawagents" / _MEMORIES_DIR_NAME


def _format_messages_segment(messages: list[Any], start: int, end: int) -> str:
    """Format a segment of messages for the extraction prompt."""
    lines: list[str] = []
    for i, msg in enumerate(messages[start:end], start=start):
        role = msg.role.upper()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        content = content[:300]  # Keep it compact
        lines.append(f"[{role} turn {i}]: {content}")
    return "\n".join(lines)


async def extract_background_memories(
    llm: Any,
    messages: list[Any],
    start_turn: int,
    end_turn: int,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Extract memories from a segment of conversation turns.

    Returns a list of memory dicts with type, content, and confidence.
    """
    from clawagents.providers.llm import LLMMessage

    segment = _format_messages_segment(messages, start_turn, end_turn)
    prompt = _MEMORY_EXTRACTION_PROMPT.format(
        start_turn=start_turn,
        end_turn=end_turn,
        conversation_segment=segment,
    )

    try:
        response = await llm.chat([LLMMessage(role="user", content=prompt)])
        # Parse JSON response
        text = response.content.strip()
        # Handle fenced code blocks
        if "```" in text:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        memories = json.loads(text)
        if not isinstance(memories, list):
            return []

        # Filter by confidence
        return [
            m for m in memories
            if isinstance(m, dict) and m.get("confidence", 0) >= 0.7
        ]
    except Exception as exc:
        logger.debug("Background memory extraction failed: %s", exc)
        return []


def save_memories(memories: list[dict[str, Any]], turn_index: int) -> Optional[str]:
    """Save extracted memories to .clawagents/memories/ as typed markdown files."""
    if not memories:
        return None

    try:
        mem_dir = _get_memories_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        lines: list[str] = []
        for m in memories:
            mem_type = m.get("type", "general")
            content = m.get("content", "")
            confidence = m.get("confidence", 0)

            # Write as typed memory with frontmatter
            entry = f"""---
type: {mem_type}
confidence: {confidence}
turn: {turn_index}
timestamp: {ts}
---
{content}
"""
            lines.append(entry)

        # Append to memories file
        mem_file = mem_dir / "extracted.md"
        with open(mem_file, "a") as f:
            f.write("\n".join(lines) + "\n")

        return str(mem_file)
    except Exception as exc:
        logger.debug("Failed to save memories: %s", exc)
        return None


async def maybe_extract_memories(
    llm: Any,
    messages: list[Any],
    round_idx: int,
    last_extraction_turn: int,
    interval: int = _EXTRACTION_INTERVAL,
) -> int:
    """Run background memory extraction if enough turns have passed.

    Returns the updated last_extraction_turn value.
    """
    from clawagents.config.features import is_enabled
    if not is_enabled("background_memory"):
        return last_extraction_turn

    if round_idx - last_extraction_turn < interval:
        return last_extraction_turn

    # Run extraction as a background task
    try:
        start = max(0, last_extraction_turn)
        end = min(len(messages), round_idx + 2)  # +2 for the latest pair
        memories = await extract_background_memories(llm, messages, start, end)
        if memories:
            save_memories(memories, round_idx)
            logger.debug("Extracted %d background memories at turn %d", len(memories), round_idx)
    except Exception as exc:
        logger.debug("Background memory extraction error: %s", exc)

    return round_idx
