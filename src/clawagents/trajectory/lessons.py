"""Prompt-Time Reinforcement Learning (PTRL) — lesson extraction & injection.

Three layers that create a feedback loop without model fine-tuning:

  1. **Post-run self-analysis**: After a run completes, the LLM reviews its own
     trajectory and extracts actionable lessons (what worked, what didn't, tips
     for next time). Stored in .clawagents/lessons.md.

  2. **Pre-run lesson injection**: Before a new run starts, any existing lessons
     are loaded from .clawagents/lessons.md and prepended to the system prompt
     so the agent doesn't repeat past mistakes.

  3. **Enhanced mid-run rethink**: When consecutive failures are detected (via
     the rethink flag), relevant lessons are injected alongside the generic
     "stop and rethink" prompt.

Controlled by the CLAW_LEARN flag (or learn=True in create_claw_agent).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CLAWAGENTS_DIR = Path.cwd() / ".clawagents"
_LESSONS_FILE = _CLAWAGENTS_DIR / "lessons.md"
_MAX_LESSONS = 50
_MAX_LESSONS_CHARS = 4000

# ─── Post-Run Self-Analysis ──────────────────────────────────────────────────

_SELF_ANALYSIS_PROMPT = """\
You are reviewing your own agent run trajectory. Analyze the run and extract \
concise, actionable lessons.

## Run Summary
- Task: {task}
- Outcome: {outcome}
- Run score: {run_score}/3  (3=clean, 2=efficient, 1=messy success, 0=ambiguous, -1=failed)
- Quality: {quality}
- Total turns: {total_turns}
- Mid-run failures: {mid_run_failures}
- Duration: {duration_s}s

## Key Turns (failures and pivots)
{key_turns}

## Instructions
Based on this trajectory:
1. What went wrong? (specific tool failures, bad strategies, repeated mistakes)
2. What worked? (successful approaches, efficient patterns)
3. What should the agent do differently next time?

Respond with a markdown list of 2-5 concise lessons. Each lesson should be a \
single line starting with "- ". Focus on ACTIONABLE advice, not vague platitudes.

Example format:
- When file X doesn't exist, check parent directory first instead of retrying the same path
- Use grep to search before attempting to read large files
- Prefer execute_command over write_file+execute for one-off scripts
"""


def _extract_key_turns(turns: list[dict[str, Any]], max_turns: int = 10) -> str:
    """Pick the most informative turns: failures, pivots (score changed sign), early and late."""
    if not turns:
        return "(no turns recorded)"

    key: list[dict[str, Any]] = []
    prev_score = 0
    for t in turns:
        score = t.get("score", 0)
        is_failure = score < 0
        is_pivot = (score > 0 and prev_score < 0) or (score < 0 and prev_score > 0)
        if is_failure or is_pivot:
            key.append(t)
        prev_score = score

    if turns:
        if turns[0] not in key:
            key.insert(0, turns[0])
        if turns[-1] not in key:
            key.append(turns[-1])

    key = key[:max_turns]

    lines: list[str] = []
    for t in key:
        idx = t.get("turn_index", "?")
        calls_info = []
        for tc in t.get("tool_calls", []):
            status = "OK" if tc.get("success") else "FAIL"
            name = tc.get("tool_name", "?")
            preview = tc.get("output_preview", "")[:80]
            calls_info.append(f"  - [{status}] {name}: {preview}")
        resp = (t.get("response_text", "") or "")[:200]
        lines.append(f"### Turn {idx} (score={t.get('score', 0)})")
        if resp:
            lines.append(f"Response: {resp}")
        if calls_info:
            lines.append("\n".join(calls_info))
    return "\n".join(lines) if lines else "(no key turns)"


async def extract_lessons(
    llm: Any,
    summary: dict[str, Any],
    turns: list[dict[str, Any]],
) -> str | None:
    """Use the LLM to self-analyze a completed run and extract lessons.

    Returns the raw markdown lesson text, or None on failure.
    """
    from clawagents.providers.llm import LLMMessage

    key_turns = _extract_key_turns(turns)
    prompt = _SELF_ANALYSIS_PROMPT.format(
        task=summary.get("task", "unknown"),
        outcome=summary.get("outcome", "unknown"),
        run_score=summary.get("run_score", 0),
        quality=summary.get("quality", "unknown"),
        total_turns=summary.get("total_turns", 0),
        mid_run_failures=summary.get("mid_run_failures", 0),
        duration_s=summary.get("duration_s", 0),
        key_turns=key_turns,
    )

    try:
        messages = [
            LLMMessage(role="system", content="You are a self-improvement analyst for an AI coding agent."),
            LLMMessage(role="user", content=prompt),
        ]
        response = await llm.chat(messages)
        text = response.content.strip() if response.content else None
        return text
    except Exception:
        logger.debug("PTRL: lesson extraction failed", exc_info=True)
        return None


# ─── Lesson Storage ──────────────────────────────────────────────────────────

def save_lessons(new_lessons: str, task: str, outcome: str) -> None:
    """Append new lessons to .clawagents/lessons.md, keeping it bounded."""
    try:
        _CLAWAGENTS_DIR.mkdir(parents=True, exist_ok=True)

        header = f"\n## Lessons from run ({outcome}) — {task[:80]}\n"
        entry = header + new_lessons.strip() + "\n"

        existing = ""
        if _LESSONS_FILE.exists():
            existing = _LESSONS_FILE.read_text(encoding="utf-8")

        combined = existing + "\n" + entry

        lines = combined.strip().split("\n")
        if len(lines) > _MAX_LESSONS * 5:
            lines = lines[-((_MAX_LESSONS * 5)):]
        if len("\n".join(lines)) > _MAX_LESSONS_CHARS * 3:
            lines = lines[-((_MAX_LESSONS * 3)):]

        _LESSONS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.debug("PTRL: saved lessons to %s", _LESSONS_FILE)
    except Exception:
        logger.debug("PTRL: failed to save lessons", exc_info=True)


def load_lessons(max_chars: int = _MAX_LESSONS_CHARS) -> str:
    """Load existing lessons from .clawagents/lessons.md.

    Returns empty string if no lessons exist or file is not readable.
    """
    try:
        if not _LESSONS_FILE.exists():
            return ""
        text = _LESSONS_FILE.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        if len(text) > max_chars:
            text = text[-max_chars:]
            nl = text.find("\n")
            if nl > 0:
                text = text[nl + 1:]
        return text
    except Exception:
        logger.debug("PTRL: failed to load lessons", exc_info=True)
        return ""


def build_lesson_preamble() -> str:
    """Build a system prompt section with past lessons, if any exist."""
    lessons = load_lessons()
    if not lessons:
        return ""
    return (
        "\n\n## Lessons from Past Runs\n"
        "These lessons were extracted from previous agent runs. "
        "Apply them to avoid repeating past mistakes:\n\n"
        f"{lessons}\n"
    )


def build_rethink_with_lessons(generic_rethink: str) -> str:
    """Enhance a generic rethink message with relevant past lessons."""
    lessons = load_lessons(max_chars=1500)
    if not lessons:
        return generic_rethink
    return (
        f"{generic_rethink}\n\n"
        "## Relevant Lessons from Past Runs\n"
        "Consider these lessons from previous runs:\n\n"
        f"{lessons}\n"
    )
