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
from typing import Any

logger = logging.getLogger(__name__)

def _get_clawagents_dir() -> Path:
    return Path.cwd() / ".clawagents"

def _get_lessons_file() -> Path:
    return _get_clawagents_dir() / "lessons.md"
_MAX_LESSONS = 50
_MAX_LESSONS_CHARS = 3200  # Harness²: keep always-on lesson budget compact; promote procedures to on-demand skills
_LESSON_MAX_AGE_RUNS = 50   # Feature 2: drop lessons older than this many runs
_DEFAULT_OBS_TRUNC = 1200   # Harness²: avoid 80-char clipping that fabricates defects

# ─── Post-Run Self-Analysis (Harness² INV / DOM / WORKSPACE_FACT Taxonomy) ───

_SELF_ANALYSIS_PROMPT = """\
You are reviewing an AI coding agent trajectory to extract causal, checkable lessons \
without bloating always-on context.

## Run Summary
- Task: {task}
- Task type: {task_type}
- Outcome: {outcome}
- Finish reason: {finish_reason}
- Run score: {run_score}/3  (3=clean, 2=efficient, 1=messy success, 0=ambiguous, -1=failed)
- Quality: {quality}
- Verified score: {verified_score} (objective, from tool outputs; confidence={verified_confidence}, method={verified_method})
- Total turns: {total_turns}
- Mid-run failures: {mid_run_failures}
- Format errors: {format_failures} (bad JSON, wrong params, unknown tools)
- Logic errors: {logic_failures} (valid calls, wrong approach)
- Duration: {duration_s}s

## Key Turns (failures and pivots)
{key_turns}

## Instructions (Harness² Orthogonal Dimensions)
Separate your diagnosis along two orthogonal dimensions plus workspace facts:
1. **[INV] Invariant Machinery**: Tool error recovery, bounded fallbacks, deterministic helper scripts (`scripts/*.py`), verification order, and output budget protection.
2. **[DOM] Domain / Practitioner Contract**: Deliverable structure, cross-file reconciliation checks, required fields/columns, and explicit uncertainty handling.
3. **[WORKSPACE_FACT] Anchored Workspace Fact**: Verified join keys, schema mappings, or paths discovered in THIS workspace. Always prefix with `In this workspace, ...` so a reader can tell workspace fact from portable procedure. Never guess facts.
4. **Causal Evidence Rule**: Every lesson must trace to a specific turn/observation in the trajectory. If the run shows no concrete, reusable contrast, respond with `NO_ENTRY`.

Respond with 1-4 concise markdown bullets (`- [INV] ...`, `- [DOM] ...`, or `- [WORKSPACE_FACT] In this workspace, ...`), optionally followed by one `<experience>` block for the single best transferable procedure:

<experience>
cue: <the shape of situation this applies to>
lesson: <the checkable operation to perform>
evidence: <the turn and tool contrast supporting this lesson>
confidence: <high|medium|low>
dimension: <INV|DOM>
kind: <procedure|workspace_fact>
</experience>
"""


def _extract_key_turns(
    turns: list[dict[str, Any]],
    max_turns: int = 10,
    obs_trunc: int = _DEFAULT_OBS_TRUNC,
) -> str:
    """Pick the most informative turns: failures, pivots (score changed sign), early and late.

    Uses a generous `obs_trunc` (default 1200 chars instead of 80) so reflection
    reads actual error signatures and table schemas rather than fabricating defects.
    """
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
        productivity = t.get("productivity_score", 0)
        calls_info = []
        for tc in t.get("tool_calls", []):
            status = "OK" if tc.get("success") else "FAIL"
            name = tc.get("tool_name", "?")
            ft = tc.get("failure_type", "")
            ft_tag = f" [{ft}]" if ft and not tc.get("success") else ""
            preview = (tc.get("output_preview", "") or "")[:obs_trunc]
            calls_info.append(f"  - [{status}{ft_tag}] {name}: {preview}")
        resp = (t.get("response_text", "") or "")[:600]
        obs = (t.get("observation_context", "") or "")[:400]
        lines.append(f"### Turn {idx} (score={t.get('score', 0)}, productivity={productivity})")
        if obs:
            lines.append(f"Context: {obs}")
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
    v_score = summary.get("verified_score")
    prompt = _SELF_ANALYSIS_PROMPT.format(
        task=summary.get("task", "unknown"),
        task_type=summary.get("task_type", "general"),
        outcome=summary.get("outcome", "unknown"),
        finish_reason=summary.get("finish_reason", "unknown"),
        run_score=summary.get("run_score", 0),
        quality=summary.get("quality", "unknown"),
        verified_score=f"{v_score:.2f}" if v_score is not None else "N/A",
        verified_confidence=summary.get("verified_confidence", "N/A"),
        verified_method=summary.get("verified_method", "N/A"),
        total_turns=summary.get("total_turns", 0),
        mid_run_failures=summary.get("mid_run_failures", 0),
        format_failures=summary.get("format_failures", 0),
        logic_failures=summary.get("logic_failures", 0),
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


# ─── Feature 1: Quality Gate ─────────────────────────────────────────────────

def should_extract_lessons(summary: dict[str, Any]) -> bool:
    """Determine if a run has enough signal for lesson extraction.

    SkyRL-inspired: only extract lessons when the trajectory has contrast
    (both successes and failures), not when it's uniformly good or bad.
    Zero-variance runs carry no learning signal.

    Feature A: When verified_score is available and disagrees with run_score,
    that's a high-signal run worth learning from.
    """
    quality = summary.get("quality", "")
    run_score = summary.get("run_score", 0)
    has_mixed = summary.get("has_mixed_outcomes", False)
    mid_run_failures = summary.get("mid_run_failures", 0)
    total_turns = summary.get("total_turns", 0)
    verified_score = summary.get("verified_score")

    # Feature A: score disagreement is high signal
    if verified_score is not None:
        run_normalized = run_score / 3.0
        if abs(run_normalized - verified_score) > 0.4:
            return True

    # Always extract from failed runs with at least some turns (negative examples)
    if run_score <= -1 and total_turns >= 3:
        return True

    # Mixed-outcome runs are the most valuable (natural A/B within a single run)
    if has_mixed:
        return True

    # Clean all-success runs: nothing to learn
    if run_score >= 3 and mid_run_failures == 0:
        return False

    # Noisy runs with many failures: worth analyzing
    if quality == "noisy":
        return True

    # Efficient runs (score 2) with some failures: borderline useful
    if run_score == 2 and mid_run_failures > 0:
        return True

    # Default: skip (all-success or trivial runs)
    return False


# ─── Lesson Storage ──────────────────────────────────────────────────────────

def save_lessons(
    new_lessons: str,
    task: str,
    outcome: str,
    model: str = "",
) -> None:
    """Append new lessons to .clawagents/lessons.md with staleness metadata and deduplication.

    Harness² budget discipline:
      - `NO_ENTRY` or grader verdict leaks are never written.
      - Duplicate lesson bullets already present in `.clawagents/lessons.md` are skipped
        ("cut-to-pay": never pay twice in always-on tokens for the same rule).
      - If a structured `<experience>` block is present, it is also recorded into the
        domain `ExperienceLibrary` (`.clawagents/experience/<domain>/INDEX.md`).
    """
    try:
        from clawagents.trajectory.harness_surface import (
            ExperienceLibrary,
            assert_no_grader_verdicts,
            parse_experience_reply,
        )

        cleaned_raw = (new_lessons or "").strip()
        if not cleaned_raw or cleaned_raw == "NO_ENTRY":
            return
        assert_no_grader_verdicts(cleaned_raw)

        claw_dir = _get_clawagents_dir()
        claw_dir.mkdir(parents=True, exist_ok=True)

        # Persist structured <experience> to per-domain library if present
        exp_entry = parse_experience_reply(cleaned_raw)
        if exp_entry is not None:
            lib = ExperienceLibrary(claw_dir / "experience", domain="general")
            lib.add(task[:64] or "run", exp_entry, label=model or "agent")

        # Strip <edits>/<experience> XML blocks from the always-on bullet file
        import re as _re

        bullet_part = _re.sub(r"<(?:edits|experience)>.*?</(?:edits|experience)>", "", cleaned_raw, flags=_re.S).strip()
        if not bullet_part and exp_entry is not None:
            prefix = "[WORKSPACE_FACT] " if exp_entry.kind == "workspace_fact" else f"[{exp_entry.dimension}] "
            bullet_part = f"- {prefix}When {exp_entry.cue}: {exp_entry.lesson}"
        if not bullet_part:
            return

        existing = ""
        lessons_file = _get_lessons_file()
        if lessons_file.exists():
            existing = lessons_file.read_text(encoding="utf-8")

        existing_keys = {lesson_key(b) for b in parse_lesson_bullets(existing)}
        novel_lines: list[str] = []
        for ln in bullet_part.splitlines():
            s = ln.strip()
            if s.startswith("- "):
                k = lesson_key(s)
                if k in existing_keys:
                    continue
                existing_keys.add(k)
            novel_lines.append(ln)

        novel_body = "\n".join(novel_lines).strip()
        if not novel_body:
            return

        # Feature 2: tag with timestamp and model for staleness decay
        ts = int(time.time())
        model_tag = f" [{model}]" if model else ""
        header = f"\n## Lessons from run ({outcome}) — {task[:80]}{model_tag} @{ts}\n"
        entry = header + novel_body + "\n"

        combined = existing + "\n" + entry

        lines = combined.strip().split("\n")
        if len(lines) > _MAX_LESSONS * 5:
            lines = lines[-((_MAX_LESSONS * 5)):]
        if len("\n".join(lines)) > _MAX_LESSONS_CHARS * 3:
            lines = lines[-((_MAX_LESSONS * 3)):]

        lessons_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.debug("PTRL: saved lessons to %s", lessons_file)
    except Exception:
        logger.debug("PTRL: failed to save lessons", exc_info=True)


def load_lessons(max_chars: int = _MAX_LESSONS_CHARS, max_age_s: int = 0) -> str:
    """Load existing lessons from .clawagents/lessons.md.

    Feature 2: If max_age_s > 0, only include lesson blocks whose @timestamp
    is within the given age window. Stale lessons are silently dropped.

    Returns empty string if no lessons exist or file is not readable.
    """
    try:
        lessons_file = _get_lessons_file()
        if not lessons_file.exists():
            return ""
        text = lessons_file.read_text(encoding="utf-8").strip()
        if not text:
            return ""

        # Feature 2: filter by age if requested
        if max_age_s > 0:
            import re
            now = int(time.time())
            cutoff = now - max_age_s
            blocks = re.split(r"(?=\n## Lessons from run)", text)
            fresh: list[str] = []
            for block in blocks:
                ts_match = re.search(r"@(\d{10,})", block)
                if ts_match:
                    ts = int(ts_match.group(1))
                    if ts >= cutoff:
                        fresh.append(block)
                else:
                    fresh.append(block)
            text = "\n".join(fresh).strip()
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
    """Build a system prompt section with past lessons and auto-wired skill triggers."""
    lessons = load_lessons()
    triggers_file = _get_clawagents_dir() / "skill-triggers.md"
    triggers = ""
    if triggers_file.is_file():
        try:
            triggers = triggers_file.read_text(encoding="utf-8").strip()
        except OSError:
            triggers = ""
    if not lessons and not triggers:
        return ""
    parts: list[str] = []
    if lessons:
        parts.append(
            "\n\n## Lessons from Past Runs\n"
            "These lessons were extracted from previous agent runs. "
            "Apply them to avoid repeating past mistakes:\n\n"
            f"{lessons}\n"
        )
    if triggers:
        parts.append(f"\n\n{triggers}\n")
    return "".join(parts)


def build_rethink_with_lessons(
    generic_rethink: str,
    format_failure_count: int = 0,
    logic_failure_count: int = 0,
) -> str:
    """Enhance a rethink message with past lessons and failure-type guidance.

    Feature 3: If recent failures are predominantly format errors, add
    specific guidance about tool call formatting.
    """
    parts = [generic_rethink]

    if format_failure_count > 0 and format_failure_count >= logic_failure_count:
        parts.append(
            "\n\n## Format Error Guidance\n"
            "Your recent tool call failures are FORMAT errors (bad JSON, wrong parameter "
            "names, unknown tools). Before retrying:\n"
            "- Check that tool names match exactly (case-sensitive)\n"
            "- Ensure all required parameters are provided\n"
            "- Verify JSON syntax is valid (matching braces, quoted strings)\n"
            "- Review the tool descriptions above for correct parameter names"
        )
    elif logic_failure_count > 0:
        parts.append(
            "\n\n## Strategy Guidance\n"
            "Your recent failures are LOGIC errors (correct tool calls, wrong approach). "
            "The tools work but your strategy needs adjustment. "
            "Try a fundamentally different approach."
        )

    lessons = load_lessons(max_chars=1500)
    if lessons:
        parts.append(
            "\n\n## Relevant Lessons from Past Runs\n"
            "Consider these lessons from previous runs:\n\n"
            f"{lessons}"
        )

    return "\n".join(parts)


def classify_lesson_dimension(bullet: str) -> tuple[str, str]:
    """Return `(dimension, kind)` for a lesson bullet (`INV|DOM`, `procedure|workspace_fact`)."""
    raw = (bullet or "").strip().lstrip("- ").strip()
    upper = raw.upper()
    if upper.startswith("[WORKSPACE_FACT]") or "in this workspace" in raw.lower():
        return "DOM", "workspace_fact"
    if upper.startswith("[DOM]"):
        return "DOM", "procedure"
    if upper.startswith("[INV]"):
        return "INV", "procedure"
    return "INV", "procedure"


def strip_lesson_tag(text: str) -> str:
    """Strip leading `[INV]`, `[DOM]`, or `[WORKSPACE_FACT]` tag from a lesson bullet."""
    import re as _re

    return _re.sub(r"^\s*-\s*\[(?:INV|DOM|WORKSPACE_FACT)\]\s*", "", text.strip(), flags=_re.I).lstrip("- ").strip()


def normalize_lesson(text: str) -> str:
    return " ".join(strip_lesson_tag(text).lower().split())


def lesson_key(text: str) -> str:
    import hashlib

    return hashlib.sha256(normalize_lesson(text).encode()).hexdigest()[:16]


def parse_lesson_bullets(markdown: str) -> list[str]:
    bullets: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            body = stripped[2:].strip()
            if body:
                bullets.append(body)
    return bullets


def slugify_lesson_name(text: str) -> str:
    import re

    words = re.sub(r"[^a-z0-9\s-]", "", normalize_lesson(text)).split()[:5]
    slug = "-".join(words)[:48].strip("-")
    return slug or "recurring-lesson"


def export_lessons(output_path: str | Path | None = None) -> str:
    """Export current lessons to a JSON file. Returns the file path."""
    lessons = load_lessons(max_chars=999999)
    path = Path(output_path) if output_path else _get_clawagents_dir() / "lessons_export.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "exported_at": int(time.time()),
        "lessons_md": lessons,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def import_lessons(input_path: str | Path) -> bool:
    """Import lessons from a JSON export file. Returns True on success."""
    try:
        path = Path(input_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not data.get("lessons_md"):
            logger.warning("Invalid lessons export format")
            return False

        _get_clawagents_dir().mkdir(parents=True, exist_ok=True)
        lessons_file = _get_lessons_file()

        existing = ""
        if lessons_file.exists():
            existing = lessons_file.read_text(encoding="utf-8")

        combined = existing + "\n\n## Imported Lessons\n" + data["lessons_md"]
        lessons_file.write_text(combined.strip() + "\n", encoding="utf-8")
        return True
    except Exception:
        logger.debug("Failed to import lessons", exc_info=True)
        return False
