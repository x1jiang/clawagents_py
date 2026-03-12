"""Feature G: LLM-as-Judge verification.

After a run completes, makes a separate, focused LLM call to evaluate
whether the agent actually accomplished the task. This is more reliable
than heuristic scoring or LLM self-grading because:

  1. It's a separate call with a specific evaluation rubric
  2. The judge sees a compressed summary, not the full messy trajectory
  3. The rubric is designed to catch common false-positives

Returns a 0-3 score with justification, stored alongside heuristic scores.
Controlled by learn=True (same flag as PTRL).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """\
You are an impartial judge evaluating whether an AI coding agent successfully \
completed a task. You are NOT the agent — you are a separate evaluator.

## Task
{task}

## Task Type
{task_type}

## Run Summary
- Outcome reported by agent: {outcome}
- Total turns: {total_turns}
- Tool calls: {total_tool_calls} ({tool_success_rate:.0%} success rate)
- Duration: {duration_s}s
- Mid-run failures: {mid_run_failures}
- Deterministic score: {verified_score} (from actual tool outputs)

## Final Result
{final_result}

## Key Events
{key_events}

## Scoring Rubric
Rate the run on a 0-3 scale:

- **3 — Full success**: Task is completely and correctly accomplished. \
Evidence of correct output visible in tool results.
- **2 — Partial success**: Task is mostly done but with minor issues \
(e.g., works but has edge cases, missing error handling).
- **1 — Minimal progress**: Agent made some progress but did not complete \
the task (e.g., identified the problem but didn't fix it).
- **0 — Failure**: Task was not accomplished. Agent may have gone in \
circles, hit errors it couldn't recover from, or produced wrong output.

## Response Format
Respond with EXACTLY two lines:
SCORE: <0|1|2|3>
REASON: <one sentence justification>

Do not add any other text.
"""


async def judge_run(
    llm: Any,
    task: str,
    summary: dict[str, Any],
    final_result: str,
    key_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Use LLM as an impartial judge to score a completed run.

    Returns:
        {"judge_score": int (0-3), "judge_justification": str}
        or {"judge_score": None, "judge_justification": "judge failed"}
    """
    from clawagents.providers.llm import LLMMessage

    key_events = _format_key_events(key_turns or [], max_events=8)
    v_score = summary.get("verified_score")

    prompt = _JUDGE_PROMPT.format(
        task=task,
        task_type=summary.get("task_type", "general"),
        outcome=summary.get("outcome", "unknown"),
        total_turns=summary.get("total_turns", 0),
        total_tool_calls=summary.get("total_tool_calls", 0),
        tool_success_rate=summary.get("tool_success_rate", 0),
        duration_s=summary.get("duration_s", 0),
        mid_run_failures=summary.get("mid_run_failures", 0),
        verified_score=f"{v_score:.2f}" if v_score is not None else "N/A",
        final_result=(final_result or "")[:500],
        key_events=key_events,
    )

    try:
        messages = [
            LLMMessage(role="system", content="You are an impartial task completion judge."),
            LLMMessage(role="user", content=prompt),
        ]
        response = await llm.chat(messages)
        text = (response.content or "").strip()
        return _parse_judge_response(text)
    except Exception:
        logger.debug("LLM-as-Judge evaluation failed", exc_info=True)
        return {"judge_score": None, "judge_justification": "judge call failed"}


def _format_key_events(turns: list[dict[str, Any]], max_events: int = 8) -> str:
    """Compact summary of key events for the judge."""
    if not turns:
        return "(no events recorded)"

    events: list[str] = []
    for t in turns[:max_events]:
        idx = t.get("turn_index", t.get("turnIndex", "?"))
        score = t.get("score", 0)
        calls = t.get("tool_calls", t.get("toolCalls", []))
        for tc in calls:
            name = tc.get("tool_name", tc.get("toolName", "?"))
            success = tc.get("success", False)
            preview = (tc.get("output_preview", tc.get("outputPreview", "")) or "")[:100]
            status = "OK" if success else "FAIL"
            events.append(f"Turn {idx}: [{status}] {name} — {preview}")

    return "\n".join(events) if events else "(no tool events)"


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the judge's SCORE/REASON response."""
    score = None
    reason = ""

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                val = line.split(":", 1)[1].strip()
                score = int(val)
                if score < 0 or score > 3:
                    score = max(0, min(3, score))
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if score is None:
        # Fallback: try to find a digit 0-3 in the response
        import re
        m = re.search(r'\b([0-3])\b', text)
        if m:
            score = int(m.group(1))
            reason = reason or text[:200]

    return {
        "judge_score": score,
        "judge_justification": reason or text[:200],
    }
