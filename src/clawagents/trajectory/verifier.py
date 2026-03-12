"""Task verification and deterministic reward signals.

Feature A: Deterministic rewards from tool execution — uses exit codes and
           tool outputs as objective ground truth instead of LLM self-grading.
Feature C: Task-type-aware verification — auto-detects task type and applies
           the right verifier (coding, file, search, general).
Feature F: Adaptive rethink threshold — adjusts rethink sensitivity based on
           task complexity and run progress.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ─── Feature A: Deterministic Rewards from Tool Execution ─────────────────

def compute_deterministic_score(tool_calls: list[dict[str, Any]]) -> float | None:
    """Compute an objective reward signal from execution tool results.

    Uses actual exit codes and outputs rather than LLM self-assessment.
    Returns a float in [-1.0, 1.0] or None if no execution tools were used.
    """
    exec_tools = _filter_execution_calls(tool_calls)
    if not exec_tools:
        return None

    scores: list[float] = []
    for tc in exec_tools:
        output = tc.get("output_preview", "") or ""
        error = tc.get("error", "") or ""
        success = tc.get("success", False)

        if success:
            if _has_test_results(output):
                scores.append(_score_test_output(output))
            elif _has_exit_code_zero(output):
                scores.append(1.0)
            else:
                scores.append(0.8)
        else:
            if _is_compilation_error(error + output):
                scores.append(-1.0)
            elif _is_test_failure(error + output):
                scores.append(-0.5)
            else:
                scores.append(-0.7)

    return round(sum(scores) / len(scores), 2) if scores else None


_EXEC_TOOL_NAMES = frozenset({
    "execute", "execute_command", "run_command", "bash", "shell",
})


def _filter_execution_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tc for tc in tool_calls if tc.get("tool_name", "") in _EXEC_TOOL_NAMES]


def _has_test_results(output: str) -> bool:
    return bool(re.search(r"(\d+)\s+(passed|failed|error)", output, re.IGNORECASE)
                or re.search(r"(PASS|FAIL|OK)\b", output)
                or "test" in output.lower() and ("pass" in output.lower() or "fail" in output.lower()))


def _score_test_output(output: str) -> float:
    """Extract pass/fail counts and compute a score."""
    m = re.search(r"(\d+)\s+passed", output, re.IGNORECASE)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+)\s+failed", output, re.IGNORECASE)
    failed = int(m.group(1)) if m else 0

    total = passed + failed
    if total == 0:
        if "PASS" in output or "OK" in output:
            return 1.0
        if "FAIL" in output:
            return -0.5
        return 0.5

    return round((passed - failed) / total, 2)


def _has_exit_code_zero(output: str) -> bool:
    return "exit code: 0" in output.lower() or "exit code 0" in output.lower()


def _is_compilation_error(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in (
        "syntaxerror", "compileerror", "compile error",
        "indentationerror", "nameerror", "typeerror",
        "cannot find module", "module not found",
    ))


def _is_test_failure(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ("assertionerror", "failed", "failure", "error"))


# ─── Feature C: Task-Type-Aware Verification ──────────────────────────────

_TASK_TYPE_PATTERNS: dict[str, list[str]] = {
    "coding": [
        r"write.*code", r"implement", r"create.*function", r"add.*test",
        r"fix.*bug", r"build.*module", r"create.*class",
        r"write.*script", r"write.*program", r"create.*api",
        r"write.*function", r"function.*that", r"def\s+\w+",
        r"class\s+\w+", r"import\s+", r"sort.*list",
    ],
    "file": [
        r"create.*file", r"move.*file", r"rename", r"organize",
        r"copy.*file", r"delete.*file", r"create.*directory",
    ],
    "search": [
        r"find.*all", r"search.*for", r"list.*all", r"how many",
        r"where.*is", r"what.*is", r"analyze", r"summarize",
    ],
    "refactor": [
        r"refactor", r"rename.*across", r"update.*imports",
        r"migrate", r"convert.*to",
    ],
}


def detect_task_type(task: str) -> str:
    """Auto-detect task type from the task description.

    Returns one of: "coding", "file", "search", "refactor", "general".
    """
    t = task.lower()
    scores: dict[str, int] = {}
    for task_type, patterns in _TASK_TYPE_PATTERNS.items():
        scores[task_type] = sum(1 for p in patterns if re.search(p, t))

    if not scores or max(scores.values()) == 0:
        return "general"

    return max(scores, key=lambda k: scores[k])


def verify_task_outcome(
    task_type: str,
    turns: list[dict[str, Any]],
    outcome: str,
) -> dict[str, Any]:
    """Verify task outcome using task-type-specific heuristics.

    Returns a dict with:
      - verified_score: float [-1.0 to 1.0] or None if no objective signal
      - confidence: "high" | "medium" | "low"
      - method: description of how the score was computed
    """
    if task_type == "coding":
        return _verify_coding(turns, outcome)
    elif task_type == "file":
        return _verify_file(turns, outcome)
    elif task_type == "refactor":
        return _verify_refactor(turns, outcome)
    else:
        return _verify_general(turns, outcome)


def _verify_coding(turns: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    """Coding tasks: use test results and exit codes."""
    all_exec_calls = []
    for t in turns:
        for tc in t.get("tool_calls", []):
            if tc.get("tool_name", "") in _EXEC_TOOL_NAMES:
                all_exec_calls.append(tc)

    if not all_exec_calls:
        return {"verified_score": None, "confidence": "low", "method": "no_execution_tools"}

    last_exec = all_exec_calls[-1]
    score = compute_deterministic_score([last_exec])

    if score is not None and _has_test_results(last_exec.get("output_preview", "")):
        return {"verified_score": score, "confidence": "high", "method": "test_results"}
    elif score is not None:
        return {"verified_score": score, "confidence": "medium", "method": "exit_code"}

    return {"verified_score": None, "confidence": "low", "method": "no_objective_signal"}


def _verify_file(turns: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    """File tasks: check if file operations succeeded."""
    file_ops = []
    for t in turns:
        for tc in t.get("tool_calls", []):
            if tc.get("tool_name", "") in ("write_file", "edit_file", "create_file"):
                file_ops.append(tc)

    if not file_ops:
        return {"verified_score": None, "confidence": "low", "method": "no_file_ops"}

    succeeded = sum(1 for op in file_ops if op.get("success", False))
    rate = succeeded / len(file_ops) if file_ops else 0

    return {
        "verified_score": round(rate * 2 - 1, 2),  # map [0,1] to [-1,1]
        "confidence": "medium",
        "method": f"file_ops_{succeeded}/{len(file_ops)}",
    }


def _verify_refactor(turns: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    """Refactoring tasks: check edits + optional test run."""
    edits = []
    test_results = []
    for t in turns:
        for tc in t.get("tool_calls", []):
            name = tc.get("tool_name", "")
            if name in ("edit_file", "write_file"):
                edits.append(tc)
            if name in _EXEC_TOOL_NAMES:
                test_results.append(tc)

    if not edits:
        return {"verified_score": None, "confidence": "low", "method": "no_edits"}

    edit_success = sum(1 for e in edits if e.get("success", False)) / len(edits)

    if test_results:
        last_test = test_results[-1]
        test_score = compute_deterministic_score([last_test])
        if test_score is not None:
            combined = round((edit_success + (test_score + 1) / 2) / 2 * 2 - 1, 2)
            return {"verified_score": combined, "confidence": "high", "method": "edits_plus_tests"}

    return {
        "verified_score": round(edit_success * 2 - 1, 2),
        "confidence": "medium",
        "method": f"edits_only_{sum(1 for e in edits if e.get('success'))}/{len(edits)}",
    }


def _verify_general(turns: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    """General tasks: fall back to heuristic scoring."""
    det = compute_deterministic_score(
        [tc for t in turns for tc in t.get("tool_calls", [])]
    )
    if det is not None:
        return {"verified_score": det, "confidence": "medium", "method": "execution_heuristic"}
    return {"verified_score": None, "confidence": "low", "method": "no_objective_signal"}


# ─── Feature F: Adaptive Rethink Threshold ────────────────────────────────

_BASE_THRESHOLD = 3
_MIN_THRESHOLD = 2
_MAX_THRESHOLD = 6


def compute_adaptive_rethink_threshold(
    task_type: str,
    current_turn: int,
    total_tool_count: int,
) -> int:
    """Compute a dynamic rethink threshold based on task complexity.

    Simple tasks (search, file) get a lower threshold (trigger rethink sooner).
    Complex tasks (coding, refactor) get a higher threshold (more patience).
    Threshold decreases as the run progresses (less patience late in run).
    """
    complexity_bonus = {
        "coding": 2,
        "refactor": 2,
        "general": 1,
        "search": 0,
        "file": 0,
    }.get(task_type, 1)

    threshold = _BASE_THRESHOLD + complexity_bonus

    # Late in the run: decrease threshold (we're running out of patience)
    if current_turn > 50:
        threshold = _MIN_THRESHOLD
    elif current_turn > 20:
        threshold = max(threshold - 1, _MIN_THRESHOLD)

    return min(threshold, _MAX_THRESHOLD)
