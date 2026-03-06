"""Structured trajectory logging for ClawAgents.

Records every agent turn as NDJSON — one line per turn, one file per run.
Storage: .clawagents/trajectories/{run_id}.jsonl

Enable via create_claw_agent(trajectory=True) or CLAW_TRAJECTORY=1 in .env.

Scoring (inspired by CUDA-Agent discrete reward bands):
  Turn score: weighted by tool type — execution tools count double.
  Run score:  -1 (failed), 0 (ambiguous), +1 (completed),
              +2 (efficient), +3 (clean first-attempt success).
  Quality:    "clean" / "noisy" / "failed" — for trajectory filtering.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRAJECTORIES_DIR = Path.cwd() / ".clawagents" / "trajectories"

# Tools whose results are not meaningful reward signals (gameable / no side effects)
_SCORELESS_TOOLS: frozenset[str] = frozenset({
    "think", "todolist", "todo_write", "todo_read",
    "use_skill", "ask_user",
})

# Tools whose success/failure carries extra weight (real execution with side effects)
_HIGH_WEIGHT_TOOLS: frozenset[str] = frozenset({
    "execute", "execute_command", "run_command", "bash",
})


@dataclass
class ToolCallRecord:
    tool_name: str
    args: dict[str, Any]
    success: bool
    output_preview: str
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class TurnRecord:
    run_id: str
    turn_index: int
    timestamp: float
    response_text: str
    model: str
    tokens_used: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    score: int = 0            # weighted turn score
    cumulative_score: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_id: str
    task: str
    model: str
    total_turns: int
    total_tool_calls: int
    tool_success_rate: float
    turn_scores: list[int]
    outcome: str              # "success" | "error" | "cancelled" | "max_iterations"
    aggregate_score: float
    run_score: int            # discrete band: -1, 0, +1, +2, +3
    quality: str              # "clean" | "noisy" | "failed"
    mid_run_failures: int     # how many turns had failures before final success
    duration_s: float
    tokens_total: int
    trajectory_file: str


def _score_turn(calls: list[ToolCallRecord]) -> int:
    """Score a turn based on its tool calls, weighting by tool importance.

    - Gameable tools (think, todolist, etc.) are ignored.
    - Execution tools (execute, bash, etc.) count double.
    - Other tools count normally (+1 success, -1 failure).
    """
    if not calls:
        return 0

    total = 0
    scored_count = 0
    for tc in calls:
        if tc.tool_name in _SCORELESS_TOOLS:
            continue
        scored_count += 1
        weight = 2 if tc.tool_name in _HIGH_WEIGHT_TOOLS else 1
        total += weight if tc.success else -weight

    if scored_count == 0:
        return 0
    # Normalize to -1 / 0 / +1 range
    if total > 0:
        return 1
    elif total < 0:
        return -1
    return 0


def _compute_run_score(
    outcome: str,
    turns: list[TurnRecord],
    mid_run_failures: int,
) -> int:
    """Discrete reward band inspired by CUDA-Agent.

    -1: task failed (error, cancelled, max_iterations)
     0: ambiguous outcome
    +1: task completed
    +2: task completed efficiently (≤ median effort, i.e. few failures)
    +3: task completed cleanly (zero mid-run failures)
    """
    if outcome in ("error", "cancelled", "max_iterations"):
        return -1
    if outcome == "done" and not turns:
        return 0

    if outcome in ("done", "success"):
        if mid_run_failures == 0:
            return 3   # clean first-attempt success
        scored_turns = [t for t in turns if t.score != 0]
        if not scored_turns:
            return 1
        failure_rate = mid_run_failures / len(scored_turns)
        if failure_rate <= 0.2:
            return 2   # efficient — at most 20% of scored turns had failures
        return 1       # completed but with significant mid-run failures

    return 0


def _compute_quality(run_score: int, mid_run_failures: int, total_turns: int) -> str:
    """Classify trajectory quality for RFT-style filtering.

    "clean":  worth learning from (high run_score, minimal failures).
    "noisy":  succeeded but with excessive retries — risky to learn from.
    "failed": task didn't succeed — only useful as negative examples.
    """
    if run_score <= 0:
        return "failed"
    if run_score >= 2:
        return "clean"
    # run_score == 1: completed but lots of failures
    if total_turns > 0 and mid_run_failures / max(total_turns, 1) > 0.4:
        return "noisy"
    return "clean"


class TrajectoryRecorder:
    """Writes per-turn NDJSON records to .clawagents/trajectories/{run_id}.jsonl."""

    def __init__(self, task: str, model: str = "", response_chars: int = 500):
        self.run_id = uuid.uuid4().hex[:12]
        self.task = task
        self.model = model
        self._response_chars = response_chars
        self._turns: list[TurnRecord] = []
        self._cumulative_score = 0
        self._total_tokens = 0
        self._mid_run_failures = 0
        self._path = _TRAJECTORIES_DIR / f"{self.run_id}.jsonl"
        self._t0 = time.monotonic()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            _TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("Failed to create trajectories directory", exc_info=True)

    def record_turn(
        self,
        response_text: str,
        model: str,
        tokens_used: int,
        tool_calls: list[ToolCallRecord] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnRecord:
        if model and not self.model:
            self.model = model

        calls = tool_calls or []
        score = _score_turn(calls)

        if score < 0:
            self._mid_run_failures += 1

        self._cumulative_score += score
        self._total_tokens += tokens_used

        turn = TurnRecord(
            run_id=self.run_id,
            turn_index=len(self._turns),
            timestamp=time.time(),
            response_text=response_text[:self._response_chars],
            model=model,
            tokens_used=tokens_used,
            tool_calls=calls,
            score=score,
            cumulative_score=self._cumulative_score,
            metadata=metadata or {},
        )
        self._turns.append(turn)
        self._write_turn(turn)
        return turn

    def _write_turn(self, turn: TurnRecord) -> None:
        try:
            data = asdict(turn)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception:
            logger.debug("Failed to write trajectory turn", exc_info=True)

    def finalize(self, outcome: str) -> RunSummary:
        elapsed = time.monotonic() - self._t0
        tool_total = sum(len(t.tool_calls) for t in self._turns)
        tool_ok = sum(
            1 for t in self._turns for tc in t.tool_calls if tc.success
        )
        scores = [t.score for t in self._turns]

        run_score = _compute_run_score(outcome, self._turns, self._mid_run_failures)
        quality = _compute_quality(run_score, self._mid_run_failures, len(self._turns))

        summary = RunSummary(
            run_id=self.run_id,
            task=self.task[:200],
            model=self.model,
            total_turns=len(self._turns),
            total_tool_calls=tool_total,
            tool_success_rate=tool_ok / tool_total if tool_total else 1.0,
            turn_scores=scores,
            outcome=outcome,
            aggregate_score=self._cumulative_score / len(self._turns) if self._turns else 0.0,
            run_score=run_score,
            quality=quality,
            mid_run_failures=self._mid_run_failures,
            duration_s=round(elapsed, 2),
            tokens_total=self._total_tokens,
            trajectory_file=str(self._path),
        )

        self._write_summary(summary)
        return summary

    def _write_summary(self, summary: RunSummary) -> None:
        try:
            runs_file = _TRAJECTORIES_DIR / "runs.jsonl"
            data = asdict(summary)
            with open(runs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception:
            logger.debug("Failed to write run summary", exc_info=True)

    @property
    def turns(self) -> list[TurnRecord]:
        return list(self._turns)
