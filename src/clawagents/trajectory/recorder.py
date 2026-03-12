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

def _get_trajectories_dir() -> Path:
    return Path.cwd() / ".clawagents" / "trajectories"

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
    failure_type: str = ""    # "format" | "logic" | "" (Feature 3)


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
    observation_context: str = ""       # what agent saw before deciding (Feature 4)
    productivity_score: float = 0.0     # per-step productivity: -1.0 to 1.0 (Feature 4)
    deterministic_score: float | None = None  # objective score from exec tools (Feature A)
    prompt_token_count: int = 0         # tokens in prompt at this step (Feature E: RFT-ready)
    response_token_count: int = 0       # tokens in response at this step (Feature E: RFT-ready)
    thinking: str | None = None         # preserved <think> content (Feature H)
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
    format_failures: int = 0  # count of format-type failures (Feature 3)
    logic_failures: int = 0   # count of logic-type failures (Feature 3)
    has_mixed_outcomes: bool = False    # True if run had both successes and failures (Feature 1)
    finish_reason: str = ""   # why the run ended (Feature 4)
    task_type: str = ""       # auto-detected: "coding"|"file"|"search"|"refactor"|"general" (Feature C)
    verified_score: float | None = None  # deterministic score from tool outputs (Feature A)
    verified_confidence: str = ""        # "high"|"medium"|"low" (Feature A)
    verified_method: str = ""            # how the score was computed (Feature A)
    judge_score: int | None = None       # LLM-as-Judge score 0-3 (Feature G)
    judge_justification: str = ""        # LLM judge's reasoning (Feature G)
    duration_s: float = 0.0
    tokens_total: int = 0
    trajectory_file: str = ""


# ─── Feature 3: Format vs. Logic Failure Classification ───────────────────

_FORMAT_ERROR_PATTERNS = (
    "invalid json", "json decode", "json parse", "unexpected token",
    "missing required", "unknown tool", "unrecognized tool",
    "no tool named", "expected string", "expected number",
    "not a valid", "malformed", "syntax error in args",
    "missing parameter", "unknown parameter", "extra parameter",
)


def classify_failure(tool_name: str, error: str | None, output: str | None) -> str:
    """Classify a tool failure as 'format' or 'logic'.

    Format errors: bad JSON, wrong parameter names, unknown tool names.
    Logic errors: valid call but wrong approach (file not found, command failed, etc.).
    """
    if not error and not output:
        return "unknown"
    text = ((error or "") + " " + (output or "")).lower()
    for pattern in _FORMAT_ERROR_PATTERNS:
        if pattern in text:
            return "format"
    return "logic"


# ─── Feature 4: Per-Step Productivity Scoring ─────────────────────────────

def _compute_productivity(
    calls: list[ToolCallRecord],
    prev_cumulative: int,
    turn_index: int,
) -> float:
    """Compute a per-step productivity score from -1.0 to 1.0.

    Factors:
    - Did the tools succeed? (primary signal)
    - Is this a recovery from failure? (pivot bonus)
    - Are we making progress? (cumulative trend)
    """
    if not calls:
        return 0.0

    scored = [c for c in calls if c.tool_name not in _SCORELESS_TOOLS]
    if not scored:
        return 0.0

    successes = sum(1 for c in scored if c.success)
    failures = len(scored) - successes
    base = (successes - failures) / len(scored)  # -1.0 to 1.0

    # Pivot bonus: recovering from a negative cumulative score
    if prev_cumulative < 0 and base > 0:
        base = min(base + 0.2, 1.0)

    return round(base, 2)


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
        self._has_successes = False
        self._has_failures = False
        self._path = _get_trajectories_dir() / f"{self.run_id}.jsonl"
        self._t0 = time.monotonic()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            _get_trajectories_dir().mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("Failed to create trajectories directory", exc_info=True)

    def record_turn(
        self,
        response_text: str,
        model: str,
        tokens_used: int,
        tool_calls: list[ToolCallRecord] | None = None,
        metadata: dict[str, Any] | None = None,
        observation_context: str = "",
        prompt_token_count: int = 0,
        response_token_count: int = 0,
        thinking: str | None = None,
    ) -> TurnRecord:
        if model and not self.model:
            self.model = model

        calls = tool_calls or []
        score = _score_turn(calls)

        # Feature 3: classify failure types
        for tc in calls:
            if not tc.success and not tc.failure_type:
                tc.failure_type = classify_failure(tc.tool_name, tc.error, tc.output_preview)

        if score < 0:
            self._mid_run_failures += 1
        if score > 0:
            self._has_successes = True
        if score < 0:
            self._has_failures = True

        # Feature 4: per-step productivity
        productivity = _compute_productivity(calls, self._cumulative_score, len(self._turns))

        # Feature A: deterministic score from execution tools
        det_score = None
        try:
            from clawagents.trajectory.verifier import compute_deterministic_score
            call_dicts = [{"tool_name": c.tool_name, "success": c.success,
                           "output_preview": c.output_preview, "error": c.error}
                          for c in calls]
            det_score = compute_deterministic_score(call_dicts)
        except Exception:
            pass

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
            observation_context=observation_context[:300] if observation_context else "",
            productivity_score=productivity,
            deterministic_score=det_score,
            prompt_token_count=prompt_token_count,
            response_token_count=response_token_count,
            thinking=thinking[:500] if thinking else None,
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

        # Feature 3: count format vs. logic failures
        format_failures = sum(
            1 for t in self._turns for tc in t.tool_calls
            if not tc.success and tc.failure_type == "format"
        )
        logic_failures = sum(
            1 for t in self._turns for tc in t.tool_calls
            if not tc.success and tc.failure_type == "logic"
        )

        # Feature 1: mixed outcomes detection
        has_mixed = self._has_successes and self._has_failures

        # Feature 4: map outcome to finish_reason
        finish_reason_map = {
            "done": "success", "success": "success",
            "error": "error", "cancelled": "cancelled",
            "max_iterations": "max_iterations",
        }
        finish_reason = finish_reason_map.get(outcome, outcome)

        # Feature C: task-type detection + Feature A: deterministic verification
        task_type = ""
        verified_score = None
        verified_confidence = ""
        verified_method = ""
        try:
            from clawagents.trajectory.verifier import detect_task_type, verify_task_outcome
            task_type = detect_task_type(self.task)
            turn_dicts = [asdict(t) for t in self._turns]
            result = verify_task_outcome(task_type, turn_dicts, outcome)
            verified_score = result.get("verified_score")
            verified_confidence = result.get("confidence", "")
            verified_method = result.get("method", "")
        except Exception:
            logger.debug("Verification failed", exc_info=True)

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
            format_failures=format_failures,
            logic_failures=logic_failures,
            has_mixed_outcomes=has_mixed,
            finish_reason=finish_reason,
            task_type=task_type,
            verified_score=verified_score,
            verified_confidence=verified_confidence,
            verified_method=verified_method,
            duration_s=round(elapsed, 2),
            tokens_total=self._total_tokens,
            trajectory_file=str(self._path),
        )

        self._write_summary(summary)
        return summary

    def _write_summary(self, summary: RunSummary) -> None:
        try:
            runs_file = _get_trajectories_dir() / "runs.jsonl"
            data = asdict(summary)
            with open(runs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception:
            logger.debug("Failed to write run summary", exc_info=True)

        # Feature E: export RFT-ready transitions
        try:
            rft_file = _get_trajectories_dir() / f"{self.run_id}_rft.json"
            transitions = self.export_rft_transitions()
            rft_file.write_text(json.dumps({
                "run_id": self.run_id,
                "task": self.task,
                "model": self.model,
                "outcome": summary.outcome,
                "run_score": summary.run_score,
                "quality": summary.quality,
                "verified_score": summary.verified_score,
                "task_type": summary.task_type,
                "transitions": transitions,
            }, indent=2, default=str), encoding="utf-8")
        except Exception:
            logger.debug("Failed to write RFT transitions", exc_info=True)

    @property
    def turns(self) -> list[TurnRecord]:
        return list(self._turns)

    def export_rft_transitions(self) -> list[dict[str, Any]]:
        """Export turns as RFT-ready (observation, action, reward, done) transitions.

        Feature E: Structured format compatible with future Rejection Fine-Tuning
        pipelines. Each transition contains:
          - observation: what the agent saw (context + previous tool results)
          - action: what the agent did (response text + tool calls)
          - reward: per-step score (deterministic if available, else heuristic)
          - done: whether this was the final turn
          - metadata: model, tokens, timestamps for provenance
        """
        transitions: list[dict[str, Any]] = []
        for i, turn in enumerate(self._turns):
            is_last = (i == len(self._turns) - 1)
            reward = turn.deterministic_score if turn.deterministic_score is not None else turn.productivity_score

            transitions.append({
                "observation": turn.observation_context,
                "action": {
                    "response_text": turn.response_text,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "args": tc.args,
                            "success": tc.success,
                            "output_preview": tc.output_preview,
                            "failure_type": tc.failure_type,
                        }
                        for tc in turn.tool_calls
                    ],
                },
                "reward": round(reward, 3) if reward is not None else 0.0,
                "done": is_last,
                "step_index": turn.turn_index,
                "timestamp": turn.timestamp,
                "model": turn.model,
                "prompt_tokens": turn.prompt_token_count,
                "response_tokens": turn.response_token_count,
                "heuristic_score": turn.score,
                "cumulative_score": turn.cumulative_score,
            })
        return transitions


def prune_trajectories(max_age_days: int = 30) -> int:
    """Delete trajectory files older than max_age_days. Returns count of files removed."""
    import time as _time
    traj_dir = _get_trajectories_dir()
    if not traj_dir.exists():
        return 0
    cutoff = _time.time() - max_age_days * 86400
    removed = 0
    for f in traj_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    return removed
