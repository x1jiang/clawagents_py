"""Feature B: Multi-sample comparison (GRPO-inspired).

Runs the same task N times and picks the best result based on objective scoring.
Inspired by SkyRL's Group Relative Policy Optimization where multiple completions
are generated and ranked by reward to compute advantages.

Usage:
    result = await compare_samples(
        task="Write a function to sort a list",
        llm=llm, tools=tools,
        n_samples=3,  # run 3 times, pick best
    )

The best sample is selected by:
  1. Verified score (deterministic, from tool outputs) — highest confidence
  2. Run score (heuristic, from trajectory analysis) — medium confidence
  3. Success rate — fallback
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def compare_samples(
    task: str,
    llm: Any,
    tools: Any = None,
    system_prompt: Optional[str] = None,
    n_samples: int = 3,
    max_iterations: int = 200,
    streaming: bool = False,
    context_window: int = 1_000_000,
    on_event: Any = None,
    use_native_tools: bool = True,
    rethink: bool = True,
    learn: bool = False,
    preview_chars: int = 120,
    response_chars: int = 500,
) -> dict[str, Any]:
    """Run a task N times and return the best result.

    Returns a dict with:
      - best_result: str (the best answer)
      - best_score: float (score of the winning sample)
      - best_index: int (which sample won)
      - all_scores: list of dicts with each sample's score info
      - comparison_method: str (how the winner was chosen)
    """
    from clawagents.graph.agent_loop import run_agent_graph

    samples: list[dict[str, Any]] = []

    for i in range(n_samples):
        try:
            state = await run_agent_graph(
                task=task,
                llm=llm,
                tools=tools,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                streaming=streaming,
                context_window=context_window,
                on_event=on_event,
                use_native_tools=use_native_tools,
                trajectory=True,
                rethink=rethink,
                learn=False,  # don't learn during comparison; learn from the best one
                preview_chars=preview_chars,
                response_chars=response_chars,
            )
            samples.append({
                "index": i,
                "result": state.result,
                "status": state.status,
                "iterations": state.iterations,
                "tool_calls": state.tool_calls,
                "trajectory_file": state.trajectory_file,
            })
        except Exception as e:
            logger.debug("Sample %d failed: %s", i, e)
            samples.append({
                "index": i,
                "result": str(e),
                "status": "error",
                "iterations": 0,
                "tool_calls": 0,
                "trajectory_file": "",
            })

    scored = _score_samples(samples)

    best = max(scored, key=lambda s: s["composite_score"])
    best_idx = best["index"]

    # If learn is enabled, extract lessons from the best run only
    if learn and best.get("trajectory_file"):
        try:
            from clawagents.trajectory.lessons import save_lessons
            save_lessons(
                f"- This approach scored {best['composite_score']:.2f} out of {n_samples} samples",
                task, "success", model="",
            )
        except Exception:
            pass

    return {
        "best_result": best.get("result", ""),
        "best_score": best["composite_score"],
        "best_index": best_idx,
        "all_scores": scored,
        "comparison_method": best.get("scoring_method", "composite"),
        "n_samples": n_samples,
    }


def _score_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score each sample and add scoring metadata."""
    scored = []
    for s in samples:
        score = 0.0
        method = "status"

        if s.get("status") == "done":
            score = 0.5
        elif s.get("status") == "error":
            score = -1.0

        iterations = s.get("iterations", 0)
        tool_calls = s.get("tool_calls", 0)
        if iterations > 0 and s.get("status") == "done":
            efficiency = max(0, 1.0 - (iterations / 100))
            score += efficiency * 0.3
            method = "efficiency"

        traj_file = s.get("trajectory_file", "")
        if traj_file:
            try:
                from clawagents.trajectory.verifier import compute_deterministic_score
                import json
                from pathlib import Path

                lines = Path(traj_file).read_text(encoding="utf-8").strip().split("\n")
                all_calls = []
                for line in lines:
                    try:
                        turn = json.loads(line)
                        for tc in turn.get("tool_calls", []):
                            all_calls.append(tc)
                    except json.JSONDecodeError:
                        pass
                det = compute_deterministic_score(all_calls)
                if det is not None:
                    score = det  # override with deterministic score
                    method = "deterministic"
            except Exception:
                pass

        scored.append({
            **s,
            "composite_score": round(score, 3),
            "scoring_method": method,
        })

    return scored
