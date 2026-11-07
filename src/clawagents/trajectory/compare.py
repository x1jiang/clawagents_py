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

import logging
import re
from typing import Any, Optional

from clawagents.trajectory.harness_surface import (
    assign_anon_ids,
    build_blind_selector_prompt,
    parse_choice_reply,
)

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|FIXME)\b|<placeholder>", re.IGNORECASE)


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
    blind_select: bool = False,
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
    selection_why = ""
    comparison_method = best.get("scoring_method", "composite")

    if blind_select:
        sel = await select_best_candidate_blind(task=task, scored_samples=scored, llm=llm)
        if sel.get("ok") and sel.get("chosen_index") is not None:
            chosen_idx = int(sel["chosen_index"])
            matched = next((s for s in scored if s["index"] == chosen_idx), None)
            if matched is not None:
                best = matched
                best_idx = chosen_idx
                selection_why = sel.get("why", "")
                comparison_method = "blind_selector"

    # If learn is enabled, extract causal selection rationale when available
    if learn and (selection_why or best.get("trajectory_file")):
        try:
            from clawagents.trajectory.lessons import save_lessons

            note = (
                f"- [DOM] Selection rationale ({comparison_method}): {selection_why}"
                if selection_why
                else f"- [INV] Verified candidate {best_idx} selected ({best['composite_score']:.2f} across {n_samples} samples)"
            )
            save_lessons(note, task, "success", model="")
        except Exception:
            pass

    return {
        "best_result": best.get("result", ""),
        "best_score": best["composite_score"],
        "best_index": best_idx,
        "all_scores": scored,
        "comparison_method": comparison_method,
        "selection_why": selection_why,
        "n_samples": n_samples,
    }


async def select_best_candidate_blind(
    task: str,
    scored_samples: list[dict[str, Any]],
    llm: Any,
    *,
    task_id: str = "task",
) -> dict[str, Any]:
    """Harness² blind anonymized artifact selector (COVERAGE -> CORRECTNESS -> FIDELITY).

    Drops failed/empty candidates first, returns trivially when only 1 valid candidate
    exists (zero LLM overhead), and otherwise asks the selector LLM to pick among
    shuffled anonymous IDs (`A`, `B`, ...).
    """
    valid = [
        s
        for s in scored_samples
        if s.get("status") == "done" and str(s.get("result") or "").strip()
    ]
    if not valid:
        return {"ok": False, "error": "no candidate produced a deliverable"}
    if len(valid) == 1:
        only = valid[0]
        return {
            "ok": True,
            "trivial": True,
            "chosen_id": "A",
            "chosen_index": only["index"],
            "why": "trivial pool: single valid deliverable; selected without a selector call",
        }

    labels = [f"cand_{s['index']}" for s in valid]
    by_label = {f"cand_{s['index']}": s for s in valid}
    id_map = assign_anon_ids(labels, task_id=task_id, k=len(valid), mode="bon")
    anon_artifacts = {
        anon_id: str(by_label[lbl].get("result") or "").strip()
        for anon_id, lbl in id_map.items()
    }
    prompt = build_blind_selector_prompt(task, anon_artifacts)
    try:
        from clawagents.providers.llm import LLMMessage

        resp = await llm.chat(
            [
                LLMMessage(
                    role="system",
                    content="You are an impartial deliverable selector. Judge strictly by COVERAGE, CORRECTNESS, and FIDELITY.",
                ),
                LLMMessage(role="user", content=prompt),
            ]
        )
        anon_id, chosen_label, why = parse_choice_reply(resp.content or "", id_map)
        return {
            "ok": True,
            "trivial": False,
            "chosen_id": anon_id,
            "chosen_index": by_label[chosen_label]["index"],
            "id_map": id_map,
            "why": why,
        }
    except Exception as exc:
        fallback = max(valid, key=lambda s: s["composite_score"])
        return {
            "ok": True,
            "fallback": True,
            "chosen_id": "",
            "chosen_index": fallback["index"],
            "id_map": id_map,
            "why": f"fallback to highest deterministic/composite score ({exc})",
        }


def _score_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score each sample with deliverable fidelity checks and deterministic tool verification."""
    scored = []
    for s in samples:
        score = 0.0
        method = "status"
        result_text = str(s.get("result") or "").strip()
        delivered = bool(result_text) and s.get("status") == "done"

        if delivered:
            score = 0.6
            # Penalize unresolved placeholders / TODOs (Harness² FIDELITY criterion)
            if _PLACEHOLDER_RE.search(result_text):
                score -= 0.15
        elif s.get("status") == "done":
            # Finished with empty deliverable
            score = 0.1
        elif s.get("status") == "error":
            score = -1.0

        iterations = s.get("iterations", 0)
        if iterations > 0 and delivered:
            # Minor tie-breaker only (0.05 max) so thorough multi-step runs are not
            # penalized below shallow 1-turn completions.
            efficiency = max(0.0, 1.0 - (iterations / 100.0))
            score += efficiency * 0.05
            method = "deliverable+efficiency"

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
                    score = det + (0.1 if delivered and not _PLACEHOLDER_RE.search(result_text) else 0.0)
                    method = "deterministic"
            except Exception:
                pass

        scored.append({
            **s,
            "delivered": delivered,
            "composite_score": round(score, 3),
            "scoring_method": method,
        })

    return scored
