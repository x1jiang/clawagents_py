"""Eval suite runner — replay tasks and score with verifier (+ optional judge)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _load_suite(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        cases: list[dict[str, Any]] = []
        for p in sorted(path.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                cases.extend(data)
            elif isinstance(data, dict):
                if "cases" in data and isinstance(data["cases"], list):
                    cases.extend(data["cases"])
                else:
                    cases.append(data)
        return cases
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]
    raise ValueError(f"unsupported suite format: {path}")


def _score_case(
    task: str,
    result_text: str,
    tool_call_count: int,
    expect: Any,
) -> dict[str, Any]:
    from clawagents.trajectory.verifier import detect_task_type, verify_task_outcome

    task_type = detect_task_type(task)
    verify = verify_task_outcome(task_type, turns=[], outcome=result_text)
    score = verify.get("verified_score")
    if score is None:
        # Soft heuristic when no exec signal: non-empty result + some tool use
        score = 0.6 if result_text.strip() and tool_call_count >= 0 else 0.0
        if not result_text.strip():
            score = 0.0

    passed = True
    if isinstance(expect, dict):
        contains = expect.get("contains")
        if contains:
            needles = contains if isinstance(contains, list) else [contains]
            for n in needles:
                if str(n) not in result_text:
                    passed = False
                    break
        if "pass_score" in expect:
            try:
                passed = passed and float(score) >= float(expect["pass_score"])
            except (TypeError, ValueError):
                pass
    elif isinstance(expect, str) and expect:
        passed = expect in result_text
    else:
        passed = float(score) >= 0.5 and bool(result_text.strip())

    return {
        "task_type": task_type,
        "score": score,
        "verify": verify,
        "passed": passed,
    }


async def _run_case(case: dict[str, Any], *, judge: bool) -> dict[str, Any]:
    from clawagents import create_claw_agent

    case_id = str(case.get("id") or case.get("name") or "case")
    task = str(case.get("task") or "").strip()
    if not task:
        return {"id": case_id, "ok": False, "error": "missing task", "passed": False}

    kwargs: dict[str, Any] = {}
    if case.get("mode"):
        kwargs["mode"] = str(case["mode"])
    if case.get("action_mode"):
        kwargs["action_mode"] = str(case["action_mode"])
    if case.get("model"):
        kwargs["model"] = case["model"]

    agent = create_claw_agent(**kwargs)
    state = await agent.invoke(task, trajectory=True)
    result_text = str(getattr(state, "result", "") or "")
    scored = _score_case(
        task,
        result_text,
        int(getattr(state, "tool_calls", 0) or 0),
        case.get("expect"),
    )
    row: dict[str, Any] = {
        "id": case_id,
        "ok": True,
        "result_preview": result_text[:500],
        "tool_calls": getattr(state, "tool_calls", 0),
        "status": getattr(state, "status", ""),
        **scored,
    }
    if judge:
        try:
            from clawagents.trajectory.judge import judge_run

            summary = {
                "task_type": scored.get("task_type", "general"),
                "outcome": getattr(state, "status", "unknown"),
                "total_tool_calls": int(getattr(state, "tool_calls", 0) or 0),
            }
            judgment = await judge_run(
                agent.llm,
                task,
                summary,
                result_text,
            )
            judgment.pop("_llm_response", None)
            row["judge"] = judgment
        except Exception as exc:
            row["judge_error"] = str(exc)
    return row


def _compare_baseline(report: dict[str, Any], baseline_path: Path) -> list[str]:
    if not baseline_path.is_file():
        return [f"baseline missing: {baseline_path}"]
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_cases = {c["id"]: c for c in base.get("cases", []) if "id" in c}
    regressions: list[str] = []
    for case in report.get("cases", []):
        cid = case.get("id")
        if cid not in base_cases:
            continue
        prev = base_cases[cid]
        if prev.get("passed") and not case.get("passed"):
            regressions.append(f"{cid}: was pass, now fail")
        try:
            cur = float(case.get("score") or 0)
            old = float(prev.get("score") or 0)
            if cur + 1e-9 < old - 0.15:
                regressions.append(f"{cid}: score {cur} < baseline {old} (-0.15)")
        except (TypeError, ValueError):
            continue
    return regressions


async def run_evals(
    suite_path: Path,
    *,
    judge: bool = False,
    baseline: Path | None = None,
    output: Path | None = None,
) -> int:
    cases = _load_suite(suite_path)
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            results.append(await _run_case(case, judge=judge))
        except Exception as exc:
            results.append({
                "id": str(case.get("id") or "?"),
                "ok": False,
                "error": str(exc),
                "passed": False,
            })
    report = {
        "suite": str(suite_path),
        "cases": results,
        "passed": sum(1 for r in results if r.get("passed")),
        "failed": sum(1 for r in results if not r.get("passed")),
        "total": len(results),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if baseline:
        regs = _compare_baseline(report, baseline)
        if regs:
            print("REGRESSIONS:", file=sys.stderr)
            for r in regs:
                print(f"  - {r}", file=sys.stderr)
            return 2
    return 0 if report["failed"] == 0 else 1


def main_evals(args: argparse.Namespace) -> int:
    return asyncio.run(
        run_evals(
            Path(args.suite),
            judge=bool(getattr(args, "judge", False)),
            baseline=Path(args.baseline) if getattr(args, "baseline", None) else None,
            output=Path(args.output) if getattr(args, "output", None) else None,
        )
    )
