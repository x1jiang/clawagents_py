#!/usr/bin/env python3
"""Matched ClawAgents task benchmark; uses existing SDK and isolated workspaces.

Example: .venv/bin/python scripts/benchmark_meta_glimmer.py --repeats 3 --output results.json
Requires OPENAI_API_KEY for Luna. Optional LUNA_BASE_URL/LUNA_API_KEY support gateways.
No judge model: checks read answers, generated JSON, and executable code independently.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

from clawagents.agent import create_claw_agent
from clawagents.config.config import load_config
from clawagents.harness_profiles import resolve_harness_profile

TASKS = {
    "read": "Read config.json and report the release version and retry count. Do not modify files.",
    "aggregate": "Read orders.json. Sum amount for rows with status paid, grouped by customer. Write totals.json as a JSON object mapping customer to total. Do not include unpaid orders. Verify your output.",
    "repair": "Read ranges.py. Fix inclusive_sum so it sums every integer from start through end inclusive, returning 0 when start exceeds end. Run a Python check covering normal, negative, single-value, and reversed ranges. Keep the function name and signature.",
}
FIXTURES = {
    "config.json": '{"release":"3.17.9","retry_count":7}\n',
    "orders.json": json.dumps([
        {"customer": "Ada", "amount": 17, "status": "paid"},
        {"customer": "Lin", "amount": 9, "status": "paid"},
        {"customer": "Ada", "amount": 5, "status": "paid"},
        {"customer": "Lin", "amount": 800, "status": "pending"},
        {"customer": "Sam", "amount": 1, "status": "cancelled"},
    ]),
    "ranges.py": 'def inclusive_sum(start, end):\n    return sum(range(start, end))\n',
}


def check(task: str, root: Path, result: str) -> bool:
    if task == "read":
        return "3.17.9" in result and re.search(r"(?<![\d.])7(?![\d.])", result) is not None and (root / "config.json").read_text() == FIXTURES["config.json"]
    if task == "aggregate":
        try:
            return json.loads((root / "totals.json").read_text()) == {"Ada": 22, "Lin": 9}
        except (OSError, ValueError):
            return False
    # Verify hidden cases independently of the model's self-reported check.
    code = "from ranges import inclusive_sum as f; assert all(f(a,b)==sum(range(a,b+1)) for a in range(-8,9) for b in range(-8,9))"
    return subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, timeout=10).returncode == 0


async def run_one(arm: str, task: str, repeat: int, args) -> dict:
    row = {"arm": arm, "task": task, "repeat": repeat, "passed": False}
    with tempfile.TemporaryDirectory(prefix="claw-meta-bench-") as temp:
        root = Path(temp)
        for name, content in FIXTURES.items():
            (root / name).write_text(content)
        kwargs = dict(workspace=root, skills=[], memory=[], streaming=True,
                      max_tokens=4096, max_iterations=12, temperature=0,
                      trajectory=False, rethink=False, learn=False,
                      mode="ci", tool_discovery=False,
                      context_window=196608,
                      features={"background_memory": False, "core_memory": False,
                                "memory_bank": False, "memory_dream": False,
                                "smart_memory": False, "context_ledger": False,
                                "fact_store": False, "repo_map_inject": False})
        if arm.startswith("glimmer"):
            kwargs.update(profile="meta")
        else:
            kwargs.update(model="gpt-5.6-luna", api_key=args.luna_key,
                          base_url=args.luna_base, reasoning_effort="medium")
        # Recreate pre-integration Glimmer: full tool surface and legacy string coercion.
        def without_glimmer(model, explicit=None):
            if model and "muse-glimmer-30b" in model.lower():
                return None
            return resolve_harness_profile(model, explicit)
        context = ExitStack()
        agent = None
        started = time.perf_counter()
        try:
            with context:
                if arm == "glimmer-baseline":
                    context.enter_context(patch("clawagents.harness_profiles.resolve_harness_profile", without_glimmer))
                    from clawagents.tools.validate import _COERCERS
                    context.enter_context(patch.dict(_COERCERS, {"string": lambda value: str(value) if value is not None else None}))
                agent = create_claw_agent(**kwargs)
                row["model"] = agent.llm.model
                row["wire_api"] = "responses" if agent.llm._should_use_responses(True) else "chat_completions"
                row["active_tool_count"] = len(agent.tools.list())
                result = await asyncio.wait_for(agent.invoke(TASKS[task], max_iterations=12), timeout=args.timeout)
                row.update(status=result.status, iterations=result.iterations,
                           tool_calls=result.tool_calls, result=result.result,
                           usage=asdict(result.usage))
                row["artifact_correct"] = check(task, root, result.result)
                row["budget_exhausted"] = result.result.startswith("Reached maximum of ")
                row["passed"] = (
                    result.status == "done" and result.tool_calls > 0
                    and row["artifact_correct"] and not row["budget_exhausted"]
                )
        except Exception as exc:
            # Avoid exception strings that might echo a credential or request body.
            row.update(status="error", error_type=type(exc).__name__)
        finally:
            row["seconds"] = round(time.perf_counter() - started, 4)
            if agent is not None:
                await agent.llm.client.close()
        return row


def summarize(rows):
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        items = [r for r in rows if r["arm"] == arm]
        out[arm] = {
            "passed": sum(r["passed"] for r in items), "runs": len(items),
            "median_seconds": round(statistics.median(r["seconds"] for r in items), 3),
            "total_seconds": round(sum(r["seconds"] for r in items), 3),
            "total_tool_calls": sum(r.get("tool_calls", 0) for r in items),
            "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in items),
            "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in items),
        }
    return out


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=["glimmer-baseline", "glimmer-tuned", "luna"], default=["glimmer-baseline", "glimmer-tuned", "luna"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    config = load_config()
    args.luna_key = os.getenv("LUNA_API_KEY") or config.openai_api_key
    args.luna_base = os.getenv("LUNA_BASE_URL") or config.openai_base_url or "https://api.openai.com/v1"
    if "luna" in args.arms and not args.luna_key:
        parser.error("OPENAI_API_KEY or LUNA_API_KEY is required for Luna")
    # Prevent ambient learning/advisor state from affecting the matched runs.
    os.environ.pop("ADVISOR_MODEL", None)
    os.environ["CLAWAGENTS_DOTENV_OVERRIDE"] = "0"
    rows = []
    spec = {"tasks": TASKS, "fixtures": FIXTURES}
    payload = {"started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "task_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "method": "Serial randomized paired order; fresh workspace/session each run; same registered tools, context cap, max output and iterations; tuned Glimmer and Luna use core active tools, baseline Glimmer uses full active tools and legacy Python-repr coercion; medium reasoning for Luna, server default for Glimmer; no cache flush; independent deterministic validators.",
               "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(72)
    for repeat in range(args.repeats):
        for task in TASKS:
            arms = list(args.arms)
            rng.shuffle(arms)
            for arm in arms:
                row = await run_one(arm, task, repeat, args)
                rows.append(row)
                payload["summary"] = summarize(rows)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps({k: row[k] for k in ("arm", "task", "repeat", "passed", "seconds")}), flush=True)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
