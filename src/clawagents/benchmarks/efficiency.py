"""Frozen paired experiments with separate correctness and efficiency gates.

Fixtures exercise the real agent/tool loop but cannot qualify a performance
claim. Live cost is computed from reported usage and frozen prices, not invoices.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS = ("baseline", "candidate")
TOKEN_FIELDS = ("prompt_tokens", "input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens", "total_tokens", "requests")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def runtime_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob("*.py"))})


def _number(value: Any, name: str, *, integer: bool = False, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0) or (integer and not isinstance(value, int)):
        raise ValueError(f"invalid {name}")


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_suite(suite: dict[str, Any]) -> None:
    tasks = suite.get("tasks", [])
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("suite requires unique tasks")
    if {t.get("split") for t in tasks} != {"development", "heldout"}:
        raise ValueError("suite requires disjoint development and heldout tasks")
    identities = [digest({k: v for k, v in task.items() if k not in ("id", "split")}) for task in tasks]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate task content leaks across repetitions or splits")
    for task in tasks:
        if not all(isinstance(task.get(k), str) and task[k] for k in ("id", "prompt", "source", "solution", "function")):
            raise ValueError("invalid task text")
        if not task.get("checks") or not task.get("parameters"):
            raise ValueError("task requires exact checks and parameters")
        # The supplied reference is an offline fixture, never passed to the model.
        if not verify_source(task["solution"], task):
            raise ValueError(f"invalid reference for {task['id']}")


def freeze(suite: dict[str, Any], *, model: str, provider: str, base_url: str = "",
           repeats: int = 2, full_sends: int = 1, prices: dict[str, Any] | None = None,
           baseline_sha256: str | None = None, candidate_sha256: str | None = None,
           max_iterations: int = 12, timeout_seconds: int = 120,
           min_cost_reduction: float = 0.05, max_latency_ratio: float = 1.25,
           reasoning_effort: str = "", max_tokens: int = 2048) -> dict[str, Any]:
    validate_suite(suite)
    current = runtime_digest()
    manifest = {
        "schema": 1, "suite_sha256": digest(suite),
        "evidence_scope": suite.get("evidence_scope", "smoke"),
        "cases": [{"id": t["id"], "split": t["split"], "sha256": digest(t)} for t in suite["tasks"]],
        "provider": provider, "model": model, "base_url": base_url,
        "repeats": repeats, "max_iterations": max_iterations, "timeout_seconds": timeout_seconds,
        "serving": {"max_tokens": max_tokens, "temperature": 0.0, "reasoning_effort": reasoning_effort},
        "arms": {"baseline": {"runtime_sha256": baseline_sha256 or current, "observation_full_sends": 0},
                 "candidate": {"runtime_sha256": candidate_sha256 or current, "observation_full_sends": full_sends}},
        "prices_per_million": prices or {},
        "gate": {"min_cost_reduction": min_cost_reduction, "max_latency_ratio": max_latency_ratio},
    }
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any], suite: dict[str, Any] | None = None) -> None:
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest.get("schema") != 1 or manifest.get("sha256") != digest(body):
        raise ValueError("manifest changed after freeze")
    if manifest.get("evidence_scope") not in ("smoke", "private-heldout"):
        raise ValueError("invalid evidence scope")
    if not manifest.get("model") or manifest.get("provider") not in ("fixture", "openai", "mantle"):
        raise ValueError("explicit supported provider and model required")
    from urllib.parse import urlsplit
    endpoint = urlsplit(manifest["base_url"])
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("base_url must not contain credentials, query or fragment")
    if manifest["provider"] == "mantle" and (not endpoint.hostname or "bedrock-mantle." not in endpoint.hostname):
        raise ValueError("Mantle requires an explicit regional endpoint")
    for key in ("repeats", "max_iterations", "timeout_seconds"):
        _number(manifest[key], key, integer=True, positive=True)
    if manifest["repeats"] > 100 or manifest["max_iterations"] > 50 or manifest["timeout_seconds"] > 600:
        raise ValueError("experiment exceeds bounded run limits")
    _number(manifest["serving"]["max_tokens"], "max_tokens", integer=True, positive=True)
    if manifest["serving"]["max_tokens"] > 16384 or manifest["serving"]["temperature"] != 0:
        raise ValueError("unsupported serving configuration")
    if set(manifest["arms"]) != set(ARMS):
        raise ValueError("exactly baseline and candidate required")
    for arm in manifest["arms"].values():
        if not _sha(arm["runtime_sha256"]) or type(arm["observation_full_sends"]) is not int or arm["observation_full_sends"] not in (0, 1, 2):
            raise ValueError("invalid frozen arm")
    cases = manifest["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases) or {c["split"] for c in cases} != {"development", "heldout"}:
        raise ValueError("invalid frozen case split")
    if any(not _sha(c["sha256"]) for c in cases):
        raise ValueError("invalid task digest")
    gate = manifest["gate"]
    _number(gate["min_cost_reduction"], "min_cost_reduction", positive=True)
    _number(gate["max_latency_ratio"], "max_latency_ratio", positive=True)
    if gate["min_cost_reduction"] >= 1:
        raise ValueError("cost reduction must be below 1")
    if manifest["provider"] != "fixture":
        prices = manifest["prices_per_million"]
        for key in ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens"):
            _number(prices.get(key), key + " price")
        _number(prices.get("max_prompt_tokens"), "price max_prompt_tokens", integer=True, positive=True)
        if not any(prices[k] > 0 for k in ("input_tokens", "output_tokens")) or not prices.get("source"):
            raise ValueError("live experiments require nonzero frozen prices and source")
    if suite is not None:
        validate_suite(suite)
        if digest(suite) != manifest["suite_sha256"]:
            raise ValueError("task suite changed after freeze")
        expected = [{"id": t["id"], "split": t["split"], "sha256": digest(t)} for t in suite["tasks"]]
        if expected != cases:
            raise ValueError("frozen case identities do not match suite")


def verify_source(source: str, task: dict[str, Any]) -> bool:
    """Check small numeric functions without permitting imports or I/O.

    Only one undecorated function returning an arithmetic expression is accepted.
    No loops, attributes, comprehensions, subscripts or user-defined calls execute.
    """
    import ast
    if not isinstance(source, str) or len(source) > 64_000:
        return False
    try:
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return False
        fn = tree.body[0]
        if fn.name != task["function"] or fn.decorator_list or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
            return False
        if fn.returns or fn.args.defaults or fn.args.kw_defaults or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg:
            return False
        if [a.arg for a in fn.args.args] != task["parameters"] or any(a.annotation for a in fn.args.args):
            return False
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Load,
                   ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.BoolOp, ast.Name,
                   ast.Constant, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                   ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
                   ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        nodes = list(ast.walk(tree))
        if len(nodes) > 200:
            return False
        names = set(task["parameters"]) | {"min", "max", "abs"}
        for node in nodes:
            if not isinstance(node, allowed):
                return False
            if isinstance(node, ast.Name) and node.id not in names:
                return False
            if isinstance(node, ast.Constant) and (type(node.value) not in (int, float, bool) or abs(node.value) > 1_000_000):
                return False
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in {"min", "max", "abs"} or node.keywords):
                return False
        namespace: dict[str, Any] = {"__builtins__": {}, "min": min, "max": max, "abs": abs}
        exec(compile(tree, "<benchmark-answer>", "exec"), namespace)
        for check in task["checks"]:
            if len(check["args"]) != len(task["parameters"]) or any(type(x) not in (int, float, bool) or not math.isfinite(x) or abs(x) > 1_000_000 for x in check["args"]):
                return False
            actual = namespace[fn.name](*check["args"])
            if actual != check["expected"] or not isinstance(actual, (int, float, bool)):
                return False
        return bool(task["checks"])
    except (SyntaxError, TypeError, ValueError, ArithmeticError, KeyError, RecursionError):
        return False


def cost_from_usage(usage: dict[str, Any], prices: dict[str, Any]) -> float:
    return sum(usage[key] * prices[key] for key in ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")) / 1_000_000


def compare(manifest: dict[str, Any], rows: list[dict[str, Any]], *, split: str = "heldout") -> dict[str, Any]:
    validate_manifest(manifest)
    if split not in ("development", "heldout"):
        raise ValueError("invalid split")
    cases = {c["id"]: c for c in manifest["cases"] if c["split"] == split}
    expected = {(c, r, a) for c in cases for r in range(manifest["repeats"]) for a in ARMS}
    found: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["case"], row["repeat"], row["arm"])
        if key not in expected or key in found or type(row["repeat"]) is not int:
            raise ValueError("duplicate, unknown or wrong-split row")
        if any(row.get(k) != manifest[k] for k in ("provider", "model")) or row.get("manifest_sha256") != manifest["sha256"]:
            raise ValueError("provider/model/manifest identity mismatch")
        if row.get("task_sha256") != cases[row["case"]]["sha256"] or row.get("runtime_sha256") != manifest["arms"][row["arm"]]["runtime_sha256"]:
            raise ValueError("task/runtime identity mismatch")
        mode = "fixture" if manifest["provider"] == "fixture" else "live"
        if row.get("mode") != mode or type(row.get("passed")) is not bool:
            raise ValueError("invalid execution mode or verifier outcome")
        usage = row["usage"]
        for field in TOKEN_FIELDS:
            _number(usage[field], field, integer=True)
        if usage["prompt_tokens"] != usage["input_tokens"] + usage["cached_input_tokens"] + usage["cache_creation_tokens"] or usage["total_tokens"] != usage["prompt_tokens"] + usage["output_tokens"]:
            raise ValueError("inconsistent token accounting")
        _number(row["latency_seconds"], "latency_seconds", positive=True)
        for field in ("then_run_requested", "then_run_completed"):
            _number(row[field], field, integer=True)
        if row["then_run_completed"] > row["then_run_requested"]:
            raise ValueError("impossible fusion uptake")
        if row.get("provider_retries") is not None:
            _number(row["provider_retries"], "provider_retries", integer=True)
        models = row.get("response_models")
        if not isinstance(models, list) or not models or any(not isinstance(m, str) or not m for m in models):
            raise ValueError("actual response model identity missing")
        if row.get("token_measurement") != ("provider_reported" if mode == "live" else "estimated_characters_divided_by_four"):
            raise ValueError("token measurement provenance mismatch")
        if mode == "live":
            _number(row.get("estimated_cost_usd"), "estimated_cost_usd")
            if row.get("measurement_complete") is not True or not usage["requests"] or not usage["total_tokens"]:
                raise ValueError("live provider accounting incomplete; no efficiency verdict")
            requests = row.get("request_usage", [])
            if len(requests) != usage["requests"]:
                raise ValueError("per-request accounting incomplete")
            for request in requests:
                for field in TOKEN_FIELDS[:-1]:
                    _number(request[field], "request " + field, integer=True)
                if request["prompt_tokens"] > manifest["prices_per_million"]["max_prompt_tokens"]:
                    raise ValueError("request exceeds frozen pricing tier; no cost verdict")
                if request["prompt_tokens"] != request["input_tokens"] + request["cached_input_tokens"] + request["cache_creation_tokens"] or request["total_tokens"] != request["prompt_tokens"] + request["output_tokens"]:
                    raise ValueError("inconsistent per-request accounting")
            if any(sum(request[field] for request in requests) != usage[field] for field in TOKEN_FIELDS[:-1]):
                raise ValueError("per-request totals differ from aggregate usage")
            if not math.isclose(row["estimated_cost_usd"], cost_from_usage(usage, manifest["prices_per_million"]), abs_tol=1e-12, rel_tol=1e-9):
                raise ValueError("cost does not match frozen usage/prices")
        elif row.get("estimated_cost_usd") is not None:
            raise ValueError("fixture token estimates are not provider cost")
        found[key] = row
    if set(found) != expected:
        raise ValueError("incomplete paired experiment (every case/repeat/arm required)")
    for case in cases:
        for repeat in range(manifest["repeats"]):
            if set(found[case, repeat, "baseline"]["response_models"]) != set(found[case, repeat, "candidate"]["response_models"]):
                raise ValueError("actual response models differ across paired arms")
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        successes = sum(r["passed"] for r in selected)
        cost = sum(r["estimated_cost_usd"] for r in selected) if manifest["provider"] != "fixture" else None
        summary[arm] = {
            "runs": len(selected), "successes": successes, "score": successes / len(selected),
            "total_tokens": sum(r["usage"]["total_tokens"] for r in selected),
            "latency_seconds": sum(r["latency_seconds"] for r in selected),
            "estimated_cost_usd": cost, "cost_per_successful_task_usd": cost / successes if cost is not None and successes else None,
            "then_run_requested": sum(r["then_run_requested"] for r in selected),
            "then_run_completed": sum(r["then_run_completed"] for r in selected),
            "provider_retries": sum(r["provider_retries"] for r in selected) if all(r.get("provider_retries") is not None for r in selected) else None,
        }
    regressions = [f"{case}:{repeat}" for case in cases for repeat in range(manifest["repeats"])
                   if found[case, repeat, "baseline"]["passed"] and not found[case, repeat, "candidate"]["passed"]]
    baseline, candidate = summary["baseline"], summary["candidate"]
    capability = not regressions and candidate["successes"] >= baseline["successes"] and baseline["successes"] > 0
    cost_before, cost_after = baseline["cost_per_successful_task_usd"], candidate["cost_per_successful_task_usd"]
    saving = 1 - cost_after / cost_before if cost_before and cost_after is not None else None
    latency_ratio = candidate["latency_seconds"] / baseline["latency_seconds"]
    efficiency = saving is not None and saving >= manifest["gate"]["min_cost_reduction"] and latency_ratio <= manifest["gate"]["max_latency_ratio"]
    qualified = split == "heldout" and manifest["provider"] != "fixture" and manifest["evidence_scope"] == "private-heldout" and capability and efficiency
    return {"manifest_sha256": manifest["sha256"], "split": split, "mode": "fixture" if manifest["provider"] == "fixture" else "live",
            "summary": summary, "capability_pass": capability, "paired_regressions": regressions,
            "cost_reduction": saving, "latency_ratio": latency_ratio, "efficiency_pass": efficiency,
            "qualified": qualified,
            "limitations": ["Fixture runs are plumbing checks, never model-quality or provider-savings evidence." if manifest["provider"] == "fixture" else "Cost uses provider-reported usage and frozen prices; unreported failed-request charges are unknown.",
                            "Provider-internal retries are unavailable unless explicitly instrumented.",
                            "Public example cases are smoke tasks, not a private held-out capability benchmark. Private-heldout scope is an operator attestation, not proof of task secrecy.",
                            "A point-estimate gate is not statistical significance; use multiple private cases and repeats."]}
