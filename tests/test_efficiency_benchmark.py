"""Negative controls for paired experiment provenance, accounting and qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from clawagents.benchmarks.efficiency import (
    compare, cost_from_usage, digest, freeze, validate_manifest, verify_source,
)
from clawagents.benchmarks.runner import run


@pytest.fixture
def suite():
    return json.loads((Path(__file__).parents[1] / "benchmarks/efficiency/smoke.json").read_text())


@pytest.fixture
def manifest(suite):
    suite["evidence_scope"] = "private-heldout"
    return freeze(suite, model="same-pinned-model", provider="openai", repeats=2,
                  prices={"input_tokens": 2, "cached_input_tokens": 0.2,
                          "cache_creation_tokens": 2.5, "output_tokens": 8, "max_prompt_tokens": 272000, "source": "test prices"})


def rows_for(manifest, *, split="heldout"):
    rows = []
    for case in manifest["cases"]:
        if case["split"] != split:
            continue
        for repeat in range(manifest["repeats"]):
            for arm in ("baseline", "candidate"):
                n = 100 if arm == "baseline" else 50
                usage = dict(prompt_tokens=n, input_tokens=n, cached_input_tokens=0,
                             cache_creation_tokens=0, output_tokens=10, total_tokens=n+10, requests=2)
                rows.append(dict(case=case["id"], repeat=repeat, arm=arm, passed=True,
                                 provider=manifest["provider"], model=manifest["model"],
                                 manifest_sha256=manifest["sha256"], task_sha256=case["sha256"],
                                 runtime_sha256=manifest["arms"][arm]["runtime_sha256"],
                                 mode="live", measurement_complete=True, usage=usage,
                                 response_models=[manifest["model"]], token_measurement="provider_reported",
                                 estimated_cost_usd=cost_from_usage(usage, manifest["prices_per_million"]),
                                 request_usage=[{k: v // 2 for k, v in usage.items() if k != "requests"}] * 2,
                                 latency_seconds=1, then_run_requested=1, then_run_completed=1,
                                 provider_retries=None))
    return rows


def test_accepts_complete_pairs_and_uses_total_cost_including_failures(manifest):
    rows = rows_for(manifest)
    report = compare(manifest, rows)
    assert report["qualified"]
    assert report["summary"]["candidate"]["provider_retries"] is None
    rows[0]["passed"] = rows[1]["passed"] = False
    report = compare(manifest, rows)
    candidate = report["summary"]["candidate"]
    assert candidate["cost_per_successful_task_usd"] == candidate["estimated_cost_usd"] / 3


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop(),
    lambda r: r.append(deepcopy(r[0])),
    lambda r: r[0].update(model="other-model"),
    lambda r: r[0].update(provider="mantle"),
    lambda r: r[0].update(task_sha256="0"*64),
    lambda r: r[0].update(runtime_sha256="0"*64),
    lambda r: r[0].update(manifest_sha256="0"*64),
    lambda r: r[0].update(repeat=True),
    lambda r: r[0].update(repeat=10),
    lambda r: r[0].update(arm="other"),
    lambda r: r[0].update(case="unknown"),
    lambda r: r[0].update(mode="fixture"),
    lambda r: r[0].update(response_models=["silently-changed-model"]),
    lambda r: r[0].update(response_models=[]),
    lambda r: r[0].update(token_measurement="estimated_characters_divided_by_four"),
    lambda r: r[0].update(passed="yes"),
    lambda r: r[0].update(measurement_complete=False),
    lambda r: r[0].update(estimated_cost_usd=None),
    lambda r: r[0].update(estimated_cost_usd=100),
    lambda r: r[0].update(latency_seconds=float("nan")),
    lambda r: r[0].update(then_run_completed=2),
    lambda r: r[0].update(provider_retries=-1),
    lambda r: r[0]["usage"].update(total_tokens=-1),
    lambda r: r[0]["usage"].update(input_tokens=99),
    lambda r: r[0]["usage"].update(total_tokens=float("inf")),
    lambda r: r[0]["usage"].update(requests=0),
    lambda r: r[0]["usage"].update(requests=True),
])
def test_rejects_invalid_or_incomplete_pairs(manifest, mutation):
    rows = rows_for(manifest)
    mutation(rows)
    with pytest.raises(ValueError):
        compare(manifest, rows)


def test_each_regression_fails_even_with_offsetting_improvement(manifest):
    rows = rows_for(manifest)
    rows[1]["passed"] = False
    rows[2]["passed"] = False
    report = compare(manifest, rows)
    assert report["summary"]["baseline"]["score"] == report["summary"]["candidate"]["score"]
    assert not report["capability_pass"] and not report["qualified"]
    assert report["paired_regressions"]


def test_zero_successes_cannot_qualify(manifest):
    rows = rows_for(manifest)
    for row in rows:
        row["passed"] = False
    report = compare(manifest, rows)
    assert not report["qualified"] and report["cost_reduction"] is None


@pytest.mark.parametrize("slow,costly", [(True, False), (False, True)])
def test_efficiency_regression_fails(manifest, slow, costly):
    rows = rows_for(manifest)
    for row in rows:
        if row["arm"] == "candidate":
            if slow:
                row["latency_seconds"] = 2
            if costly:
                row["usage"].update(input_tokens=1000, prompt_tokens=1000, total_tokens=1010)
                row["request_usage"] = [{k: v // 2 for k, v in row["usage"].items() if k != "requests"}] * 2
                row["estimated_cost_usd"] = cost_from_usage(row["usage"], manifest["prices_per_million"])
    report = compare(manifest, rows)
    assert report["capability_pass"] and not report["qualified"]


def test_development_cannot_qualify_and_mixed_splits_rejected(manifest):
    assert not compare(manifest, rows_for(manifest, split="development"), split="development")["qualified"]
    with pytest.raises(ValueError):
        compare(manifest, rows_for(manifest, split="development"))


def test_freeze_detects_changed_serving_configuration_or_task(manifest, suite):
    manifest["serving"]["max_tokens"] += 1
    with pytest.raises(ValueError, match="changed after freeze"):
        validate_manifest(manifest)
    manifest["serving"]["max_tokens"] -= 1
    suite["tasks"][0]["prompt"] += " more"
    with pytest.raises(ValueError, match="suite changed"):
        validate_manifest(manifest, suite)


@pytest.mark.parametrize("kwargs", [dict(repeats=0), dict(repeats=101), dict(full_sends=True),
                                    dict(full_sends=3), dict(max_iterations=51),
                                    dict(min_cost_reduction=1), dict(max_latency_ratio=float("nan")),
                                    dict(base_url="https://user:secret@example.org/v1")])
def test_invalid_freeze_rejected(suite, kwargs):
    with pytest.raises(ValueError):
        freeze(suite, model="fixture", provider="fixture", **kwargs)


def test_no_missing_prices_or_mantle_endpoint(suite):
    with pytest.raises(ValueError):
        freeze(suite, model="model", provider="openai")
    with pytest.raises(ValueError, match="endpoint"):
        freeze(suite, model="model", provider="mantle")


@pytest.mark.parametrize("source", [
    "import os\ndef clamp(value, lower, upper):\n return 0",
    "@print\ndef clamp(value, lower, upper):\n return 0",
    "def clamp(value, lower, upper):\n return __import__('os').system('echo bad')",
    "def clamp(value, lower, upper):\n return value.__class__",
    "def clamp(value, lower, upper):\n return 10 ** value",
    "def clamp(value, lower, upper):\n while True: pass",
    "def clamp(value, lower, upper):\n return [x for x in range(100)]",
    "def clamp(value, lower, upper):\n return 0",
])
def test_verifier_rejects_unsafe_or_wrong_code(suite, source):
    assert not verify_source(source, suite["tasks"][0])


async def test_live_execution_needs_explicit_opt_in(manifest, suite, tmp_path):
    with pytest.raises(ValueError, match="allow-live"):
        await run(manifest, suite, split="heldout", output=tmp_path / "runs.jsonl")
    assert not list(tmp_path.iterdir())


async def test_real_agent_loop_fixture_does_not_claim_provider_savings(suite, tmp_path):
    manifest = freeze(suite, model="fixture", provider="fixture", repeats=1, full_sends=1)
    path = tmp_path / "runs.jsonl"
    rows = await run(manifest, suite, split="heldout", output=path)
    report = compare(manifest, rows)
    assert len(rows) == 4 and all(row["passed"] for row in rows)
    assert all(row["usage"]["requests"] >= 5 for row in rows)
    assert all(row["then_run_requested"] == 1 for row in rows)
    assert all(row["then_run_completed"] == 1 for row in rows)
    assert not report["qualified"] and report["cost_reduction"] is None
    assert all(row["estimated_cost_usd"] is None for row in rows)
    assert any(row["efficiency"]["tokens_avoided_by_handles"] > 0 for row in rows if row["arm"] == "candidate")
    with pytest.raises(FileExistsError):
        await run(manifest, suite, split="heldout", output=path)


async def test_wrong_runtime_rejected_before_calling_provider(suite, tmp_path):
    manifest = freeze(suite, model="fixture", provider="fixture", repeats=1, baseline_sha256="0"*64)
    with pytest.raises(ValueError, match="package differs"):
        await run(manifest, suite, split="heldout", output=tmp_path / "runs.jsonl")


def test_manifest_case_hash_cannot_be_resealed_to_other_cases(manifest, suite):
    manifest["cases"][0]["id"] = "invented"
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
    with pytest.raises(ValueError, match="identities"):
        validate_manifest(manifest, suite)


def test_public_smoke_cannot_qualify_even_with_live_usage(manifest):
    manifest["evidence_scope"] = "smoke"
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
    result = compare(manifest, rows_for(manifest))
    assert result["capability_pass"] and result["efficiency_pass"]
    assert not result["qualified"]


def test_crossing_frozen_pricing_tier_rejected(manifest):
    manifest["prices_per_million"]["max_prompt_tokens"] = 40
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
    with pytest.raises(ValueError, match="pricing tier"):
        compare(manifest, rows_for(manifest))


def test_inconsistent_per_request_usage_rejected(manifest):
    rows = rows_for(manifest)
    rows[0]["request_usage"] = []
    with pytest.raises(ValueError, match="per-request"):
        compare(manifest, rows)


async def test_provider_construction_failure_retains_sanitized_failed_pair(suite, manifest, tmp_path, monkeypatch):
    from clawagents.providers import llm
    def broken(*args, **kwargs):
        raise ValueError("credential-shaped-provider-error-must-not-be-recorded")
    monkeypatch.setattr(llm, "create_provider", broken)
    rows = await run(manifest, suite, split="heldout", output=tmp_path / "failed.jsonl", allow_live=True)
    assert len(rows) == 8
    assert all(row["error_type"] == "ValueError" and not row["measurement_complete"] for row in rows)
    assert "credential-shaped" not in json.dumps(rows)
    with pytest.raises(ValueError):
        compare(manifest, rows)


def test_same_task_cannot_be_relabeled_as_heldout(suite):
    duplicate = deepcopy(suite["tasks"][0])
    duplicate.update(id="relabeled-holdout", split="heldout")
    suite["tasks"].append(duplicate)
    with pytest.raises(ValueError, match="leaks"):
        freeze(suite, model="fixture", provider="fixture")
