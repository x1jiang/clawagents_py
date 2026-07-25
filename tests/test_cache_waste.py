"""Prompt-cache waste attribution."""

from __future__ import annotations

from clawagents.cache_waste import (
    IDLE_TTL_SECONDS,
    NOISE_FLOOR_TOKENS,
    analyze_cache_waste,
)
from clawagents.usage import RequestUsage


def _req(prompt, cached, *, model="m1", at=1000.0):
    return RequestUsage(
        model=model,
        prompt_tokens=prompt,
        cached_input_tokens=cached,
        started_at=at,
    )


def test_healthy_cache_reports_no_waste():
    reqs = [
        _req(10_000, 0, at=1000),
        _req(12_000, 10_000, at=1010),
        _req(14_000, 12_000, at=1020),
    ]
    report = analyze_cache_waste(reqs)
    assert report.cache_reporting
    assert report.total_rebilled_tokens == 0
    assert report.summary() == ""


def test_model_switch_is_named_as_the_cause():
    reqs = [
        _req(100_000, 90_000, model="a", at=1000),
        _req(120_000, 0, model="b", at=1010),
    ]
    report = analyze_cache_waste(reqs)
    assert report.total_rebilled_tokens == 100_000
    (miss,) = report.misses
    assert miss.cause == "model_switch"
    assert "a -> b" in miss.detail
    assert "model switch" in report.summary()


def test_idle_gap_past_ttl_is_named_as_the_cause():
    reqs = [
        _req(80_000, 70_000, at=1000),
        _req(90_000, 0, at=1000 + IDLE_TTL_SECONDS + 120),
    ]
    report = analyze_cache_waste(reqs)
    (miss,) = report.misses
    assert miss.cause == "idle_expiry"
    assert "cache TTL" in miss.detail


def test_context_change_rounds_are_exempt_from_waste():
    reqs = [
        _req(200_000, 190_000, at=1000),
        _req(40_000, 0, at=1010),  # post-compaction: prefix legitimately gone
    ]
    report = analyze_cache_waste(reqs, context_change_rounds={1})
    (miss,) = report.misses
    assert miss.cause == "context_change"
    # Reported for transparency, but never counted against the user.
    assert report.total_rebilled_tokens == 0
    assert not report.significant


def test_shortfall_below_noise_floor_is_ignored():
    reqs = [
        _req(50_000, 49_000, at=1000),
        _req(50_500, 50_000 - (NOISE_FLOOR_TOKENS - 1), at=1010),
    ]
    report = analyze_cache_waste(reqs)
    assert report.misses == []


def test_provider_that_never_reports_cache_is_not_blamed():
    """Absence of cache reporting must not read as 100% waste."""
    reqs = [_req(100_000, 0, at=1000), _req(120_000, 0, at=1010)]
    report = analyze_cache_waste(reqs)
    assert report.cache_reporting is False
    assert report.total_rebilled_tokens == 0
    assert not report.significant


def test_single_request_run_is_a_noop():
    assert analyze_cache_waste([_req(1000, 0)]).total_rebilled_tokens == 0
    assert analyze_cache_waste([]).misses == []


def test_report_serializes_with_cause_breakdown():
    reqs = [
        _req(100_000, 90_000, model="a", at=1000),
        _req(120_000, 0, model="b", at=1010),
    ]
    data = analyze_cache_waste(reqs).to_dict()
    assert data["by_cause"] == {"model_switch": 100_000}
    assert data["misses"][0]["index"] == 1
