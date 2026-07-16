"""Tests for circuit breaker half-open probe-lease reclamation."""

from __future__ import annotations

import pytest

from clawagents.circuit_breaker import (
    BreakerConfig,
    BreakerOpen,
    BreakerState,
    CircuitBreaker,
    Outcome,
    reset_provider_breakers,
)


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_abandoned_probe_slot_reclaimed_after_lease_expiry():
    clock = FakeClock()
    cb = CircuitBreaker(
        BreakerConfig(
            min_samples=1,
            error_rate_threshold=0.01,
            open_duration=0.05,
            half_open_max_probes=1,
            window_duration=60.0,
        ),
        clock=clock,
    )
    cb.record(Outcome.FAILURE)
    assert cb.state() == BreakerState.OPEN

    clock.advance(0.07)
    cb.check()  # admit probe
    assert cb.state() == BreakerState.HALF_OPEN
    with pytest.raises(BreakerOpen):
        cb.check()  # slot claimed

    # Abandon: no record(). After lease expiry, reclaim.
    clock.advance(0.05)
    cb.check()
    with pytest.raises(BreakerOpen):
        cb.check()

    cb.record(Outcome.SUCCESS)
    assert cb.state() == BreakerState.CLOSED


def test_repeatedly_abandoned_probes_keep_recovery_alive():
    clock = FakeClock()
    cb = CircuitBreaker(
        BreakerConfig(
            min_samples=1,
            error_rate_threshold=0.01,
            open_duration=0.05,
            half_open_max_probes=1,
        ),
        clock=clock,
    )
    cb.record(Outcome.FAILURE)
    clock.advance(0.07)

    for _ in range(3):
        cb.check()
        with pytest.raises(BreakerOpen):
            cb.check()
        clock.advance(0.05)

    cb.check()
    cb.record(Outcome.SUCCESS)
    assert cb.state() == BreakerState.CLOSED


def test_half_open_serialises_probes():
    clock = FakeClock()
    cb = CircuitBreaker(
        BreakerConfig(
            min_samples=1,
            error_rate_threshold=0.01,
            open_duration=0.05,
            half_open_max_probes=1,
        ),
        clock=clock,
    )
    for _ in range(3):
        cb.record(Outcome.FAILURE)
    assert cb.state() == BreakerState.OPEN
    clock.advance(0.06)
    cb.check()
    for _ in range(5):
        with pytest.raises(BreakerOpen):
            cb.check()


def test_provider_registry():
    reset_provider_breakers()
    from clawagents.circuit_breaker import get_provider_breaker

    a = get_provider_breaker("openai")
    b = get_provider_breaker("openai")
    assert a is b
    reset_provider_breakers()
