"""Provider circuit breaker with half-open probe-lease reclamation.

Grok Build parity (xai-circuit-breaker): Closed → Open → HalfOpen with a
sliding-window trip rule. In HalfOpen, only ``half_open_max_probes`` requests
are admitted. A probe whose owner never calls ``record()`` (cancelled future)
would otherwise strand the breaker forever — so a claim older than
``open_duration`` is treated as abandoned and exactly one caller may reclaim
the slot (CAS on ``probe_claimed_at``).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class BreakerOpen(Exception):
    """Raised when the breaker is shedding traffic."""

    def __init__(self, retry_after: float = 0.0):
        self.retry_after = float(retry_after)
        super().__init__(f"circuit breaker open (retry_after={self.retry_after:.3f}s)")


@dataclass
class BreakerConfig:
    window_duration: float = 60.0
    min_samples: int = 5
    error_rate_threshold: float = 0.5
    open_duration: float = 60.0
    half_open_max_probes: int = 1
    failure_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )
    enabled: bool = True

    @classmethod
    def client(cls) -> BreakerConfig:
        return cls(
            window_duration=60.0,
            min_samples=5,
            error_rate_threshold=0.5,
            open_duration=60.0,
            half_open_max_probes=1,
            failure_codes=frozenset({429, 500, 502, 503, 504}),
        )

    @classmethod
    def server(cls) -> BreakerConfig:
        return cls(
            window_duration=60.0,
            min_samples=10,
            error_rate_threshold=0.5,
            open_duration=10.0,
            half_open_max_probes=1,
        )


@dataclass
class _Sample:
    failure: bool
    at: float


class CircuitBreaker:
    """Thread-safe three-state breaker with abandoned-probe lease reclaim."""

    def __init__(
        self,
        config: BreakerConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ):
        self.config = config or BreakerConfig.client()
        self.config.half_open_max_probes = max(1, int(self.config.half_open_max_probes))
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._state = BreakerState.CLOSED
        self._baseline = self._clock()
        self._opened_at = 0.0
        self._half_open_probes = 0
        self._probe_claimed_at = 0.0
        self._window: list[_Sample] = []

    def _now(self) -> float:
        return self._clock()

    def _elapsed(self) -> float:
        return max(0.0, self._now() - self._baseline)

    def state(self) -> BreakerState:
        with self._lock:
            return self._state

    def is_open(self) -> bool:
        return self.state() == BreakerState.OPEN

    def error_rate(self) -> float:
        with self._lock:
            self._evict(self._now())
            if not self._window:
                return 0.0
            fails = sum(1 for s in self._window if s.failure)
            return fails / len(self._window)

    def is_failure_status(self, status: int) -> bool:
        return int(status) in self.config.failure_codes

    def check(self) -> None:
        """Admit a request or raise :class:`BreakerOpen`."""
        if not self.config.enabled:
            return
        with self._lock:
            if self._state == BreakerState.CLOSED:
                return
            if self._state == BreakerState.OPEN:
                self._check_open()
                return
            # HalfOpen
            self._try_half_open_probe()

    def record(self, outcome: Outcome) -> None:
        if not self.config.enabled:
            return
        is_failure = outcome is Outcome.FAILURE
        now = self._now()
        with self._lock:
            prev = self._state
            if prev == BreakerState.CLOSED:
                self._window.append(_Sample(failure=is_failure, at=now))
                self._evict(now)
                n = len(self._window)
                fails = sum(1 for s in self._window if s.failure)
                rate = (fails / n) if n else 0.0
                if n >= self.config.min_samples and rate >= self.config.error_rate_threshold:
                    self._trip(prev)
            elif prev == BreakerState.HALF_OPEN:
                if is_failure:
                    self._trip(prev)
                else:
                    self._close(prev)
            else:
                self._window.append(_Sample(failure=is_failure, at=now))
                self._evict(now)

    def force_half_open(self) -> None:
        """Test hook: jump straight to HalfOpen."""
        with self._lock:
            self._state = BreakerState.HALF_OPEN
            self._half_open_probes = 0

    def _evict(self, now: float) -> None:
        cutoff = now - self.config.window_duration
        self._window = [s for s in self._window if s.at >= cutoff]

    def _trip(self, prev: BreakerState) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._elapsed()
        self._half_open_probes = 0

    def _close(self, prev: BreakerState) -> None:
        self._state = BreakerState.CLOSED
        self._window.clear()
        self._half_open_probes = 0

    def _check_open(self) -> None:
        elapsed = self._elapsed() - self._opened_at
        if elapsed >= self.config.open_duration:
            # Open → HalfOpen, then claim a probe through the shared path.
            self._state = BreakerState.HALF_OPEN
            # Do NOT reset half_open_probes here (Grok CAS race note).
            self._try_half_open_probe()
            return
        raise BreakerOpen(
            retry_after=max(0.0, self.config.open_duration - elapsed)
        )

    def _try_half_open_probe(self) -> None:
        now = self._elapsed()
        if self._half_open_probes < self.config.half_open_max_probes:
            self._half_open_probes += 1
            self._probe_claimed_at = now
            return

        # All slots claimed — reclaim abandoned lease (older than open_duration).
        # Under the lock this is the CAS analogue: stamp a new claim time so
        # exactly one takeover succeeds; subsequent callers see a fresh lease.
        if now - self._probe_claimed_at >= self.config.open_duration:
            self._probe_claimed_at = now
            return

        raise BreakerOpen(retry_after=min(0.05, self.config.open_duration))


# Process-wide breakers keyed by endpoint identity (not bare provider class).
# Ollama-via-OpenAI and cloud OpenAI must not share a breaker.
_REGISTRY: dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()


def breaker_key(tag: str, *, base_url: str | None = None, model: str | None = None) -> str:
    """Stable breaker identity: tag + base_url + model."""
    t = (tag or "default").strip().lower() or "default"
    b = (base_url or "").strip().rstrip("/").lower()
    m = (model or "").strip().lower()
    return f"{t}|{b}|{m}"


def get_provider_breaker(tag: str, config: BreakerConfig | None = None) -> CircuitBreaker:
    """Return (or create) a breaker for ``tag``.

    Prefer passing a key from :func:`breaker_key` so local OpenAI-compatible
    endpoints do not trip the cloud OpenAI breaker.
    """
    key = (tag or "default").strip().lower() or "default"
    with _REGISTRY_LOCK:
        if key not in _REGISTRY:
            _REGISTRY[key] = CircuitBreaker(config or BreakerConfig.client())
        return _REGISTRY[key]


def reset_provider_breakers() -> None:
    """Test helper — clear the process registry."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


__all__ = [
    "BreakerState",
    "Outcome",
    "BreakerOpen",
    "BreakerConfig",
    "CircuitBreaker",
    "breaker_key",
    "get_provider_breaker",
    "reset_provider_breakers",
]
