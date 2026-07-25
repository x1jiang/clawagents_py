"""Attribute prompt-cache misses to a cause, and price the waste.

A long agent run re-sends a slowly-growing prompt on every round, so provider
prompt caching is usually the single largest lever on cost. When it silently
stops working the bill roughly doubles with nothing in the logs to say why.

This walks the per-request usage of one run and reports, per call, how many
prompt tokens *should* have been cache reads but were billed at full input
rate — plus the most likely reason. Ported from the Pi harness's cache-stats
idea; the causes are what make it actionable rather than merely alarming.

Deliberate design points:

* **Noise floor.** Providers cache at block granularity, so a small shortfall
  is normal. Anything under :data:`NOISE_FLOOR_TOKENS` is ignored.
* **Context-change exemption.** Compaction legitimately rewrites the prefix; a
  miss on that round is expected, not waste. Callers pass those round indices.
* **Never-reported vs. missed.** Some providers do not report cache reads at
  all. Claiming "100% waste" there would be wrong, so if *no* call in the run
  reported a cache read the report is marked ``cache_reporting=False`` and no
  waste is attributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

# Anthropic's implicit cache TTL is 5 minutes; OpenAI's is longer but similar in
# spirit. Used only to label a miss, never to decide whether one happened.
IDLE_TTL_SECONDS = 300.0
# Cache blocks are coarse; below this a shortfall is granularity, not waste.
NOISE_FLOOR_TOKENS = 1024

Cause = Literal["model_switch", "idle_expiry", "context_change", "unknown"]


@dataclass(frozen=True)
class CacheMiss:
    """One call that re-paid for tokens the cache should have covered."""

    index: int
    model: str
    rebilled_tokens: int
    cause: Cause
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "model": self.model,
            "rebilled_tokens": self.rebilled_tokens,
            "cause": self.cause,
            "detail": self.detail,
        }


@dataclass
class CacheWasteReport:
    cache_reporting: bool = True
    total_rebilled_tokens: int = 0
    misses: list[CacheMiss] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.cache_reporting and self.total_rebilled_tokens >= NOISE_FLOOR_TOKENS

    def by_cause(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.misses:
            out[m.cause] = out.get(m.cause, 0) + m.rebilled_tokens
        return out

    def summary(self) -> str:
        """One human-readable line, or '' when there is nothing to say."""
        if not self.significant:
            return ""
        parts = [
            f"{tokens:,} via {cause.replace('_', ' ')}"
            for cause, tokens in sorted(
                self.by_cause().items(), key=lambda kv: -kv[1]
            )
        ]
        return (
            f"prompt cache: {self.total_rebilled_tokens:,} tokens re-billed "
            f"({'; '.join(parts)})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_reporting": self.cache_reporting,
            "total_rebilled_tokens": self.total_rebilled_tokens,
            "by_cause": self.by_cause(),
            "misses": [m.to_dict() for m in self.misses],
        }


def _cacheable_prefix(prev: Any, cur: Any) -> int:
    """Tokens the current call could plausibly have read from cache.

    The prompt grows monotonically within a run, so the overlap with the
    previous prompt is the upper bound on what caching could have covered.
    """
    return max(0, min(int(prev.prompt_tokens or 0), int(cur.prompt_tokens or 0)))


def analyze_cache_waste(
    per_request: Sequence[Any],
    *,
    context_change_rounds: Iterable[int] = (),
    idle_ttl_seconds: float = IDLE_TTL_SECONDS,
    noise_floor: int = NOISE_FLOOR_TOKENS,
) -> CacheWasteReport:
    """Attribute prompt-cache misses across one run's requests.

    ``per_request`` is a sequence of :class:`~clawagents.usage.RequestUsage`
    (anything with ``model`` / ``prompt_tokens`` / ``cached_input_tokens`` /
    ``started_at`` works). ``context_change_rounds`` are indices whose prefix
    was legitimately rewritten — compaction, a rewind, a system-prompt change —
    and are reported as ``context_change`` rather than counted as waste.
    """
    report = CacheWasteReport()
    requests = list(per_request or [])
    if len(requests) < 2:
        return report

    if not any(int(getattr(r, "cached_input_tokens", 0) or 0) > 0 for r in requests):
        # Provider never reports cache reads — absence of evidence only.
        report.cache_reporting = False
        return report

    exempt = set(context_change_rounds)
    for i in range(1, len(requests)):
        prev, cur = requests[i - 1], requests[i]
        expected = _cacheable_prefix(prev, cur)
        actual = int(getattr(cur, "cached_input_tokens", 0) or 0)
        rebilled = expected - actual
        if rebilled < noise_floor:
            continue

        prev_model = str(getattr(prev, "model", "") or "")
        cur_model = str(getattr(cur, "model", "") or "")
        prev_at = float(getattr(prev, "started_at", 0.0) or 0.0)
        cur_at = float(getattr(cur, "started_at", 0.0) or 0.0)
        gap = cur_at - prev_at if (prev_at and cur_at) else 0.0

        if i in exempt:
            cause: Cause = "context_change"
            detail = "prefix rewritten (compaction/rewind) — expected"
        elif cur_model and prev_model and cur_model != prev_model:
            cause = "model_switch"
            detail = f"switched {prev_model} -> {cur_model}; caches are per-model"
        elif gap > idle_ttl_seconds:
            cause = "idle_expiry"
            detail = f"idle {gap / 60:.0f}m exceeded the ~{idle_ttl_seconds / 60:.0f}m cache TTL"
        else:
            cause = "unknown"
            detail = "prefix changed or provider evicted the entry"

        report.misses.append(
            CacheMiss(
                index=i,
                model=cur_model,
                rebilled_tokens=rebilled,
                cause=cause,
                detail=detail,
            )
        )
        # A legitimately-changed prefix is not waste; report it, don't bill it.
        if cause != "context_change":
            report.total_rebilled_tokens += rebilled

    return report


__all__ = [
    "Cause",
    "CacheMiss",
    "CacheWasteReport",
    "IDLE_TTL_SECONDS",
    "NOISE_FLOOR_TOKENS",
    "analyze_cache_waste",
]
