"""Measured per-run optimization counters; estimates are never billing credits."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


def empty_efficiency() -> dict[str, Any]:
    return {
        "round_trips_avoided": 0,
        "tokens_avoided_by_handles": 0,
        "reducer_bytes_saved": 0,
        "reducer_fallbacks": {},
        "compactions": {},
        # No provider cache-debt measurement is currently available.
        "cache_debt_tokens": 0,
    }


def get_efficiency(run_context: Any) -> dict[str, Any]:
    """Return the run-owned accumulator (or a disposable one without context)."""
    if run_context is None:
        return empty_efficiency()
    counters = getattr(run_context, "efficiency", None)
    if counters is None:
        counters = empty_efficiency()
        run_context.efficiency = counters
    return counters


def efficiency_snapshot(run_context: Any) -> dict[str, Any]:
    return deepcopy(get_efficiency(run_context))


def record_compaction(run_context: Any, reason: str) -> None:
    reasons = get_efficiency(run_context)["compactions"]
    reasons[reason] = reasons.get(reason, 0) + 1


_EVIDENCE_RECEIPT_LINE = re.compile(
    r"^[ \t]*(?:\[Tool Result\][ \t]*)?clawagents_evidence_receipt_v1[ \t]*\r?$",
    re.MULTILINE,
)


def contains_evidence_receipt(content: str) -> bool:
    """Recognize a receipt line, including after a fused edit confirmation."""
    return _EVIDENCE_RECEIPT_LINE.search(content) is not None
