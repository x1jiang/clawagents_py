"""Versioned event envelope for external UIs / Agent Server consumers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EVENT_SCHEMA_VERSION = "1"

EventKind = Literal[
    "action",
    "observation",
    "compaction",
    "usage",
    "status",
    "error",
    "checkpoint",
]


@dataclass
class EventEnvelope:
    kind: EventKind
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    schema_version: str = EVENT_SCHEMA_VERSION
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wrap_event(
    kind: EventKind,
    type: str,
    data: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    return EventEnvelope(
        kind=kind, type=type, data=dict(data or {}), run_id=run_id
    ).to_dict()


def map_legacy_event(legacy_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Map agent_loop emit kinds into the versioned envelope."""
    k = (legacy_kind or "").lower()
    if k in {"tool_call", "tool_start"}:
        kind: EventKind = "action"
    elif k in {"tool_result", "assistant", "text"}:
        kind = "observation"
    elif k in {"context", "compact_progress"}:
        kind = "compaction"
    elif k in {"usage"}:
        kind = "usage"
    elif k in {"error", "warn"}:
        kind = "error"
    elif k in {"checkpoint"}:
        kind = "checkpoint"
    else:
        kind = "status"
    return wrap_event(kind, legacy_kind, payload)
