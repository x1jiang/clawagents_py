"""Event envelope package."""

from clawagents.events.envelope import (
    EVENT_SCHEMA_VERSION,
    EventEnvelope,
    EventKind,
    map_legacy_event,
    wrap_event,
)

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EventEnvelope",
    "EventKind",
    "map_legacy_event",
    "wrap_event",
]
