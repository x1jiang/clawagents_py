"""Typed dataclasses for agent stream events.

Historically ``ClawAgent`` emits events via ``on_event(kind: str, data: dict)``,
where ``kind`` is one of the string literals in :data:`EventKind`. That keeps
the call site cheap but gives callers no type safety.

This module defines a parallel set of dataclasses that can be constructed
from ``(kind, data)`` via :func:`stream_event_from_kind`. They are emitted
alongside the legacy string-kind events via :meth:`ClawAgent.on_stream_event`
when the caller provides one. The old ``on_event`` path is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class StreamEvent:
    """Base class for typed stream events. ``kind`` mirrors the legacy string."""
    kind: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnStartedEvent(StreamEvent):
    iteration: int = 0
    kind: str = "turn_started"


@dataclass
class AssistantTextEvent(StreamEvent):
    content: str = ""
    thinking: str | None = None
    kind: str = "assistant_message"


@dataclass
class AssistantDeltaEvent(StreamEvent):
    """Incremental streamed chunk from the model (if streaming is enabled)."""
    delta: str = ""
    kind: str = "assistant_delta"


@dataclass
class ToolCallPlannedEvent(StreamEvent):
    tool_name: str = ""
    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    kind: str = "tool_call"


@dataclass
class ToolStartedEvent(StreamEvent):
    tool_name: str = ""
    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    kind: str = "tool_started"


@dataclass
class ToolResultEvent(StreamEvent):
    tool_name: str = ""
    call_id: str = ""
    success: bool = True
    output: str = ""
    error: str | None = None
    kind: str = "tool_result"


@dataclass
class ApprovalRequiredEvent(StreamEvent):
    tool_name: str = ""
    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    kind: str = "approval_required"


@dataclass
class UsageEvent(StreamEvent):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    kind: str = "usage"


@dataclass
class GuardrailTrippedEvent(StreamEvent):
    guardrail_name: str = ""
    where: str = ""
    behavior: str = ""
    message: str = ""
    kind: str = "guardrail_tripped"


@dataclass
class CompactProgressEvent(StreamEvent):
    """Emitted as context compaction starts, retries, completes, or falls back."""
    phase: str = ""
    message: str = ""
    current_tokens: int = 0
    budget: int = 0
    message_count: int = 0
    older_messages: int = 0
    recent_messages: int = 0
    kind: str = "compact_progress"


@dataclass
class HandoffOccurredEvent(StreamEvent):
    """Emitted when the loop transfers control to another agent via :class:`Handoff`."""
    from_agent: str = ""
    to_agent: str = ""
    tool_name: str = ""
    reason: str = ""
    kind: str = "handoff_occurred"


@dataclass
class FinalOutputEvent(StreamEvent):
    output: Any = None
    raw: str = ""
    kind: str = "final_output"


@dataclass
class ErrorStreamEvent(StreamEvent):
    error: str = ""
    recoverable: bool = False
    kind: str = "error"


# Backward-compat alias for 6.1.x callers. New code should prefer
# ``ErrorStreamEvent`` to match the TypeScript port and avoid confusion
# with the global DOM ``ErrorEvent``.
ErrorEvent = ErrorStreamEvent


AnyStreamEvent = Union[
    TurnStartedEvent,
    AssistantTextEvent,
    AssistantDeltaEvent,
    ToolCallPlannedEvent,
    ToolStartedEvent,
    ToolResultEvent,
    ApprovalRequiredEvent,
    UsageEvent,
    GuardrailTrippedEvent,
    CompactProgressEvent,
    HandoffOccurredEvent,
    FinalOutputEvent,
    ErrorStreamEvent,
    StreamEvent,
]


_KIND_TO_CLS: dict[str, type[StreamEvent]] = {
    "turn_started": TurnStartedEvent,
    "assistant_message": AssistantTextEvent,
    "assistant_delta": AssistantDeltaEvent,
    "tool_call": ToolCallPlannedEvent,
    "tool_started": ToolStartedEvent,
    "tool_result": ToolResultEvent,
    "approval_required": ApprovalRequiredEvent,
    "usage": UsageEvent,
    "guardrail_tripped": GuardrailTrippedEvent,
    "compact_progress": CompactProgressEvent,
    "handoff_occurred": HandoffOccurredEvent,
    "final_output": FinalOutputEvent,
    "error": ErrorStreamEvent,
}


def stream_event_from_kind(kind: str, data: dict[str, Any] | None = None) -> StreamEvent:
    """Promote a legacy ``(kind, data)`` event into the matching typed class.

    Unknown kinds fall back to the base :class:`StreamEvent` so callers that
    opt-in via ``on_stream_event`` never crash on new kinds.
    """
    data = data or {}
    cls = _KIND_TO_CLS.get(kind, StreamEvent)
    init_kwargs: dict[str, Any] = {"kind": kind, "data": data}
    known_fields = {f for f in cls.__dataclass_fields__}
    for k, v in data.items():
        if k in known_fields and k not in {"kind", "data"}:
            init_kwargs[k] = v
    try:
        return cls(**init_kwargs)
    except TypeError:
        return StreamEvent(kind=kind, data=data)
