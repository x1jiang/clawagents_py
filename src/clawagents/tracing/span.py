"""Span — the unit of tracing data."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SpanKind(str, Enum):
    """Categories of work clawagents traces."""

    AGENT = "agent"          # full agent.invoke() call
    TURN = "turn"            # one round of the ReAct loop (LLM + tool batch)
    GENERATION = "generation"  # one LLM request/response
    TOOL = "tool"            # one tool execution
    HANDOFF = "handoff"      # agent-to-agent transfer
    GUARDRAIL = "guardrail"  # input/output guardrail check
    SUBAGENT = "subagent"    # child agent run via task tool / as_tool
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


def _new_id(prefix: str) -> str:
    """Stable but human-debuggable id (8 hex chars). Not cryptographic."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Span:
    """A single span in a trace tree.

    The ``trace_id`` is shared by every span in one agent run; ``parent_id``
    forms the tree. ``span_id`` is unique. Times are POSIX seconds (float).
    """

    name: str
    kind: SpanKind
    trace_id: str
    span_id: str = field(default_factory=lambda: _new_id("span"))
    parent_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def end(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None) -> None:
        if self.ended_at is None:
            self.ended_at = time.time()
        self.status = status
        if error is not None:
            self.error_message = error

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.attributes.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (used by JSONL/OTLP exporters)."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "status": self.status.value,
            "attributes": dict(self.attributes),
            "error_message": self.error_message,
        }


def new_trace_id() -> str:
    return _new_id("trace")
