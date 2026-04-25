"""Span context managers + contextvar-based current-span tracking."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from clawagents.tracing.processor import get_default_processor
from clawagents.tracing.span import Span, SpanKind, SpanStatus, new_trace_id

# Per-task current-span chain. ``None`` = no active span (root agent run will create one).
_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "clawagents.tracing.current_span", default=None,
)
_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "clawagents.tracing.current_trace_id", default=None,
)


def current_span() -> Optional[Span]:
    return _current_span.get()


def current_trace_id() -> Optional[str]:
    return _current_trace_id.get()


@contextmanager
def _span(
    name: str,
    kind: SpanKind,
    *,
    attributes: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Iterator[Span]:
    """Open a span. Inherits trace_id+parent_id from the current contextvar.

    Use the kind-specific helpers below (``agent_span``, ``turn_span`` etc.) —
    this is the underlying primitive.
    """
    parent = _current_span.get()
    resolved_trace = trace_id or _current_trace_id.get() or (parent.trace_id if parent else new_trace_id())
    span = Span(
        name=name,
        kind=kind,
        trace_id=resolved_trace,
        parent_id=parent.span_id if parent else None,
        attributes=dict(attributes or {}),
    )
    span_token = _current_span.set(span)
    trace_token = _current_trace_id.set(resolved_trace)
    try:
        yield span
        if span.ended_at is None:
            span.end(status=SpanStatus.OK)
    except BaseException as e:
        if span.ended_at is None:
            span.end(status=SpanStatus.ERROR, error=str(e))
        raise
    finally:
        try:
            get_default_processor().on_span_end(span)
        except Exception:
            pass
        _current_span.reset(span_token)
        _current_trace_id.reset(trace_token)


def agent_span(name: str, **attrs: Any) -> Any:
    """Top-level span for an agent.invoke() call."""
    return _span(name, SpanKind.AGENT, attributes=attrs)


def turn_span(name: str = "turn", **attrs: Any) -> Any:
    """One iteration of the ReAct loop."""
    return _span(name, SpanKind.TURN, attributes=attrs)


def generation_span(name: str = "llm.chat", **attrs: Any) -> Any:
    """One LLM request/response."""
    return _span(name, SpanKind.GENERATION, attributes=attrs)


def tool_span(name: str, **attrs: Any) -> Any:
    """One tool execution."""
    return _span(f"tool.{name}", SpanKind.TOOL, attributes=attrs)


def handoff_span(name: str, **attrs: Any) -> Any:
    """Agent-to-agent transfer."""
    return _span(f"handoff.{name}", SpanKind.HANDOFF, attributes=attrs)


def guardrail_span(name: str, **attrs: Any) -> Any:
    """Input/output guardrail check."""
    return _span(f"guardrail.{name}", SpanKind.GUARDRAIL, attributes=attrs)


def custom_span(name: str, **attrs: Any) -> Any:
    """User-defined span."""
    return _span(name, SpanKind.CUSTOM, attributes=attrs)
