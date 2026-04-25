"""Tracing — hierarchical span model for clawagents.

Inspired by openai-agents-python's `tracing` module. Each meaningful agent
event (turn, LLM generation, tool call, handoff, guardrail check) becomes a
typed Span with a parent linkage. Spans are routed to pluggable
``TracingProcessor``s, which can in turn route to one or more
``TracingExporter``s (JSONL on disk, OTLP, Langfuse, etc.).

Public API:

  - ``Span``, ``SpanKind``, ``SpanStatus`` — data types
  - ``TracingProcessor``, ``TracingExporter`` — extension ABCs
  - ``BatchTraceProcessor`` — default processor with background flush
  - ``ConsoleSpanExporter``, ``JsonlSpanExporter``, ``NoopSpanExporter`` — built-in exporters
  - ``set_default_processor`` / ``add_trace_processor`` / ``flush_traces`` / ``shutdown_tracing``
  - context-manager helpers: ``agent_span``, ``turn_span``, ``generation_span``,
    ``tool_span``, ``handoff_span``, ``guardrail_span``, ``custom_span``

The default processor on import is a ``BatchTraceProcessor`` with a
``NoopSpanExporter``. Tracing is on, the writes are no-ops. Replace the
exporter via ``set_default_processor`` once an exporter is configured.
"""

from clawagents.tracing.span import Span, SpanKind, SpanStatus
from clawagents.tracing.processor import (
    TracingProcessor,
    TracingExporter,
    BatchTraceProcessor,
    NoopSpanExporter,
    ConsoleSpanExporter,
    JsonlSpanExporter,
    set_default_processor,
    get_default_processor,
    add_trace_processor,
    flush_traces,
    shutdown_tracing,
)
from clawagents.tracing.context import (
    agent_span,
    turn_span,
    generation_span,
    tool_span,
    handoff_span,
    guardrail_span,
    custom_span,
    current_span,
    current_trace_id,
)

__all__ = [
    "Span", "SpanKind", "SpanStatus",
    "TracingProcessor", "TracingExporter",
    "BatchTraceProcessor", "NoopSpanExporter", "ConsoleSpanExporter", "JsonlSpanExporter",
    "set_default_processor", "get_default_processor", "add_trace_processor",
    "flush_traces", "shutdown_tracing",
    "agent_span", "turn_span", "generation_span", "tool_span",
    "handoff_span", "guardrail_span", "custom_span",
    "current_span", "current_trace_id",
]
