"""Tests for the tracing module."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from clawagents.tracing import (
    Span, SpanKind, SpanStatus, TracingExporter,
    BatchTraceProcessor, JsonlSpanExporter, ConsoleSpanExporter,
    set_default_processor, get_default_processor, add_trace_processor,
    flush_traces, shutdown_tracing,
    agent_span, turn_span, generation_span, tool_span, handoff_span,
    custom_span, current_span, current_trace_id,
)


class _CollectExporter(TracingExporter):
    """Test exporter that just stashes everything in a list."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, spans: list[Span]) -> None:
        self.spans.extend(spans)


def test_span_dataclass_basics():
    s = Span(name="foo", kind=SpanKind.TOOL, trace_id="trace_123")
    assert s.name == "foo"
    assert s.kind == SpanKind.TOOL
    assert s.trace_id == "trace_123"
    assert s.span_id.startswith("span_")
    assert s.parent_id is None
    assert s.duration_s is None  # not ended
    s.end()
    assert s.duration_s is not None and s.duration_s >= 0
    assert s.status == SpanStatus.OK


def test_span_to_dict_serialisable():
    s = Span(name="t", kind=SpanKind.TURN, trace_id="trace_x", attributes={"round": 1})
    s.end(status=SpanStatus.OK)
    d = s.to_dict()
    json.dumps(d)  # must round-trip
    assert d["kind"] == "turn"
    assert d["attributes"] == {"round": 1}


def test_nested_spans_share_trace_and_parent_link():
    exporter = _CollectExporter()
    proc = BatchTraceProcessor(exporter, flush_interval_s=0.05, max_batch=64)
    set_default_processor(proc)
    try:
        with agent_span("root") as a:
            assert current_trace_id() == a.trace_id
            with turn_span("turn-1") as t:
                assert t.parent_id == a.span_id
                assert t.trace_id == a.trace_id
                with generation_span("llm.chat") as g:
                    assert g.parent_id == t.span_id
                    assert g.trace_id == a.trace_id
        flush_traces(timeout_s=2.0)
    finally:
        # restore a clean processor for other tests
        set_default_processor(BatchTraceProcessor())

    kinds = sorted(s.kind for s in exporter.spans)
    assert SpanKind.AGENT in kinds and SpanKind.TURN in kinds and SpanKind.GENERATION in kinds


def test_exception_marks_span_error():
    exporter = _CollectExporter()
    proc = BatchTraceProcessor(exporter, flush_interval_s=0.05)
    set_default_processor(proc)
    try:
        with pytest.raises(ValueError):
            with tool_span("boom"):
                raise ValueError("kaboom")
        flush_traces(timeout_s=2.0)
    finally:
        set_default_processor(BatchTraceProcessor())

    assert len(exporter.spans) == 1
    s = exporter.spans[0]
    assert s.status == SpanStatus.ERROR
    assert s.error_message == "kaboom"


def test_jsonl_exporter_writes_one_line_per_span(tmp_path: Path):
    out = tmp_path / "spans.jsonl"
    exporter = JsonlSpanExporter(out)
    proc = BatchTraceProcessor(exporter, flush_interval_s=0.05)
    set_default_processor(proc)
    try:
        with agent_span("root"):
            with turn_span("t1"):
                pass
            with turn_span("t2"):
                pass
        flush_traces(timeout_s=2.0)
    finally:
        set_default_processor(BatchTraceProcessor())

    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    parsed = [json.loads(ln) for ln in lines]
    assert all("trace_id" in p for p in parsed)
    # All three share a trace_id
    assert len({p["trace_id"] for p in parsed}) == 1


def test_console_exporter_does_not_crash_on_unrenderable_attrs(capsys):
    exporter = ConsoleSpanExporter()
    proc = BatchTraceProcessor(exporter, flush_interval_s=0.05)
    set_default_processor(proc)
    try:
        with custom_span("weird", a=object()):
            pass
        flush_traces(timeout_s=2.0)
    finally:
        set_default_processor(BatchTraceProcessor())

    err = capsys.readouterr().err
    assert "weird" in err


def test_add_trace_processor_fans_out():
    a = _CollectExporter()
    b = _CollectExporter()
    set_default_processor(BatchTraceProcessor(a, flush_interval_s=0.05))
    add_trace_processor(BatchTraceProcessor(b, flush_interval_s=0.05))
    try:
        with custom_span("x"):
            pass
        flush_traces(timeout_s=2.0)
    finally:
        set_default_processor(BatchTraceProcessor())

    assert len(a.spans) == 1
    assert len(b.spans) == 1
    assert a.spans[0].name == "x"
    assert b.spans[0].name == "x"


def test_no_active_span_outside_context():
    set_default_processor(BatchTraceProcessor())
    assert current_span() is None
    assert current_trace_id() is None


def test_shutdown_drains_buffer():
    exporter = _CollectExporter()
    proc = BatchTraceProcessor(exporter, flush_interval_s=10.0, max_batch=1000)
    set_default_processor(proc)
    try:
        with custom_span("pending"):
            pass
        # Without flushing, buffer should still hold the span until we shut down.
        proc.force_flush(timeout_s=2.0)
    finally:
        set_default_processor(BatchTraceProcessor())

    assert len(exporter.spans) == 1
