"""TracingProcessor / TracingExporter — extension surface."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Optional

from clawagents.tracing.span import Span

logger = logging.getLogger("clawagents.tracing")


# ─── Exporter ABC ────────────────────────────────────────────────────────


class TracingExporter(ABC):
    """Receives a batch of finished spans and writes them somewhere.

    Exporters MUST tolerate being called concurrently from a worker thread.
    They SHOULD swallow their own errors so a misbehaving exporter cannot
    crash the agent loop.
    """

    @abstractmethod
    def export(self, spans: list[Span]) -> None:
        ...

    def shutdown(self) -> None:
        """Flush any buffered state. Default: noop."""


class NoopSpanExporter(TracingExporter):
    """Drops spans on the floor. Default to keep the surface zero-cost."""

    def export(self, spans: list[Span]) -> None:
        return


class ConsoleSpanExporter(TracingExporter):
    """Pretty-prints one line per finished span to stderr (for debugging)."""

    def __init__(self, stream: Optional[IO[str]] = None) -> None:
        self._stream = stream or sys.stderr

    def export(self, spans: list[Span]) -> None:
        for s in spans:
            try:
                self._stream.write(self._format(s) + "\n")
            except Exception:
                pass
        try:
            self._stream.flush()
        except Exception:
            pass

    @staticmethod
    def _format(s: Span) -> str:
        dur = f"{s.duration_s * 1000:.1f}ms" if s.duration_s is not None else "open"
        attrs = ""
        if s.attributes:
            kv = ",".join(f"{k}={v!r}" for k, v in list(s.attributes.items())[:4])
            attrs = f" [{kv}]"
        err = f" error={s.error_message!r}" if s.error_message else ""
        return f"[trace {s.trace_id[-8:]}] {s.kind.value}:{s.name} ({dur}){attrs}{err}"


class JsonlSpanExporter(TracingExporter):
    """Appends one JSON object per span to a file. Thread-safe; line-buffered."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: list[Span]) -> None:
        if not spans:
            return
        try:
            payload = "\n".join(json.dumps(s.to_dict(), default=str) for s in spans) + "\n"
        except Exception as e:
            logger.warning("JsonlSpanExporter: failed to serialise %d spans: %s", len(spans), e)
            return
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(payload)
            except Exception as e:
                logger.warning("JsonlSpanExporter: write failed: %s", e)


# ─── Processor ABC ───────────────────────────────────────────────────────


class TracingProcessor(ABC):
    """Receives spans as they finish; routes to one or more exporters."""

    @abstractmethod
    def on_span_end(self, span: Span) -> None:
        ...

    def force_flush(self, timeout_s: float = 5.0) -> None:
        """Flush pending spans synchronously."""

    def shutdown(self) -> None:
        """Stop background workers and release resources."""


class BatchTraceProcessor(TracingProcessor):
    """Buffers spans, flushes in a background thread.

    - Flushes when the buffer hits ``max_batch`` or every ``flush_interval_s``.
    - ``shutdown()`` drains and stops the worker.
    """

    def __init__(
        self,
        exporters: list[TracingExporter] | TracingExporter | None = None,
        *,
        max_batch: int = 64,
        flush_interval_s: float = 1.0,
    ) -> None:
        if exporters is None:
            exporters = [NoopSpanExporter()]
        elif isinstance(exporters, TracingExporter):
            exporters = [exporters]
        self._exporters: list[TracingExporter] = list(exporters)
        self._max_batch = max_batch
        self._flush_interval_s = flush_interval_s
        self._buffer: list[Span] = []
        self._cv = threading.Condition()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="clawagents-tracing", daemon=True)
        self._thread.start()

    def add_exporter(self, exporter: TracingExporter) -> None:
        with self._cv:
            self._exporters.append(exporter)

    def on_span_end(self, span: Span) -> None:
        with self._cv:
            if self._stopped:
                return
            self._buffer.append(span)
            if len(self._buffer) >= self._max_batch:
                self._cv.notify()

    def force_flush(self, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        with self._cv:
            self._cv.notify()
            while self._buffer and time.time() < deadline:
                self._cv.wait(timeout=max(0.01, deadline - time.time()))

    def shutdown(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        self._thread.join(timeout=5.0)
        for exporter in self._exporters:
            try:
                exporter.shutdown()
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stopped and not self._buffer:
                    return
                if not self._buffer and not self._stopped:
                    self._cv.wait(timeout=self._flush_interval_s)
                batch = self._buffer
                self._buffer = []
            if batch:
                for exporter in list(self._exporters):
                    try:
                        exporter.export(batch)
                    except Exception as e:
                        logger.warning("tracing exporter %s raised: %s", type(exporter).__name__, e)
                with self._cv:
                    self._cv.notify_all()


# ─── Module-level default processor ──────────────────────────────────────

_default_processor: TracingProcessor = BatchTraceProcessor(NoopSpanExporter())
_lock = threading.Lock()


def set_default_processor(processor: TracingProcessor) -> None:
    """Replace the default processor. Old one is shut down."""
    global _default_processor
    with _lock:
        old = _default_processor
        _default_processor = processor
    try:
        old.shutdown()
    except Exception:
        pass


def get_default_processor() -> TracingProcessor:
    return _default_processor


def add_trace_processor(processor: TracingProcessor) -> None:
    """Wrap the existing default in a fan-out so spans go to both."""
    global _default_processor
    with _lock:
        existing = _default_processor
        _default_processor = _FanOutProcessor([existing, processor])


class _FanOutProcessor(TracingProcessor):
    def __init__(self, processors: list[TracingProcessor]) -> None:
        self._processors = processors

    def on_span_end(self, span: Span) -> None:
        for p in self._processors:
            try:
                p.on_span_end(span)
            except Exception:
                pass

    def force_flush(self, timeout_s: float = 5.0) -> None:
        for p in self._processors:
            try:
                p.force_flush(timeout_s=timeout_s)
            except Exception:
                pass

    def shutdown(self) -> None:
        for p in self._processors:
            try:
                p.shutdown()
            except Exception:
                pass


def flush_traces(timeout_s: float = 5.0) -> None:
    _default_processor.force_flush(timeout_s=timeout_s)


def shutdown_tracing() -> None:
    """Flush + stop the default processor. Safe to call from atexit."""
    try:
        _default_processor.force_flush(timeout_s=5.0)
    except Exception:
        pass
    try:
        _default_processor.shutdown()
    except Exception:
        pass
