"""Incrementally retain head+tail of a text stream without buffering the middle."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class BoundedTextAccumulator:
    """Keeps the beginning and end of a stream within ``max_chars``."""

    def __init__(self, max_chars: int) -> None:
        if not isinstance(max_chars, int) or max_chars < 2:
            raise ValueError("max_chars must be an integer >= 2")
        self.max_chars = max_chars
        self.total_chars = 0
        self._head_limit = (max_chars + 1) // 2
        self._tail_limit = max_chars - self._head_limit
        self._head = ""
        self._tail = ""

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self.total_chars += len(chunk)
        if len(self._head) < self._head_limit:
            needed = self._head_limit - len(self._head)
            self._head += chunk[:needed]
            chunk = chunk[needed:]
        if chunk and self._tail_limit > 0:
            self._tail = (self._tail + chunk)[-self._tail_limit :]

    @property
    def truncated_chars(self) -> int:
        return max(0, self.total_chars - self.max_chars)

    def __str__(self) -> str:
        if self.truncated_chars == 0:
            return self._head + self._tail
        return (
            f"{self._head}\n\n"
            f"... [truncated {self.truncated_chars} chars] ...\n\n"
            f"{self._tail}"
        )


class SpoolingTextAccumulator(BoundedTextAccumulator):
    """Bounded head/tail memory plus a full-output spill file when truncated.

    Bytes are written to a private temporary file as they arrive. ``close``
    removes that file for small outputs and returns its path only when the
    in-memory view was truncated.
    """

    def __init__(self, max_chars: int) -> None:
        super().__init__(max_chars)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="replace",
            prefix="clawagents-output-",
            suffix=".txt",
            delete=False,
        )
        self._handle = handle
        self._path = Path(handle.name)
        self._closed = False

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self._handle.write(chunk)
        super().append(chunk)

    def close(self) -> str | None:
        if self._closed:
            return str(self._path) if self.truncated_chars > 0 and self._path.exists() else None
        self._closed = True
        self._handle.close()
        if self.truncated_chars > 0:
            return str(self._path)
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        return None
