"""Interjection format + standalone drain + stranded promotion."""

from __future__ import annotations

from clawagents.interjection import (
    enqueue_interject,
    drain_interjects,
    format_interjection,
    take_stranded_interjects,
)
from clawagents.run_context import RunContext


def test_format_interjection_envelope():
    out = format_interjection("stop and fix the test first")
    assert out.startswith("The user sent a message while you were working:\n<user_query>\n")
    assert out.endswith("\n</user_query>")
    assert "stop and fix the test first" in out


def test_drain_never_merges_entries():
    ctx = RunContext()
    assert enqueue_interject(ctx, "one")
    assert enqueue_interject(ctx, "two")
    parts = drain_interjects(ctx)
    assert len(parts) == 2
    assert "<user_query>\none\n</user_query>" in parts[0]
    assert "<user_query>\ntwo\n</user_query>" in parts[1]
    assert drain_interjects(ctx) == []


def test_legacy_string_key_migrates():
    ctx = RunContext()
    ctx._metadata["pending_interject"] = "legacy"
    assert enqueue_interject(ctx, "new")
    parts = drain_interjects(ctx)
    assert len(parts) == 2
    assert "legacy" in parts[0]
    assert "new" in parts[1]


def test_stranded_raw_texts():
    ctx = RunContext()
    enqueue_interject(ctx, "keep me")
    stranded = take_stranded_interjects(ctx)
    assert stranded == ["keep me"]
    assert take_stranded_interjects(ctx) == []
