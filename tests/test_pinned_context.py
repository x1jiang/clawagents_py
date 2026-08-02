"""Pinned context is an always-on rules source.

The point of pinning is that short, situational facts — which virtualenv, which
skill to remember — reach *every* round rather than being stated once and lost
to compaction. So it has to ride the rules pipeline, not the first user turn.
"""

from __future__ import annotations

from clawagents.memory.rules import (
    discover_rule_paths,
    load_rules_text,
    pinned_context_path,
    read_pinned_context,
    write_pinned_context,
)


def test_unset_pinned_context_reads_as_empty(tmp_path):
    assert read_pinned_context(tmp_path) == ""
    assert not pinned_context_path(tmp_path).exists()


def test_round_trip(tmp_path):
    write_pinned_context("Use .venv at repo root; uv sync before tests.", tmp_path)
    assert read_pinned_context(tmp_path) == "Use .venv at repo root; uv sync before tests."


def test_read_hides_the_generated_heading(tmp_path):
    """An editor shows this text verbatim; the heading is ours, not the user's."""
    write_pinned_context("just my note", tmp_path)
    assert read_pinned_context(tmp_path) == "just my note"
    # The heading is still on disk, so the model gets the labelled block.
    assert "# Pinned context" in pinned_context_path(tmp_path).read_text(encoding="utf-8")


def test_repeated_round_trips_do_not_nest_headings(tmp_path):
    """Save → load → save is the normal editing loop and must be stable."""
    write_pinned_context("note", tmp_path)
    for _ in range(3):
        write_pinned_context(read_pinned_context(tmp_path), tmp_path)
    stored = pinned_context_path(tmp_path).read_text(encoding="utf-8")
    assert stored.count("# Pinned context") == 1
    assert read_pinned_context(tmp_path) == "note"


def test_a_user_heading_of_their_own_is_preserved(tmp_path):
    write_pinned_context("# My rules\n\nuse uv", tmp_path)
    assert read_pinned_context(tmp_path) == "# My rules\n\nuse uv"


def test_it_is_discovered_as_a_rule_file(tmp_path):
    write_pinned_context("always use uv", tmp_path)
    assert pinned_context_path(tmp_path).resolve() in discover_rule_paths(tmp_path)


def test_it_leads_the_injected_rules_block(tmp_path):
    """Pinned text is the shortest source and must not be the first truncated."""
    write_pinned_context("PINNED-MARKER", tmp_path)
    (tmp_path / "AGENTS.md").write_text("AGENTS-MARKER", encoding="utf-8")

    text = load_rules_text(tmp_path)
    assert text is not None
    assert text.index("PINNED-MARKER") < text.index("AGENTS-MARKER")


def test_it_reaches_the_prompt_on_every_round(tmp_path):
    write_pinned_context("PINNED-MARKER", tmp_path)
    text = load_rules_text(tmp_path)
    assert text is not None
    assert "Project Rules (always-on)" in text
    assert "PINNED-MARKER" in text


def test_clearing_removes_the_file(tmp_path):
    write_pinned_context("temporary", tmp_path)
    assert write_pinned_context("   ", tmp_path) == ""
    assert not pinned_context_path(tmp_path).exists()
    assert discover_rule_paths(tmp_path) == []


def test_oversized_input_is_bounded(tmp_path):
    """This text is re-sent every round, so a pasted log must not ride along."""
    stored = write_pinned_context("x" * 10_000, tmp_path, max_chars=500)
    assert len(stored) == 500
    assert len(read_pinned_context(tmp_path)) < 700  # + header only


def test_writing_creates_the_clawagents_directory(tmp_path):
    write_pinned_context("first ever setting", tmp_path)
    assert pinned_context_path(tmp_path).is_file()


def test_no_temp_file_is_left_behind(tmp_path):
    write_pinned_context("value", tmp_path)
    leftovers = list((tmp_path / ".clawagents").glob("*.tmp"))
    assert leftovers == []
