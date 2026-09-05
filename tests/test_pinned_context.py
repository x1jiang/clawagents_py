"""Pinned context is an always-on rules source.

The point of pinning is that short, situational facts — which virtualenv, which
skill to remember — reach *every* round rather than being stated once and lost
to compaction. So it has to ride the rules pipeline, not the first user turn.
"""

from __future__ import annotations

import os

from clawagents.memory.rules import (
    discover_rule_paths,
    load_rules_text,
    pinned_context_path,
    read_pinned_context,
    write_pinned_context,
)
from clawagents.prompts import (
    INJECTION_BEGIN,
    PINNED_BEGIN,
    PINNED_END,
    PROMPT_CACHE_BOUNDARY,
    append_pinned_context,
    append_prompt_injection,
)
from clawagents.providers.llm import LLMMessage, LLMProvider


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


# ─── Tail placement ──────────────────────────────────────────────────────
#
# Being *in* the rules blob is not enough: that blob sits mid-prompt, after the
# full tool catalog. Pinned context gets its own block at the very END of the
# system message, re-applied every LLM round, framed as taking precedence.

class _StubLLM(LLMProvider):
    name = "stub"
    model = "stub-model"

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None, **kwargs):
        raise NotImplementedError()


def _sys(content: str) -> list:
    return [LLMMessage(role="system", content=content), LLMMessage(role="user", content="hi")]


def test_tail_block_is_last_in_system_message():
    msgs = append_pinned_context(_sys(f"base\n{PROMPT_CACHE_BOUNDARY}\ndynamic"), "use uv")
    content = msgs[0].content
    assert content.rstrip().endswith(PINNED_END)
    assert "use uv" in content
    assert content.index("dynamic") < content.index(PINNED_BEGIN)


def test_tail_block_frames_precedence():
    content = append_pinned_context(_sys("base"), "use uv")[0].content
    tail = content[content.index(PINNED_BEGIN):]
    assert "## Pinned context (always applies)" in tail
    assert "precedence" in tail


def test_tail_block_is_upserted_not_duplicated():
    msgs = append_pinned_context(_sys("base"), "first")
    msgs = append_pinned_context(msgs, "second")
    content = msgs[0].content
    assert content.count(PINNED_BEGIN) == 1
    assert "first" not in content
    assert "second" in content


def test_empty_text_removes_existing_tail_block():
    msgs = append_pinned_context(_sys("base"), "note")
    msgs = append_pinned_context(msgs, "")
    assert PINNED_BEGIN not in msgs[0].content
    assert msgs[0].content.strip() == "base"


def test_tail_stays_last_after_rules_injection_reapplied():
    msgs = _sys(f"base\n{PROMPT_CACHE_BOUNDARY}")
    msgs = list(append_prompt_injection(msgs, "rules v1"))
    msgs = append_pinned_context(msgs, "pinned")
    # Next round: injection re-upserted, then tail re-upserted.
    msgs = list(append_prompt_injection(msgs, "rules v2"))
    msgs = append_pinned_context(msgs, "pinned")
    content = msgs[0].content
    assert content.count(INJECTION_BEGIN) == 1
    assert content.count(PINNED_BEGIN) == 1
    assert content.index("rules v2") < content.index(PINNED_BEGIN)
    assert content.rstrip().endswith(PINNED_END)


def test_user_message_is_untouched_by_tail():
    msgs = append_pinned_context(_sys("base"), "note")
    assert msgs[1].content == "hi"


# ─── End to end through create_claw_agent ───────────────────────────────


def _system_after_hook(agent) -> str:
    assert agent.before_llm is not None
    out = agent.before_llm(_sys(f"base\n{PROMPT_CACHE_BOUNDARY}"))
    return out[0].content


def test_agent_puts_pinned_context_at_the_tail(tmp_path, monkeypatch):
    from clawagents.agent import create_claw_agent

    write_pinned_context("PINNED-MARKER", tmp_path)
    (tmp_path / "AGENTS.md").write_text("AGENTS-MARKER", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, skills=[])
    content = _system_after_hook(agent)
    assert content.rstrip().endswith(PINNED_END)
    assert content.index("AGENTS-MARKER") < content.index("PINNED-MARKER")


def test_agent_does_not_duplicate_pinned_context_in_rules_blob(tmp_path, monkeypatch):
    from clawagents.agent import create_claw_agent

    write_pinned_context("PINNED-MARKER", tmp_path)
    (tmp_path / "AGENTS.md").write_text("AGENTS-MARKER", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, skills=[])
    content = _system_after_hook(agent)
    assert content.count("PINNED-MARKER") == 1


def test_agent_rereads_pinned_context_every_round(tmp_path, monkeypatch):
    from clawagents.agent import create_claw_agent

    write_pinned_context("v1", tmp_path)
    monkeypatch.chdir(tmp_path)
    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, skills=[])
    first = _system_after_hook(agent)
    assert "v1" in first
    write_pinned_context("v2", tmp_path)
    second = agent.before_llm(_sys(first))[0].content
    assert "v2" in second and "v1" not in second


def test_agent_with_no_pinned_context_adds_no_tail(tmp_path, monkeypatch):
    from clawagents.agent import create_claw_agent

    monkeypatch.chdir(tmp_path)
    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, skills=[])
    if agent.before_llm is None:
        return
    assert PINNED_BEGIN not in _system_after_hook(agent)


def test_agent_discovers_rules_from_workspace_not_cwd(tmp_path, monkeypatch):
    """Library callers pass workspace= without chdir; rules must still load."""
    from clawagents.agent import create_claw_agent

    ws = tmp_path / "ws"
    ws.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (ws / "AGENTS.md").write_text("AGENTS-MARKER", encoding="utf-8")
    write_pinned_context("PINNED-MARKER", ws)
    monkeypatch.chdir(elsewhere)
    agent = create_claw_agent(model=_StubLLM(), workspace=ws, skills=[])
    content = _system_after_hook(agent)
    assert "AGENTS-MARKER" in content
    assert "PINNED-MARKER" in content
    assert os.getcwd() == str(elsewhere.resolve()) or os.path.realpath(os.getcwd()) == str(elsewhere.resolve())
