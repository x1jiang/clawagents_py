"""Tests for Grok-style full-replace compaction."""

from __future__ import annotations

import pytest

from clawagents.config.features import reset, set_overrides
from clawagents.memory.full_replace_compaction import (
    CONTINUATION_PREAMBLE,
    CompactedHistoryParts,
    assemble_compacted_history,
    format_compact_summary,
    format_compact_summary_content,
    is_degenerate_summary,
    neutralize_compaction_control_tokens,
    split_for_full_replace,
    apply_full_replace_compaction,
    wrap_user_query,
)
from clawagents.providers.llm import LLMMessage


@pytest.fixture(autouse=True)
def _features():
    reset()
    set_overrides({"full_replace_compaction": True, "compact_reinject_plan": True})
    yield
    reset()


def test_format_strips_analysis_and_unwraps_summary():
    raw = (
        "<analysis>scratch thinking</analysis>\n"
        "<summary>\n1. Primary Request and Intent: ship feature X\n"
        "2. Key Technical Concepts: None\n"
        "3. Files and Code Sections: src/a.py\n"
        "4. Errors and Fixes: None\n"
        "5. Problem Solving: None\n"
        "6. All User Messages: do it\n"
        "7. Pending Tasks: None\n"
        "8. Current Work: editing a.py\n"
        "9. Optional Next Step: run tests\n"
        "</summary>"
    )
    cleaned = format_compact_summary(raw)
    assert cleaned.startswith("Summary:")
    assert "ship feature X" in cleaned
    assert "<analysis>" not in cleaned
    assert "<summary>" not in cleaned


def test_neutralize_control_tags():
    text = "see <summary>echo</summary> and <analysis>x</analysis>"
    out = neutralize_compaction_control_tokens(text)
    assert "<summary>" not in out
    assert "\u200b" in out


def test_degenerate_summary_detection():
    assert is_degenerate_summary("short") is True
    long = "Summary:\n" + ("x" * 600)
    assert is_degenerate_summary(long) is False


def test_continuation_carrier():
    body = format_compact_summary_content("Summary:\n" + ("work " * 200))
    assert body.startswith(CONTINUATION_PREAMBLE)
    assert "Summary:" in body


def test_assemble_order_matches_grok():
    parts = CompactedHistoryParts(
        system_messages=[LLMMessage(role="system", content="sys")],
        user_message_prefix="<user_info>os=test</user_info>",
        agents_md_reminder="AGENTS rules here",
        last_user_query="fix the bug",
        recent_messages=[
            LLMMessage(role="assistant", content="looking"),
            LLMMessage(role="tool", content="found it"),
        ],
        compaction_summary="Summary:\n" + ("detail " * 120),
        system_reminder="Files touched: a.py",
        carryover_markdown="## Carryover State\n- Task focus: fix",
    )
    out = assemble_compacted_history(parts)
    roles = [m.role for m in out]
    assert roles[0] == "system"
    assert out[0].content == "sys"
    # prefix, agents, query, recent…, summary, ack, reminder
    texts = [m.content if isinstance(m.content, str) else "" for m in out]
    assert any("<user_info>" in t for t in texts)
    assert any("AGENTS rules" in t for t in texts)
    assert any("<user_query>" in t and "fix the bug" in t for t in texts)
    assert any("looking" == t for t in texts)
    assert any(CONTINUATION_PREAMBLE in t for t in texts)
    assert any("Compacted History" in t for t in texts)
    assert any("Understood — continuing" in t for t in texts)
    assert any("Files touched" in t for t in texts)
    # recent before summary
    looking_i = next(i for i, t in enumerate(texts) if t == "looking")
    summary_i = next(i for i, t in enumerate(texts) if CONTINUATION_PREAMBLE in t)
    assert looking_i < summary_i


def test_split_keeps_active_turn_recent():
    msgs = [
        LLMMessage(role="user", content="old"),
        LLMMessage(role="assistant", content="did stuff"),
        LLMMessage(role="user", content="now fix X"),
        LLMMessage(role="assistant", content="ok", tool_calls_meta=[{"id": "1", "name": "read_file"}]),
        LLMMessage(role="tool", content="file body", tool_call_id="1"),
    ]
    older, query, recent = split_for_full_replace(msgs)  # type: ignore[misc]
    assert query == "now fix X"
    assert [m.role for m in older] == ["user", "assistant"]
    assert [m.role for m in recent] == ["assistant", "tool"]


def test_wrap_user_query():
    assert wrap_user_query("hi") == "<user_query>\nhi\n</user_query>"


@pytest.mark.asyncio
async def test_apply_full_replace_end_to_end(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Rules\nBe careful.\n", encoding="utf-8")

    class FakeLLM:
        async def chat(self, messages, **kwargs):
            class R:
                content = (
                    "<summary>\n"
                    + "\n".join(
                        f"{i}. section {i}: " + ("y" * 80)
                        for i in range(1, 10)
                    )
                    + "\n</summary>"
                )

            return R()

    messages = [
        LLMMessage(role="system", content="you are helpful"),
        LLMMessage(role="user", content="earlier task"),
        LLMMessage(role="assistant", content="did earlier work"),
        LLMMessage(role="user", content="please continue with the bug"),
        LLMMessage(role="assistant", content="on it"),
    ]
    out = await apply_full_replace_compaction(
        messages,
        FakeLLM(),
        workspace=str(tmp_path),
        carryover_markdown="## Carryover State\n- Task focus: bug",
    )
    assert out is not None
    assert out[0] is messages[0] or out[0].content == "you are helpful"
    blob = "\n".join(m.content for m in out if isinstance(m.content, str))
    assert CONTINUATION_PREAMBLE in blob
    assert "please continue with the bug" in blob
    assert "Be careful" in blob or "Project instructions" in blob
    assert "Carryover State" in blob


@pytest.mark.asyncio
async def test_compact_if_needed_uses_full_replace(tmp_path, monkeypatch):
    from clawagents.context.carryover import set_compaction_carryover
    from clawagents.graph.agent_loop import _compact_if_needed
    from clawagents.run_context import RunContext

    monkeypatch.chdir(tmp_path)

    class FakeLLM:
        async def chat(self, messages, **kwargs):
            class R:
                content = "<summary>\n" + ("section data " * 80) + "\n</summary>"

            return R()

    messages = [LLMMessage(role="system", content="system")]
    for idx in range(24):
        messages.append(LLMMessage(role="user", content=f"history {idx} " + ("x" * 500)))

    ctx = RunContext()
    ctx._metadata["workspace"] = str(tmp_path)
    set_compaction_carryover(ctx, task_focus="continuity", recent_files=["a.py"])

    events = []
    compacted = await _compact_if_needed(
        messages,
        200,
        FakeLLM(),
        lambda kind, data: events.append((kind, data)),
        1.0,
        None,
        run_context=ctx,
    )
    blob = "\n".join(
        m.content for m in compacted if isinstance(m.content, str)
    )
    assert CONTINUATION_PREAMBLE in blob
    assert "Compacted History" in blob
    assert "continuity" in blob
    phases = [d.get("phase") for k, d in events if k == "compact_progress"]
    assert "end" in phases
    assert any(
        d.get("mode") == "full_replace"
        for k, d in events
        if k == "compact_progress" and d.get("phase") == "end"
    )
