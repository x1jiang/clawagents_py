"""Tests for the hardened compression helpers in clawagents.memory.compaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pytest

from clawagents.memory.compaction import (
    AgentMessage,
    DEFAULT_PROTECT_FIRST,
    DEFAULT_PROTECT_LAST,
    INEFFECTIVE_SAVINGS_PCT,
    THRASH_THRESHOLD,
    compress_messages_safe,
    is_compression_thrashing,
)


@dataclass
class _StubResponse:
    content: str


class _StubLLM:
    """Minimal LLMProvider stand-in for compress_messages_safe."""

    def __init__(self, summary: str = "SUMMARY", *, fail: bool = False):
        self._summary = summary
        self._fail = fail
        self.calls = 0

    async def chat(self, messages: List[Any], **_: Any) -> _StubResponse:
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return _StubResponse(content=self._summary)


def _msg(role: str, content: str) -> AgentMessage:
    return AgentMessage(role=role, content=content)


@pytest.mark.asyncio
async def test_protects_head_and_tail():
    msgs = [
        _msg("system", "you are helpful"),
        _msg("user", "hello"),
        _msg("assistant", "hi"),
        _msg("user", "do task"),
        _msg("assistant", "ok"),
        _msg("tool", "result"),
        _msg("user", "next?"),
    ]
    res = await compress_messages_safe(
        _StubLLM("compressed-middle"),
        msgs,
        context_window=2048,
        protect_first_n=1,
        protect_last_n=2,
    )
    out = res["messages"]
    assert out[0].role == "system"
    assert out[-1].content == "next?"
    assert any(m.content == "compressed-middle" for m in out)
    assert len(res["dropped_messages_list"]) >= 1


@pytest.mark.asyncio
async def test_last_user_message_always_in_tail():
    msgs = [
        _msg("system", "sys"),
        _msg("assistant", "a1"),
        _msg("assistant", "a2"),
        _msg("assistant", "a3"),
        _msg("user", "active task"),
        _msg("assistant", "ack"),
    ]
    res = await compress_messages_safe(
        _StubLLM("S"),
        msgs,
        context_window=2048,
        protect_first_n=1,
        protect_last_n=1,
    )
    contents = [m.content for m in res["messages"]]
    assert "active task" in contents


@pytest.mark.asyncio
async def test_head_compression_note_inserted_into_system():
    msgs = [
        _msg("system", "base"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
        _msg("assistant", "a2"),
        _msg("user", "u3"),
    ]
    res = await compress_messages_safe(
        _StubLLM("S"),
        msgs,
        context_window=2048,
        protect_first_n=1,
        protect_last_n=2,
    )
    sys_msg = res["messages"][0]
    assert sys_msg.role == "system"
    assert "compacted" in (sys_msg.content or "").lower()


@pytest.mark.asyncio
async def test_summary_role_avoids_collision_with_head_and_tail():
    msgs = [
        _msg("system", "sys"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
        _msg("assistant", "a2"),
        _msg("user", "active"),
    ]
    res = await compress_messages_safe(
        _StubLLM("MID"),
        msgs,
        context_window=2048,
        protect_first_n=1,
        protect_last_n=1,
    )
    out = res["messages"]
    summary_idx = next(i for i, m in enumerate(out) if m.content == "MID")
    prev_role = out[summary_idx - 1].role
    next_role = out[summary_idx + 1].role
    assert out[summary_idx].role != prev_role or prev_role == "system"
    if next_role:
        assert out[summary_idx].role != next_role or out[summary_idx].role == "user"


@pytest.mark.asyncio
async def test_falls_back_to_static_summary_on_llm_failure():
    msgs = [
        _msg("system", "sys"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
        _msg("assistant", "a2"),
        _msg("user", "active"),
    ]
    res = await compress_messages_safe(
        _StubLLM(fail=True),
        msgs,
        context_window=2048,
        protect_first_n=1,
        protect_last_n=2,
    )
    # The fallback path inside summarize_with_fallback yields a non-empty
    # string with the "[Summarized N messages]" suffix.
    assert isinstance(res["summary"], str)
    assert res["summary"]


@pytest.mark.asyncio
async def test_no_middle_to_compress_returns_original():
    msgs = [_msg("system", "sys"), _msg("user", "go")]
    res = await compress_messages_safe(
        _StubLLM("S"), msgs, context_window=2048, protect_first_n=1, protect_last_n=2
    )
    assert res["effective"] is False
    assert [m.content for m in res["messages"]] == ["sys", "go"]


@pytest.mark.asyncio
async def test_empty_input_is_safe():
    res = await compress_messages_safe(
        _StubLLM("S"), [], context_window=2048, protect_first_n=1, protect_last_n=2
    )
    assert res["messages"] == []
    assert res["effective"] is False


@pytest.mark.asyncio
async def test_savings_pct_reflects_drop():
    msgs = (
        [_msg("system", "sys")]
        + [_msg("user" if i % 2 == 0 else "assistant", "x" * 200) for i in range(10)]
        + [_msg("user", "active")]
    )
    res = await compress_messages_safe(
        _StubLLM("tiny"),
        msgs,
        context_window=2048,
        protect_first_n=1,
        protect_last_n=2,
    )
    assert res["compression_savings_pct"] > 0


def test_thrash_detector_returns_false_when_history_short():
    assert is_compression_thrashing([]) is False
    assert is_compression_thrashing([5.0]) is False


def test_thrash_detector_flags_repeated_low_savings():
    history = [5.0, 6.0]
    assert is_compression_thrashing(history) is True


def test_thrash_detector_negative_when_recent_is_effective():
    history = [5.0, 5.0, 50.0]
    assert is_compression_thrashing(history) is False


def test_default_constants_are_sane():
    assert DEFAULT_PROTECT_FIRST >= 1
    assert DEFAULT_PROTECT_LAST >= 1
    assert INEFFECTIVE_SAVINGS_PCT > 0
    assert THRASH_THRESHOLD >= 1
