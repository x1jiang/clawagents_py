"""Regression tests for round-2 audit fixes (6.20.23)."""

from __future__ import annotations

from clawagents.graph.agent_loop import (
    _message_reuse_key,
    _reuse_messages_where_possible,
)
from clawagents.providers.llm import LLMMessage
from clawagents.tools.apply_patch import (
    _apply_search_replace,
    _parse_search_replace_hunks,
)
from clawagents.tools.skills import SkillStore


def test_reuse_does_not_swap_empty_assistant_tool_meta():
    a = LLMMessage(
        role="assistant",
        content="",
        tool_calls_meta=[{"id": "1", "name": "delete_file", "args": {}}],
    )
    b = LLMMessage(
        role="assistant",
        content="",
        tool_calls_meta=[{"id": "2", "name": "create_file", "args": {}}],
    )
    assert _message_reuse_key(a) != _message_reuse_key(b)
    rebuilt = [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "1", "name": "delete_file", "args": {}}],
        ),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "2", "name": "create_file", "args": {}}],
        ),
    ]
    out = _reuse_messages_where_possible([a, b], rebuilt)
    assert out[0] is a
    assert out[1] is b
    assert out[0].tool_calls_meta[0]["name"] == "delete_file"
    assert out[1].tool_calls_meta[0]["name"] == "create_file"


def test_apply_patch_accepts_trailing_space_on_fence():
    patch = (
        "<<<<<<< SEARCH \n"  # trailing space on marker
        "hello\n"
        "=======\n"
        "world\n"
        ">>>>>>> REPLACE\n"
    )
    hunks, status = _parse_search_replace_hunks(patch)
    assert status == "ok"
    assert hunks == [("hello", "world")]


def test_apply_patch_soft_whitespace_match():
    content = "def  foo():\n    return 1\n"
    search = "def foo():\n  return 1"
    ok, new, msg = _apply_search_replace(content, search, "def foo():\n    return 2")
    assert ok, msg
    assert "return 2" in new


def test_skill_allowed_tools_flush_left_and_crlf(tmp_path):
    from clawagents.tools.skills import parse_skill_file

    skill_md = tmp_path / "SKILL.md"
    # Flush-left dashes + CRLF — previously silent unrestricted / zero tools.
    raw = (
        b"---\r\n"
        b"name: review\r\n"
        b"description: Review code\r\n"
        b"allowed-tools:\r\n"
        b"- Read\r\n"
        b"- Bash\r\n"
        b"---\r\n"
        b"Do review.\r\n"
    )
    skill_md.write_bytes(raw)
    skill = parse_skill_file(raw.decode("utf-8"), str(skill_md))
    assert skill is not None
    assert skill.allowed_tools is not None
    lowered = [t.lower() for t in skill.allowed_tools]
    assert "read" in lowered
    assert "bash" in lowered
