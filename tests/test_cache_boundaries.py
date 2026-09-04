"""Hermetic tests for prompt cache boundary alignment and cache breakpoints."""

from typing import Any

from clawagents.prompts.cache_align import normalize_stable_prefix, sort_tool_names
from clawagents.providers.llm import _apply_conversation_cache_breakpoints


def test_normalize_stable_prefix_comprehensive():
    assert normalize_stable_prefix("") == ""
    assert normalize_stable_prefix("   \n\n  ") == ""
    
    crlf_text = "line1  \r\nline2\t \r\n\r\n\r\nline3\r"
    normalized = normalize_stable_prefix(crlf_text)
    assert normalized == "line1\nline2\n\nline3\n"
    assert not any(line.endswith((" ", "\t")) for line in normalized.splitlines())


def test_sort_tool_names_stable_and_case_insensitive():
    tools = ["read_file", "Write_File", "bash", "AskUser"]
    sorted_tools = sort_tool_names(tools)
    assert sorted_tools == ["AskUser", "bash", "read_file", "Write_File"]


def test_apply_conversation_cache_breakpoints_short_history():
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
    _apply_conversation_cache_breakpoints(msgs)
    assert msgs == [{"role": "user", "content": "hello"}]


def test_apply_conversation_cache_breakpoints_string_content():
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "task description"},
        {"role": "assistant", "content": "I will execute tool x"},
        {"role": "user", "content": "please proceed"},
    ]
    _apply_conversation_cache_breakpoints(msgs)
    
    assert msgs[1]["content"] == [
        {
            "type": "text",
            "text": "I will execute tool x",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert msgs[2]["content"] == "please proceed"


def test_apply_conversation_cache_breakpoints_blocks_content():
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thought 1"},
                {"type": "text", "text": "thought 2"},
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    _apply_conversation_cache_breakpoints(msgs)

    content = msgs[1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "thought 1"}
    assert content[1] == {
        "type": "text",
        "text": "thought 2",
        "cache_control": {"type": "ephemeral"},
    }


def test_apply_conversation_cache_breakpoints_ignores_trailing_empty_user():
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "real query"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "   \n\t  "},
    ]
    _apply_conversation_cache_breakpoints(msgs)
    # The last substantive user turn is index 0; index 0 - 1 is negative, so no breakpoint added.
    assert msgs[1]["content"] == "first answer"
