"""Chat Completions + tools compat for GPT-5.5 / GPT-5.6."""

from clawagents.providers.llm import (
    _apply_tool_reasoning_compat,
    _chat_completions_needs_reasoning_none,
)


def test_reasoning_none_required_for_56_and_55():
    assert _chat_completions_needs_reasoning_none("gpt-5.6-luna")
    assert _chat_completions_needs_reasoning_none("gpt-5.6-sol")
    assert _chat_completions_needs_reasoning_none("gpt-5.6-terra")
    assert _chat_completions_needs_reasoning_none("gpt-5.6")
    assert _chat_completions_needs_reasoning_none("gpt-5.5")
    assert _chat_completions_needs_reasoning_none("gpt-5.5-pro")
    assert not _chat_completions_needs_reasoning_none("gpt-5.4")
    assert not _chat_completions_needs_reasoning_none("gpt-4o")


def test_apply_sets_reasoning_effort_only_with_tools():
    kwargs: dict = {"model": "gpt-5.6-luna"}
    _apply_tool_reasoning_compat(kwargs, model="gpt-5.6-luna", has_tools=True)
    assert kwargs["reasoning_effort"] == "none"

    kwargs2: dict = {"model": "gpt-5.6-luna"}
    _apply_tool_reasoning_compat(kwargs2, model="gpt-5.6-luna", has_tools=False)
    assert "reasoning_effort" not in kwargs2

    kwargs3: dict = {"model": "gpt-5.4"}
    _apply_tool_reasoning_compat(kwargs3, model="gpt-5.4", has_tools=True)
    assert "reasoning_effort" not in kwargs3
