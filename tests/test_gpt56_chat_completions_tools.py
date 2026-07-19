"""Chat Completions + tools compat for GPT-5.5 / GPT-5.6."""

from clawagents.providers.llm import (
    _apply_tool_reasoning_compat,
    _chat_completions_needs_reasoning_none,
    model_supports_reasoning_effort,
    normalize_reasoning_effort,
    prefers_responses_api,
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


def test_openai_vendor_prefix_stripped_for_classifiers():
    """Catalog ids like openai.gpt-5.6-luna must match bare gpt-5.6-luna."""
    assert _chat_completions_needs_reasoning_none("openai.gpt-5.6-luna")
    assert prefers_responses_api("openai.gpt-5.6-luna", has_tools=True)
    assert prefers_responses_api("openai.gpt-5.6-luna", has_tools=False)
    assert model_supports_reasoning_effort("openai.gpt-5.6-luna")


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

    # Preferred effort is applied, then tools+5.6 still forces none.
    kwargs4: dict = {"model": "gpt-5.6-luna"}
    _apply_tool_reasoning_compat(
        kwargs4, model="gpt-5.6-luna", has_tools=True, preferred="high",
    )
    assert kwargs4["reasoning_effort"] == "none"

    # Without tools, preferred effort sticks.
    kwargs5: dict = {"model": "gpt-5.6-luna"}
    _apply_tool_reasoning_compat(
        kwargs5, model="gpt-5.6-luna", has_tools=False, preferred="high",
    )
    assert kwargs5["reasoning_effort"] == "high"

    # o3 with tools keeps preferred effort.
    kwargs6: dict = {"model": "o3"}
    _apply_tool_reasoning_compat(
        kwargs6, model="o3", has_tools=True, preferred="medium",
    )
    assert kwargs6["reasoning_effort"] == "medium"


def test_normalize_and_support_helpers():
    assert normalize_reasoning_effort("Light") == "low"
    assert normalize_reasoning_effort("Extra High") == "xhigh"
    assert normalize_reasoning_effort("") is None
    assert model_supports_reasoning_effort("gpt-5.6-luna")
    assert model_supports_reasoning_effort("o3-mini")
    assert not model_supports_reasoning_effort("gpt-4o")
