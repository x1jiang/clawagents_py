from clawagents.providers.llm import LLMMessage
from clawagents.prompts import (
    PROMPT_CACHE_BOUNDARY,
    append_prompt_injection,
    build_prompt_injection,
    build_system_prompt,
)


def test_build_system_prompt_places_tools_before_cache_boundary():
    system = build_system_prompt(
        base_prompt="base instructions",
        tool_description="tool schemas",
        lesson_preamble="\nlessons",
    )

    assert system == (
        "base instructions\nlessons\n\n"
        "tool schemas\n"
        f"{PROMPT_CACHE_BOUNDARY}"
    )


def test_append_prompt_injection_updates_system_message_without_mutating_original():
    messages = [
        LLMMessage(role="system", content="base"),
        LLMMessage(role="user", content="task"),
    ]
    injection = build_prompt_injection(
        memory_content="## Agent Memory\nremember this",
        skill_summaries="## Available Skills\n- **review**: Review code",
    )

    updated = append_prompt_injection(messages, injection)

    assert messages[0].content == "base"
    assert updated[0].content == (
        "base\n\n"
        "## Agent Memory\nremember this\n\n"
        "## Available Skills\n- **review**: Review code"
    )
    assert updated[1] is messages[1]


def test_append_prompt_injection_accepts_dict_messages_for_legacy_hooks():
    messages = [{"role": "system", "content": "base"}]

    updated = append_prompt_injection(messages, "legacy injection")

    assert updated[0].content == "base\n\nlegacy injection"
    assert messages[0]["content"] == "base"


def test_append_prompt_injection_returns_original_when_no_injection():
    messages = [LLMMessage(role="system", content="base")]

    assert append_prompt_injection(messages, None) is messages
