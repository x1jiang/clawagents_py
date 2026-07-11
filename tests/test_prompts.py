from clawagents.providers.llm import LLMMessage
from clawagents.prompts import (
    PROMPT_CACHE_BOUNDARY,
    append_prompt_injection,
    build_prompt_injection,
    build_system_prompt,
)
from clawagents.prompts.cache_align import normalize_stable_prefix


def test_build_system_prompt_places_tools_before_cache_boundary():
    system = build_system_prompt(
        base_prompt="base instructions",
        tool_description="tool schemas",
        lesson_preamble="\nlessons",
    )

    assert PROMPT_CACHE_BOUNDARY in system
    prefix, _, suffix = system.partition(PROMPT_CACHE_BOUNDARY)
    assert "base instructions" in prefix
    assert "tool schemas" in prefix
    assert "lessons" in suffix
    # Lessons must sit after the boundary so they don't bust the prefix cache.
    assert "lessons" not in prefix


def test_normalize_stable_prefix_collapses_whitespace():
    assert normalize_stable_prefix("a\n\n\n\nb  \n") == "a\n\nb\n"


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


def test_append_prompt_injection_keeps_cache_boundary_stable():
    messages = [
        LLMMessage(
            role="system",
            content=f"static tools\n{PROMPT_CACHE_BOUNDARY}\nold dynamic\n",
        )
    ]
    updated = append_prompt_injection(messages, "new injection")
    content = updated[0].content
    assert content.startswith(f"static tools\n{PROMPT_CACHE_BOUNDARY}")
    assert "new injection" in content
    # Static prefix before boundary unchanged.
    assert content.split(PROMPT_CACHE_BOUNDARY, 1)[0] == "static tools\n"


def test_append_prompt_injection_accepts_dict_messages_for_legacy_hooks():
    messages = [{"role": "system", "content": "base"}]

    updated = append_prompt_injection(messages, "legacy injection")

    assert updated[0].content == "base\n\nlegacy injection"
    assert messages[0]["content"] == "base"


def test_append_prompt_injection_returns_original_when_no_injection():
    messages = [LLMMessage(role="system", content="base")]

    assert append_prompt_injection(messages, None) is messages
