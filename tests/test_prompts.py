from clawagents.providers.llm import LLMMessage
from clawagents.prompts import (
    INJECTION_BEGIN,
    INJECTION_END,
    PROMPT_CACHE_BOUNDARY,
    append_model_identity,
    append_prompt_injection,
    build_prompt_injection,
    build_system_prompt,
    model_identity_section,
)
from clawagents.prompts.cache_align import normalize_stable_prefix


def test_model_identity_section_names_configured_model():
    block = model_identity_section("gemini", "gemini-3.1-flash-lite")
    assert "`gemini/gemini-3.1-flash-lite`" in block
    assert "Do not claim to be a different model" in block


def test_append_model_identity_is_idempotent_and_skips_empty_model():
    assert append_model_identity("base", "openai", "") == "base"
    once = append_model_identity("base", "openai", "gpt-5.6-luna")
    assert once.count("## Model identity") == 1
    assert "`openai/gpt-5.6-luna`" in once
    twice = append_model_identity(once, "openai", "gpt-5.6-luna")
    assert twice == once


def test_model_identity_stays_in_static_cache_prefix():
    base = append_model_identity("base instructions", "openai", "gpt-5.6-luna")
    system = build_system_prompt(
        base_prompt=base,
        tool_description="tools",
        lesson_preamble="lessons",
    )
    prefix, _, _ = system.partition(PROMPT_CACHE_BOUNDARY)
    assert "## Model identity" in prefix
    assert "gpt-5.6-luna" in prefix
    assert "lessons" not in prefix


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
    assert INJECTION_BEGIN in updated[0].content
    assert "## Agent Memory\nremember this" in updated[0].content
    assert "## Available Skills\n- **review**: Review code" in updated[0].content
    assert INJECTION_END in updated[0].content
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

    assert INJECTION_BEGIN in updated[0].content
    assert "legacy injection" in updated[0].content
    assert messages[0]["content"] == "base"


def test_append_prompt_injection_returns_original_when_no_injection():
    messages = [LLMMessage(role="system", content="base")]

    assert append_prompt_injection(messages, None) is messages
