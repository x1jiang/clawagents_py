"""OpenAI prompt-cache read/write extraction from usage objects."""

from __future__ import annotations

from types import SimpleNamespace

from clawagents.providers.llm import _openai_cache_tokens, _openai_cached_tokens


def test_chat_completions_prompt_tokens_details():
    usage = SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=100,
            cache_write_tokens=40,
        )
    )
    assert _openai_cache_tokens(usage) == (100, 40)
    assert _openai_cached_tokens(usage) == 100


def test_responses_input_tokens_details():
    usage = SimpleNamespace(
        input_tokens_details=SimpleNamespace(
            cached_tokens=12,
            cache_write_tokens=8,
        )
    )
    assert _openai_cache_tokens(usage) == (12, 8)


def test_dict_shaped_usage_and_aliases():
    usage = {
        "prompt_tokens_details": {
            "cached_tokens": 5,
            "cache_creation_tokens": 9,
        }
    }
    assert _openai_cache_tokens(usage) == (5, 9)


def test_missing_details_defaults_to_zero():
    assert _openai_cache_tokens(None) == (0, 0)
    assert _openai_cache_tokens(SimpleNamespace()) == (0, 0)
    assert _openai_cache_tokens({"prompt_tokens": 10}) == (0, 0)
