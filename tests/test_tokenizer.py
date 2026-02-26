"""Tests for the tokenizer module."""

import math
import pytest
from unittest.mock import patch, MagicMock

from clawagents.tokenizer import (
    count_tokens,
    count_tokens_content,
    count_messages_tokens,
    _encoding_for_model,
    _CHARS_PER_TOKEN_FALLBACK,
)
from clawagents.providers.llm import LLMMessage


class TestEncodingResolution:
    """Test model → encoding name mapping."""

    def test_gpt5_uses_o200k(self):
        assert _encoding_for_model("gpt-5") == "o200k_base"
        assert _encoding_for_model("gpt-5-mini") == "o200k_base"
        assert _encoding_for_model("gpt-5-nano") == "o200k_base"

    def test_gpt4o_uses_o200k(self):
        assert _encoding_for_model("gpt-4o") == "o200k_base"
        assert _encoding_for_model("gpt-4o-mini") == "o200k_base"

    def test_gpt4_uses_cl100k(self):
        assert _encoding_for_model("gpt-4") == "cl100k_base"
        assert _encoding_for_model("gpt-4-turbo") == "cl100k_base"

    def test_unknown_model_uses_default(self):
        assert _encoding_for_model("some-custom-model") == "o200k_base"

    def test_none_model_uses_default(self):
        assert _encoding_for_model(None) == "o200k_base"

    def test_gemini_uses_default(self):
        # Gemini models aren't in the tiktoken map, should use default
        assert _encoding_for_model("gemini-3-flash") == "o200k_base"


class TestCountTokens:
    """Test the core count_tokens function with real tiktoken."""

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_nonempty_string_returns_positive(self):
        assert count_tokens("hello world") > 0

    def test_known_token_count(self):
        # "hello world" with o200k_base should be 2 tokens
        # The heuristic would give ceil(11/4) = 3
        tokens = count_tokens("hello world", model="gpt-5")
        assert tokens == 2  # tiktoken gives exact count

    def test_long_text_more_accurate_than_heuristic(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tiktoken_count = count_tokens(text, model="gpt-5")
        heuristic_count = math.ceil(len(text) / _CHARS_PER_TOKEN_FALLBACK)
        # They should differ — tiktoken is more accurate
        assert tiktoken_count != heuristic_count

    def test_code_content(self):
        code = 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n'
        tokens = count_tokens(code)
        assert tokens > 0
        # Heuristic: ceil(73/4) = 19, tiktoken will differ
        assert tokens != math.ceil(len(code) / _CHARS_PER_TOKEN_FALLBACK)


class TestCountTokensContent:
    """Test the content-aware token counting."""

    def test_string_content(self):
        result = count_tokens_content("hello world")
        assert result > 0

    def test_multimodal_content(self):
        content = [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        result = count_tokens_content(content)
        # Should include image token estimate (~500)
        assert result >= 500

    def test_multiplier(self):
        base = count_tokens_content("hello world", multiplier=1.0)
        doubled = count_tokens_content("hello world", multiplier=2.0)
        assert doubled == math.ceil(base * 2.0)


class TestCountMessagesTokens:
    """Test message-list token counting."""

    def test_single_message(self):
        msgs = [LLMMessage(role="user", content="hello")]
        result = count_messages_tokens(msgs)
        # Should be token count + per-message overhead (4)
        assert result > count_tokens("hello")

    def test_multiple_messages(self):
        msgs = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello!"),
        ]
        result = count_messages_tokens(msgs)
        assert result > 0

    def test_empty_messages(self):
        result = count_messages_tokens([])
        assert result == 0


class TestFallbackMode:
    """Test graceful degradation when tiktoken is unavailable."""

    def test_fallback_uses_heuristic(self):
        # Simulate tiktoken not being available by clearing cache and patching
        from clawagents import tokenizer
        tokenizer._get_encoder.cache_clear()

        original_warned = tokenizer._fallback_warned
        try:
            tokenizer._fallback_warned = False
            with patch.dict("sys.modules", {"tiktoken": None}):
                tokenizer._get_encoder.cache_clear()
                result = count_tokens("hello world")
                # Should fall back to heuristic: ceil(11/4) = 3
                assert result == math.ceil(len("hello world") / _CHARS_PER_TOKEN_FALLBACK)
        finally:
            tokenizer._fallback_warned = original_warned
            tokenizer._get_encoder.cache_clear()
