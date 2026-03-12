"""Tests for config module."""

import os
from unittest.mock import patch

from clawagents.config.config import (
    EngineConfig,
    is_gemini_model,
    is_anthropic_model,
    get_default_model,
)


def test_is_gemini_model():
    assert is_gemini_model("gemini-3-flash")
    assert is_gemini_model("Gemini-Pro")
    assert not is_gemini_model("gpt-5")


def test_is_anthropic_model():
    assert is_anthropic_model("claude-sonnet-4-5")
    assert is_anthropic_model("Claude-3.5")
    assert is_anthropic_model("anthropic/claude")
    assert not is_anthropic_model("gpt-5")


def test_default_model_openai():
    config = EngineConfig(openai_api_key="test-key", openai_model="gpt-5-nano")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROVIDER", None)
        assert get_default_model(config) == "gpt-5-nano"


def test_default_model_gemini_fallback():
    config = EngineConfig(
        openai_api_key="", gemini_api_key="test-key", gemini_model="gemini-3-flash"
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROVIDER", None)
        assert get_default_model(config) == "gemini-3-flash"


def test_default_model_anthropic_fallback():
    config = EngineConfig(
        openai_api_key="", gemini_api_key="",
        anthropic_api_key="test-key", anthropic_model="claude-sonnet-4-5",
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROVIDER", None)
        assert get_default_model(config) == "claude-sonnet-4-5"


def test_engine_config_defaults():
    env_override = {
        "OPENAI_API_KEY": "", "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "",
        "MAX_TOKENS": "8192", "TEMPERATURE": "0", "CONTEXT_WINDOW": "1000000",
    }
    with patch.dict(os.environ, env_override, clear=False):
        config = EngineConfig(
            openai_api_key="", openai_model="gpt-5-nano",
            gemini_api_key="", anthropic_api_key="",
            max_tokens=8192, temperature=0.0, context_window=1000000,
        )
        assert config.max_tokens == 8192
        assert config.temperature == 0.0
        assert config.context_window == 1000000
