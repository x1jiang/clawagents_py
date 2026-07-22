"""Opus 4.7+ must not receive deprecated temperature on Messages API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clawagents.providers.llm import (
    anthropic_model_rejects_sampling_params,
    is_mantle_openai_responses_model,
    openai_model_rejects_temperature,
    _mantle_openai_model_id,
    _with_temperature,
)


@pytest.mark.parametrize(
    "model,rejects",
    [
        ("anthropic.claude-opus-4-8", True),
        ("claude-opus-4-8", True),
        ("claude-opus-4.8", True),
        ("us.anthropic.claude-opus-4-7-20250514-v1:0", True),
        ("claude-opus-4-6", False),
        ("claude-opus-4-5", False),
        ("claude-sonnet-4-5", False),
        ("anthropic.claude-sonnet-4-5", False),
        ("anthropic.claude-sonnet-5", True),
        ("anthropic.claude-fable-5", True),
        ("gpt-4o", False),
    ],
)
def test_anthropic_model_rejects_sampling_params(model: str, rejects: bool):
    assert anthropic_model_rejects_sampling_params(model) is rejects


def _fake_message_response():
    usage = MagicMock(input_tokens=1, output_tokens=1)
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    block = MagicMock(type="text", text="hi")
    return MagicMock(content=[block], usage=usage)


@pytest.mark.asyncio
async def test_anthropic_provider_omits_temperature_for_opus_48():
    from clawagents.config.config import EngineConfig
    from clawagents.providers.llm import AnthropicProvider, LLMMessage

    cfg = EngineConfig(
        anthropic_api_key="sk-ant-test",
        anthropic_model="anthropic.claude-opus-4-8",
        temperature=0.0,
    )
    with patch("clawagents.providers.llm._HAS_ANTHROPIC", True), patch(
        "clawagents.providers.llm._anthropic_mod"
    ) as mod:
        client = MagicMock()
        mod.AsyncAnthropic.return_value = client
        provider = AnthropicProvider(cfg)
        provider.model = "anthropic.claude-opus-4-8"

        captured: dict = {}

        async def _create(**kwargs):
            captured.update(kwargs)
            return _fake_message_response()

        client.messages.create = AsyncMock(side_effect=_create)
        await provider.chat([LLMMessage(role="user", content="hi")])

    assert "temperature" not in captured
    assert captured.get("model") == "anthropic.claude-opus-4-8"


@pytest.mark.parametrize(
    "model,rejects",
    [
        ("openai.gpt-5.6-luna", True),
        ("gpt-5.6-luna", True),
        ("openai.gpt-5.5", True),
        ("o3-mini", True),
        ("xai.grok-4.3", True),
        ("grok-4.3", True),
        ("gpt-4o", False),
        ("openai.gpt-oss-20b", False),
    ],
)
def test_openai_model_rejects_temperature(model: str, rejects: bool):
    assert openai_model_rejects_temperature(model) is rejects


def test_with_temperature_omits_for_gpt56_luna():
    kwargs = _with_temperature({"model": "x"}, "openai.gpt-5.6-luna", 0.0)
    assert "temperature" not in kwargs
    kwargs2 = _with_temperature({"model": "x"}, "gpt-4o", 0.2)
    assert kwargs2["temperature"] == 0.2


def test_mantle_openai_responses_accepts_bare_gpt56():
    assert is_mantle_openai_responses_model("gpt-5.6-luna")
    assert is_mantle_openai_responses_model("openai.gpt-5.6-luna")
    assert _mantle_openai_model_id("gpt-5.6-luna") == "openai.gpt-5.6-luna"
    assert not is_mantle_openai_responses_model("openai.gpt-oss-20b")


@pytest.mark.asyncio
async def test_anthropic_provider_keeps_temperature_for_sonnet():
    from clawagents.config.config import EngineConfig
    from clawagents.providers.llm import AnthropicProvider, LLMMessage

    cfg = EngineConfig(
        anthropic_api_key="sk-ant-test",
        anthropic_model="claude-sonnet-4-5",
        temperature=0.0,
    )
    with patch("clawagents.providers.llm._HAS_ANTHROPIC", True), patch(
        "clawagents.providers.llm._anthropic_mod"
    ) as mod:
        client = MagicMock()
        mod.AsyncAnthropic.return_value = client
        provider = AnthropicProvider(cfg)

        captured: dict = {}

        async def _create(**kwargs):
            captured.update(kwargs)
            return _fake_message_response()

        client.messages.create = AsyncMock(side_effect=_create)
        await provider.chat([LLMMessage(role="user", content="hi")])

    assert captured.get("temperature") == 0.0
