"""Opus 4.7+ must not receive deprecated temperature on Messages API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clawagents.providers.llm import anthropic_model_rejects_sampling_params


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
