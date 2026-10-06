"""Fast service tier is explicit and limited to supported direct OpenAI models."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawagents.config.config import EngineConfig
from clawagents.providers.llm import OpenAIProvider, supports_openai_fast_mode


@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
def test_supported_models(model):
    assert supports_openai_fast_mode(model)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-6", "openai.gpt-6-sol", "claude-sonnet-4-5"])
def test_unsupported_models(model):
    assert not supports_openai_fast_mode(model)


@pytest.mark.parametrize("enabled,tier", [(False, "default"), (True, "fast")])
async def test_fast_tier_on_responses_and_chat_completions(enabled, tier):
    provider = OpenAIProvider(EngineConfig(
        openai_model="gpt-6-sol", openai_api_key="test", openai_fast_mode=enabled,
    ))
    assert provider._responses_kwargs([{"role": "user", "content": "hello"}], None)["service_tier"] == tier
    assert provider._responses_kwargs([{"role": "user", "content": "hello"}], None, stream=True)["service_tier"] == tier

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=None,
    )
    provider.client.chat.completions.create = AsyncMock(return_value=response)
    await provider._request_once([{"role": "user", "content": "hello"}])
    assert provider.client.chat.completions.create.call_args.kwargs["service_tier"] == tier


def test_fast_rejects_unsupported_endpoint_and_model():
    for config in (
        EngineConfig(openai_model="gpt-4o", openai_api_key="test", openai_fast_mode=True),
        EngineConfig(openai_model="gpt-6-sol", openai_api_key="test", openai_fast_mode=True,
                     openai_base_url="https://proxy.example.test/v1"),
    ):
        with pytest.raises(ValueError, match="Fast mode requires"):
            OpenAIProvider(config)
