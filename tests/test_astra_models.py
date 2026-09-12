"""Astra routing/capabilities, grounded in OpenAI and AWS model cards."""
import pytest
from clawagents.config.config import EngineConfig
from clawagents.graph.model_profiles import resolve_model_profile
from clawagents.providers.llm import (
    OpenAIProvider, create_provider, prefers_responses_api,
    model_supports_reasoning_effort, openai_model_rejects_temperature,
)

@pytest.mark.parametrize("model", ["gpt-6-astra", "openai.gpt-6-astra"])
def test_astra_capabilities(model):
    assert prefers_responses_api(model)
    assert model_supports_reasoning_effort(model)
    assert openai_model_rejects_temperature(model)
    profile = resolve_model_profile(model)
    assert profile["max_input_tokens"] == 1_050_000
    assert profile["max_output_tokens"] == 128_000
    assert profile["long_context_threshold"] == 272_000

@pytest.mark.parametrize("effort, expected", [("none", "low"), ("minimal", "low"), ("max", "max")])
def test_astra_request_parameters(effort, expected):
    provider = OpenAIProvider(EngineConfig(openai_model="gpt-6-astra", openai_api_key="test", max_tokens=200_000, reasoning_effort=effort))
    kwargs = provider._responses_kwargs([{"role": "user", "content": "hello"}], None)
    assert kwargs["max_output_tokens"] == 128_000
    assert "temperature" not in kwargs
    assert kwargs["reasoning"]["effort"] == expected

@pytest.mark.parametrize("model", ["gpt-6-astra", "openai.gpt-6-astra"])
def test_astra_mantle_routes_responses(model, monkeypatch):
    monkeypatch.setenv("BEDROCK_API_KEY", "test")
    config = EngineConfig(openai_api_key="test", openai_model=model, openai_base_url="https://bedrock-mantle.us-west-2.api.aws/v1", openai_wire_api="chat_completions")
    provider = create_provider(model, config)
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "openai.gpt-6-astra"
    assert provider._base_url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
    assert provider._should_use_responses(True)

def test_astra_mantle_region_error_keeps_region(monkeypatch):
    monkeypatch.setenv("BEDROCK_API_KEY", "test")
    config = EngineConfig(openai_model="openai.gpt-6-astra", openai_base_url="https://bedrock-mantle.us-east-1.api.aws/v1")
    with pytest.raises(ValueError, match="us-west-2"):
        create_provider("openai.gpt-6-astra", config)
    assert "us-east-1" in config.openai_base_url

@pytest.mark.parametrize("mantle", [False, True])
def test_astra_chat_uses_responses_and_preserves_tools(mantle, monkeypatch):
    import asyncio
    from clawagents.providers.llm import LLMMessage, LLMResponse, NativeToolSchema

    model = "openai.gpt-6-astra" if mantle else "gpt-6-astra"
    config = EngineConfig(
        openai_model=model, openai_api_key="test", reasoning_effort="max",
        openai_base_url="https://bedrock-mantle.us-west-2.api.aws/v1" if mantle else "",
    )
    provider = create_provider(model, config)
    captured = {}

    async def responses(messages, on_chunk, cancel_event, oai_tools, **kwargs):
        captured.update(provider._responses_kwargs(messages, oai_tools))
        return LLMResponse(content="verified", model=model, tokens_used=0)

    monkeypatch.setattr(provider, "_stream_with_retry_responses", responses)
    result = asyncio.run(provider.chat(
        [LLMMessage("user", "Check this file")],
        tools=[NativeToolSchema("read_file", "Read a file", {"path": {"type": "string"}})],
    ))
    assert result.content == "verified"
    assert captured["model"] == model
    assert captured["tools"][0]["name"] == "read_file"
    assert captured["reasoning"] == {"effort": "max"}
    assert "temperature" not in captured
