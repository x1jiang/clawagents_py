"""GPT-6.1 Sol contracts from its official OpenAI model card."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawagents.config.config import EngineConfig
from clawagents.graph.model_profiles import resolve_model_profile
from clawagents.providers.llm import (
    LLMMessage, NativeToolSchema, OpenAIProvider, model_supports_reasoning_effort,
    openai_model_rejects_temperature, prefers_responses_api,
)


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "openai.gpt-6.1-sol"])
def test_gpt61_sol_capabilities(model):
    assert prefers_responses_api(model)
    assert model_supports_reasoning_effort(model)
    assert openai_model_rejects_temperature(model)
    profile = resolve_model_profile(model)
    assert profile["max_input_tokens"] == 1_050_000
    assert profile["max_output_tokens"] == 128_000
    assert profile["long_context_threshold"] == 272_000


@pytest.mark.parametrize("effort", ["", "none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_gpt61_sol_responses_parameters(effort):
    provider = OpenAIProvider(EngineConfig(openai_model="gpt-6.1-sol", openai_api_key="test",
        max_tokens=200_000, reasoning_effort=effort, temperature=0.3))
    wire = provider._responses_kwargs([{"role": "user", "content": "hello"}], None)
    assert wire["max_output_tokens"] == 128_000
    assert "temperature" not in wire
    expected = "low" if effort in ("none", "minimal") else effort
    assert wire.get("reasoning") == ({"effort": expected} if expected else None)


@pytest.mark.parametrize("wire_api", ["auto", "chat_completions"])
async def test_gpt61_sol_tool_calling_contract(wire_api, monkeypatch):
    provider = OpenAIProvider(EngineConfig(openai_model="gpt-6.1-sol", openai_api_key="test",
        openai_wire_api=wire_api, reasoning_effort="none"))
    responses = AsyncMock(return_value=SimpleNamespace(content="ok"))
    chat = AsyncMock()
    monkeypatch.setattr(provider, "_stream_with_retry_responses", responses)
    provider.client.chat.completions.create = chat
    messages = [LLMMessage("user", "Read a file")]
    tools = [NativeToolSchema("read_file", "Read", {"path": {"type": "string"}})]
    if wire_api == "auto":
        await provider.chat(messages, tools=tools)
        responses.assert_awaited_once()
    else:
        with pytest.raises(ValueError, match="requires the Responses API for tool calling"):
            await provider.chat(messages, tools=tools)
    chat.assert_not_awaited()


async def test_gpt61_sol_tool_endpoint_failure_never_falls_back(monkeypatch):
    provider = OpenAIProvider(EngineConfig(openai_model="gpt-6.1-sol", openai_api_key="test"))
    unavailable = RuntimeError("endpoint does not support Responses")
    monkeypatch.setattr(provider, "_stream_with_retry_responses", AsyncMock(side_effect=unavailable))
    chat = AsyncMock()
    provider.client.chat.completions.create = chat
    with pytest.raises(RuntimeError, match="does not support Responses"):
        await provider.chat([LLMMessage("user", "Read")],
            tools=[NativeToolSchema("read_file", "Read", {})])
    chat.assert_not_awaited()
    assert not provider._force_chat_completions


@pytest.mark.parametrize("effort", ["none", "minimal", "high", "max"])
async def test_gpt61_sol_explicit_chat_without_tools(effort):
    provider = OpenAIProvider(EngineConfig(openai_model="gpt-6.1-sol", openai_api_key="test",
        openai_wire_api="chat_completions", reasoning_effort=effort, max_tokens=200_000))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None),
        finish_reason="stop")], usage=None)
    provider.client.chat.completions.create = AsyncMock(return_value=response)
    await provider._request_once([{"role": "user", "content": "hello"}])
    wire = provider.client.chat.completions.create.call_args.kwargs
    assert wire["reasoning_effort"] == ("low" if effort in ("none", "minimal") else effort)
    assert wire["max_completion_tokens"] == 128_000
    assert "temperature" not in wire
