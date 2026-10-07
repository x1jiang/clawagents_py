"""GPT-6 family request contracts from the official OpenAI model cards."""
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from clawagents.config.config import EngineConfig
from clawagents.graph.model_profiles import resolve_model_profile
from clawagents.providers.llm import (
    LLMMessage, NativeToolSchema, OpenAIProvider, prefers_responses_api,
    model_supports_reasoning_effort,
)

MODELS = ['gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna']


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('prefix', ['', 'openai.'])
def test_gpt6_capabilities(model, prefix):
    model = prefix + model
    assert prefers_responses_api(model)
    assert model_supports_reasoning_effort(model)
    profile = resolve_model_profile(model)
    assert profile['max_input_tokens'] == 1_050_000
    assert profile['max_output_tokens'] == 128_000
    assert profile['long_context_threshold'] == 272_000


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('effort', ['', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
def test_gpt6_responses_request(model, effort):
    provider = OpenAIProvider(EngineConfig(openai_model=model, openai_api_key='test',
        max_tokens=200_000, reasoning_effort=effort, temperature=0.3))
    kwargs = provider._responses_kwargs([{'role': 'user', 'content': 'hello'}], None)
    assert kwargs['max_output_tokens'] == 128_000
    assert 'temperature' not in kwargs
    expected = 'low' if effort == 'minimal' or (model == 'gpt-6-astra' and effort == 'none') else effort
    assert kwargs.get('reasoning') == ({'effort': expected} if expected else None)


@pytest.mark.parametrize('model', MODELS)
async def test_gpt6_tool_chat_routes_responses(model, monkeypatch):
    provider = OpenAIProvider(EngineConfig(openai_model=model, openai_api_key='test', reasoning_effort='high'))
    result = SimpleNamespace(content='ok')
    responses = AsyncMock(return_value=result)
    monkeypatch.setattr(provider, '_stream_with_retry_responses', responses)
    await provider.chat([LLMMessage('user', 'Read a file')],
        tools=[NativeToolSchema('read_file', 'Read', {'path': {'type': 'string'}})])
    responses.assert_awaited_once()


@pytest.mark.parametrize('model', ['gpt-6-sol', 'gpt-6-luna'])
@pytest.mark.parametrize('has_tools', [False, True])
async def test_gpt6_explicit_chat_compatibility(model, has_tools):
    provider = OpenAIProvider(EngineConfig(openai_model=model, openai_api_key='test',
        openai_wire_api='chat_completions', reasoning_effort='high', max_tokens=200_000))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok', tool_calls=None),
        finish_reason='stop')], usage=None)
    provider.client.chat.completions.create = AsyncMock(return_value=response)
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'parameters': {'type': 'object'}}}] if has_tools else None
    await provider._request_once([{'role': 'user', 'content': 'hello'}], tools)
    wire = provider.client.chat.completions.create.call_args.kwargs
    assert wire['reasoning_effort'] == ('none' if has_tools else 'high')
    assert wire['max_completion_tokens'] == 128_000
    assert 'temperature' not in wire


@pytest.mark.parametrize('model', ['gpt-6-astra', 'gpt-6.1-sol'])
async def test_responses_only_tools_bridge_from_chat_completions(model, monkeypatch, caplog):
    # Live API (2026-10-07): these reject Chat Completions tools at every effort.
    provider = OpenAIProvider(EngineConfig(openai_model=model, openai_api_key='test',
        openai_wire_api='chat_completions', reasoning_effort='high'))
    responses = AsyncMock(return_value=SimpleNamespace(content='ok'))
    monkeypatch.setattr(provider, '_stream_with_retry_responses', responses)
    chat = AsyncMock()
    provider.client.chat.completions.create = chat
    tools = [NativeToolSchema('read_file', 'Read', {'path': {'type': 'string'}})]
    with caplog.at_level('WARNING', logger='clawagents.providers.llm'):
        await provider.chat([LLMMessage('user', 'Read a file')], tools=tools)
        await provider.chat([LLMMessage('user', 'Read again')], tools=tools)
    assert responses.await_count == 2
    chat.assert_not_awaited()
    assert sum('routing to /v1/responses' in r.getMessage() for r in caplog.records) == 1


def test_responses_only_tools_without_tools_keep_chat_completions():
    provider = OpenAIProvider(EngineConfig(openai_model='gpt-6-astra', openai_api_key='test',
        openai_wire_api='chat_completions'))
    assert not provider._should_use_responses(has_tools=False)
    assert provider._should_use_responses(has_tools=True)


def test_responses_only_tools_azure_keeps_explicit_wire():
    provider = OpenAIProvider(EngineConfig(openai_model='gpt-6-astra', openai_api_key='test',
        openai_wire_api='chat_completions'))
    provider._api_type = 'azure'
    assert not provider._should_use_responses(has_tools=True)
    with pytest.raises(ValueError, match='GPT-6 Astra requires the Responses API for tool calling'):
        from clawagents.providers.llm import _apply_tool_reasoning_compat
        _apply_tool_reasoning_compat({}, model='gpt-6-astra', has_tools=True, preferred='high')


async def test_gpt6_astra_tool_endpoint_failure_never_falls_back(monkeypatch):
    provider = OpenAIProvider(EngineConfig(openai_model='gpt-6-astra', openai_api_key='test',
        openai_wire_api='chat_completions'))
    unavailable = RuntimeError('endpoint does not support Responses')
    monkeypatch.setattr(provider, '_stream_with_retry_responses', AsyncMock(side_effect=unavailable))
    chat = AsyncMock()
    provider.client.chat.completions.create = chat
    with pytest.raises(ValueError, match='GPT-6 Astra tool calls require the Responses API, which this endpoint'):
        await provider.chat([LLMMessage('user', 'Read')],
            tools=[NativeToolSchema('read_file', 'Read', {})])
    chat.assert_not_awaited()
    assert not provider._force_chat_completions


async def test_gpt6_astra_explicit_chat_without_tools():
    provider = OpenAIProvider(EngineConfig(openai_model='gpt-6-astra', openai_api_key='test',
        openai_wire_api='chat_completions', reasoning_effort='none', max_tokens=200_000))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok', tool_calls=None),
        finish_reason='stop')], usage=None)
    provider.client.chat.completions.create = AsyncMock(return_value=response)
    await provider._request_once([{'role': 'user', 'content': 'hello'}])
    wire = provider.client.chat.completions.create.call_args.kwargs
    assert wire['reasoning_effort'] == 'low'
    assert 'temperature' not in wire
