"""Glimmer's documented prompt control and single-owner transport retries."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from clawagents.config.config import EngineConfig
from clawagents.providers.llm import LLMMessage, LLMResponse, OpenAIProvider


@pytest.mark.parametrize("effort,expected", [("", "medium"), ("low", "low"), ("high", "high"), ("max", "xhigh"), ("none", "low")])
def test_glimmer_effort_reaches_actual_prompt_without_mutating_history(effort, expected):
    async def run():
        provider = OpenAIProvider(EngineConfig(openai_api_key="fake", openai_model="Muse-Glimmer-30B", reasoning_effort=effort))
        provider._chat_dispatch = AsyncMock(return_value=LLMResponse(content="ok", model=provider.model, tokens_used=1))
        messages = [LLMMessage(role="system", content="Help with coding."), LLMMessage(role="user", content="Do the task.")]
        try:
            await provider.chat(messages)
            formatted = provider._chat_dispatch.call_args.args[0]
            assert f"Reasoning strength: {expected}" in formatted[0]["content"]
            assert messages[0].content == "Help with coding."
        finally:
            await provider.client.close()
    asyncio.run(run())


def test_explicit_strength_wins_and_is_not_duplicated():
    async def run():
        provider = OpenAIProvider(EngineConfig(openai_api_key="fake", openai_model="Muse-Glimmer-30B", reasoning_effort="low"))
        provider._chat_dispatch = AsyncMock(return_value=LLMResponse(content="ok", model=provider.model, tokens_used=1))
        try:
            await provider.chat([LLMMessage(role="system", content="Reasoning strength: high\nTask instructions.")])
            assert provider._chat_dispatch.call_args.args[0][0]["content"] == "Reasoning strength: high\nTask instructions."
        finally:
            await provider.client.close()
    asyncio.run(run())


@pytest.mark.parametrize("model", ["Muse-Glimmer-30B", "gpt-5.6-luna"])
def test_sdk_retry_is_disabled_so_harness_owns_attempt_budget(model):
    async def run():
        provider = OpenAIProvider(EngineConfig(openai_api_key="fake", openai_model=model))
        try:
            assert provider.client.max_retries == 0
        finally:
            await provider.client.close()
    asyncio.run(run())


def test_other_model_prompt_is_unchanged():
    async def run():
        provider = OpenAIProvider(EngineConfig(openai_api_key="fake", openai_model="gpt-5.6-luna", reasoning_effort="high"))
        provider._chat_dispatch = AsyncMock(return_value=LLMResponse(content="ok", model=provider.model, tokens_used=1))
        try:
            await provider.chat([LLMMessage(role="system", content="Help.")])
            assert provider._chat_dispatch.call_args.args[0][0]["content"] == "Help."
        finally:
            await provider.client.close()
    asyncio.run(run())


def test_custom_served_alias_and_multimodal_system_preserve_input(monkeypatch):
    from clawagents import harness_profiles
    from clawagents.providers.glimmer import is_glimmer_model, reasoning_strength_messages

    monkeypatch.setitem(harness_profiles._MODEL_ALIASES, "custom-local-30b", "meta-glimmer")
    assert is_glimmer_model("custom-local-30b")
    messages = [{"role": "system", "content": [{"type": "text", "text": "Help."}]}]
    result = reasoning_strength_messages(messages, "low")
    assert result[0]["content"][0]["text"].startswith("Reasoning strength: low")
    assert messages[0]["content"] == [{"type": "text", "text": "Help."}]
    assert reasoning_strength_messages(result, "high") == result


def test_no_system_prompt_gets_directive_and_user_text_cannot_override_it():
    from clawagents.providers.glimmer import reasoning_strength_messages

    messages = [{"role": "user", "content": "Reasoning strength: high"}]
    result = reasoning_strength_messages(messages, "low")
    assert result[0] == {"role": "system", "content": "Reasoning strength: low"}
    assert len(messages) == 1


def test_retry_policy_counts_actual_http_attempts():
    import httpx
    from clawagents.retry import RetryPolicy

    async def run():
        attempts = []

        def transport(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(500, json={"error": {"message": "temporary"}})
            return httpx.Response(200, json={"id": "ok", "object": "chat.completion", "created": 1, "model": "Muse-Glimmer-30B", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})

        provider = OpenAIProvider(EngineConfig(openai_api_key="fake", openai_model="Muse-Glimmer-30B", openai_base_url="http://retry-test.invalid/v1", openai_wire_api="chat_completions"))
        await provider.client.close()
        from openai import AsyncOpenAI
        provider.client = AsyncOpenAI(api_key="fake", base_url="http://retry-test.invalid/v1", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
        provider.retry_policy = RetryPolicy(max_retries=1, base_delay=0, jitter=0)
        try:
            response = await provider.chat([LLMMessage(role="user", content="Hi")])
            assert response.content == "ok"
            assert len(attempts) == 2
        finally:
            await provider.client.close()
    asyncio.run(run())
