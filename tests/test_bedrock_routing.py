"""Unit tests for Amazon Bedrock provider routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clawagents.config.config import (
    EngineConfig,
    is_bedrock_model_id,
    strip_bedrock_prefix,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("amazon.nova-pro-v1:0", True),
        ("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("meta.llama3-70b-instruct-v1:0", True),
        ("apac.amazon.nova-pro-v1:0", True),
        ("global.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        # Geo prefix alone is not enough — must still look like a Bedrock FM id.
        ("us.custom-router", False),
        ("eu.my-internal-model", False),
        ("global.something", False),
        ("claude-sonnet-4-5", False),
        ("gpt-5-mini", False),
        ("gemini-3-flash", False),
    ],
)
def test_is_bedrock_model_id(model: str, expected: bool):
    assert is_bedrock_model_id(model) is expected


def test_strip_bedrock_prefix():
    assert (
        strip_bedrock_prefix("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert strip_bedrock_prefix("amazon.nova-pro-v1:0") == "amazon.nova-pro-v1:0"


def test_create_provider_routes_claude_bedrock_to_bedrock_provider():
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(aws_region="us-east-1")
    fake = MagicMock()
    fake.name = "bedrock"
    with patch.object(llm_mod, "BedrockProvider", return_value=fake) as ctor:
        provider = llm_mod.create_provider(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            cfg,
        )
    assert provider is fake
    ctor.assert_called_once()
    passed = ctor.call_args[0][0]
    assert passed.bedrock_model == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert passed.anthropic_model == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def test_create_provider_routes_nova_to_converse():
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(aws_region="us-east-1")
    fake = MagicMock()
    fake.name = "bedrock-converse"
    with patch.object(llm_mod, "BedrockConverseProvider", return_value=fake) as ctor:
        provider = llm_mod.create_provider("amazon.nova-pro-v1:0", cfg)
    assert provider is fake
    ctor.assert_called_once()
    assert ctor.call_args[0][0].bedrock_model == "amazon.nova-pro-v1:0"


def test_create_provider_gateway_keeps_openai_for_bedrock_ids():
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(
        openai_base_url="http://localhost:8000/api/v1",
        openai_api_key="bedrock",
    )
    fake = MagicMock()
    fake.name = "openai"
    with patch.object(llm_mod, "OpenAIProvider", return_value=fake) as ctor:
        with patch.object(llm_mod, "BedrockProvider") as bedrock_ctor:
            provider = llm_mod.create_provider(
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                cfg,
            )
    assert provider is fake
    ctor.assert_called_once()
    bedrock_ctor.assert_not_called()


def test_create_provider_plain_claude_stays_anthropic():
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(anthropic_api_key="sk-ant-test")
    fake = MagicMock()
    fake.name = "anthropic"
    with patch.object(llm_mod, "AnthropicProvider", return_value=fake) as ctor:
        with patch.object(llm_mod, "BedrockProvider") as bedrock_ctor:
            provider = llm_mod.create_provider("claude-sonnet-4-5", cfg)
    assert provider is fake
    ctor.assert_called_once()
    bedrock_ctor.assert_not_called()


def test_create_provider_mantle_claude_uses_anthropic_messages():
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(
        openai_base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
        openai_api_key="mantle-key",
        openai_wire_api="chat_completions",
    )
    fake = MagicMock()
    fake.name = "anthropic"
    with patch.object(llm_mod, "AnthropicProvider", return_value=fake) as ctor:
        with patch.object(llm_mod, "OpenAIProvider") as openai_ctor:
            provider = llm_mod.create_provider("anthropic.claude-haiku-4-5", cfg)
    assert provider is fake
    openai_ctor.assert_not_called()
    passed = ctor.call_args[0][0]
    assert passed.anthropic_api_key == "mantle-key"
    assert passed.anthropic_model == "anthropic.claude-haiku-4-5"
    assert (
        passed.anthropic_base_url
        == "https://bedrock-mantle.us-east-1.api.aws/anthropic"
    )


def test_create_provider_mantle_gpt56_uses_openai_responses():
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(
        openai_base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
        openai_api_key="mantle-key",
        openai_wire_api="chat_completions",
    )
    fake = MagicMock()
    fake.name = "openai"
    with patch.object(llm_mod, "OpenAIProvider", return_value=fake) as ctor:
        provider = llm_mod.create_provider("openai.gpt-5.6-sol", cfg)
    assert provider is fake
    passed = ctor.call_args[0][0]
    assert passed.openai_base_url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    assert passed.openai_wire_api == "responses"
    assert passed.openai_model == "openai.gpt-5.6-sol"


def test_create_provider_mantle_gpt_oss_stays_chat():
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(
        openai_base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
        openai_api_key="mantle-key",
        openai_wire_api="auto",
    )
    fake = MagicMock()
    fake.name = "openai"
    with patch.object(llm_mod, "OpenAIProvider", return_value=fake) as ctor:
        provider = llm_mod.create_provider("openai.gpt-oss-20b", cfg)
    assert provider is fake
    passed = ctor.call_args[0][0]
    assert passed.openai_base_url == "https://bedrock-mantle.us-east-1.api.aws/v1"
    assert passed.openai_wire_api == "chat_completions"


def test_create_provider_mantle_xai_grok_uses_openai_frontier_path():
    """xai.grok-4.3 on plain …/v1 returns Berm access_denied; must use …/openai."""
    from clawagents.providers import llm as llm_mod

    cfg = EngineConfig(
        openai_base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
        openai_api_key="mantle-key",
        openai_wire_api="chat_completions",
    )
    fake = MagicMock()
    fake.name = "openai"
    with patch.object(llm_mod, "OpenAIProvider", return_value=fake) as ctor:
        provider = llm_mod.create_provider("xai.grok-4.3", cfg)
    assert provider is fake
    passed = ctor.call_args[0][0]
    assert passed.openai_base_url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
    assert passed.openai_wire_api == "responses"
    assert passed.openai_model == "xai.grok-4.3"


def test_bedrock_profile_is_native():
    from clawagents.provider_profiles import resolve_provider_profile

    resolved = resolve_provider_profile("bedrock")
    assert resolved.provider == "bedrock"
    assert resolved.model and resolved.model.startswith("us.anthropic.")
    assert resolved.api_key is None
    assert not resolved.base_url


def test_bedrock_gateway_profile():
    from clawagents.provider_profiles import resolve_provider_profile

    resolved = resolve_provider_profile(
        "bedrock-gateway",
        base_url="http://localhost:8000/api/v1",
    )
    assert resolved.provider == "openai"
    assert resolved.api_key == "bedrock"
    assert resolved.base_url == "http://localhost:8000/api/v1"


def test_resolve_model_skips_anthropic_key_for_native_bedrock(monkeypatch):
    from clawagents.agent import _resolve_model

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = MagicMock()
    fake.name = "bedrock"
    captured: dict = {}

    def _capture(model_name, config, **_kwargs):
        captured["anthropic_api_key"] = config.anthropic_api_key
        captured["model"] = model_name
        return fake

    with patch("clawagents.providers.llm.create_provider", side_effect=_capture):
        provider = _resolve_model(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            True,
            api_key="should-be-ignored",
        )
    assert provider is fake
    assert captured["anthropic_api_key"] == ""
    assert "us.anthropic." in captured["model"]
