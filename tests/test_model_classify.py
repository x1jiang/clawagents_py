"""Canonical model → provider classification."""

from __future__ import annotations

import os
from unittest.mock import patch

from clawagents.config.config import EngineConfig, get_default_model, is_bedrock_model_id
from clawagents.providers.model_classify import (
    BEDROCK_GEO_PREFIXES,
    api_key_field_for,
    classify_model,
    parse_model_ref,
    strip_bedrock_geo_prefix,
)


def test_apac_geo_prefix_not_ap():
    assert "apac." in BEDROCK_GEO_PREFIXES
    assert "ap." not in BEDROCK_GEO_PREFIXES
    assert is_bedrock_model_id("apac.anthropic.claude-sonnet-4-5-20250929-v1:0")
    stripped = strip_bedrock_geo_prefix(
        "apac.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert stripped.startswith("anthropic.")


def test_parse_litellm_prefixes():
    assert parse_model_ref("anthropic/claude-sonnet-4-5").bare_id == "claude-sonnet-4-5"
    assert parse_model_ref("openai/gpt-4o").bare_id == "gpt-4o"
    assert parse_model_ref("bedrock/us.anthropic.claude-x").prefix_hint == "bedrock"


def test_classify_respects_provider_hint():
    assert (
        classify_model("internal-claude-large", provider_hint="anthropic") == "anthropic"
    )
    assert api_key_field_for("anthropic") == "anthropic_api_key"


def test_provider_env_hint_not_key_gated():
    """PROVIDER=anthropic must not fall through to a stale OpenAI key."""
    config = EngineConfig(
        openai_api_key="sk-openai",
        openai_model="gpt-4o",
        anthropic_api_key="",
        anthropic_model="claude-sonnet-4-5",
    )
    with patch.dict(os.environ, {"PROVIDER": "anthropic"}, clear=False):
        assert get_default_model(config) == "claude-sonnet-4-5"


def test_create_provider_strips_anthropic_prefix():
    from clawagents.config.config import EngineConfig
    from clawagents.providers.llm import AnthropicProvider, create_provider

    cfg = EngineConfig(anthropic_api_key="ak-test", openai_api_key="")
    provider = create_provider("anthropic/claude-sonnet-4-5", cfg)
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-4-5"
    assert "anthropic/" not in provider.model


def test_resolve_model_routes_key_via_profile_hint(monkeypatch):
    from unittest.mock import patch

    from clawagents.agent import _resolve_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    captured: dict = {}

    def _capture(model_name, config, **kwargs):
        captured["model"] = model_name
        captured["anthropic_api_key"] = config.anthropic_api_key
        captured["openai_api_key"] = config.openai_api_key
        captured["hint"] = kwargs.get("provider_hint")
        return object()

    with patch("clawagents.providers.llm.create_provider", side_effect=_capture):
        _resolve_model(
            "internal-claude-large",
            streaming=False,
            api_key="sk-ant-secret",
            provider="anthropic",
        )
    assert captured["anthropic_api_key"] == "sk-ant-secret"
    assert not str(captured["openai_api_key"] or "").startswith("sk-ant")
    assert captured["hint"] == "anthropic"
