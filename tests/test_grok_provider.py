"""xAI Grok support.

Grok is served over an OpenAI-compatible wire at ``https://api.x.ai/v1``, so it
routes through :class:`OpenAIProvider` with a base_url/key override rather than
a bespoke provider class.
"""

from __future__ import annotations

import pytest

from clawagents.config.config import EngineConfig
from clawagents.graph.model_profiles import resolve_context_budget
from clawagents.providers.llm import (
    clamp_reasoning_effort_for_model,
    create_provider,
    model_supports_reasoning_effort,
)
from clawagents.providers.model_classify import (
    XAI_BASE_URL,
    classify_model,
    is_grok_model,
    parse_model_ref,
)


@pytest.mark.parametrize(
    "model", ["grok-4.5", "grok-4", "xai/grok-4.5", "grok/grok-4.5", "GROK-4.5"]
)
def test_grok_ids_are_recognized(model):
    assert is_grok_model(model)


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "claude-opus-4-8", "gemini-3.1-pro"])
def test_non_grok_ids_are_not_misrouted(model):
    assert not is_grok_model(model)


def test_litellm_prefix_is_stripped_before_the_sdk_sees_it():
    assert parse_model_ref("xai/grok-4.5").bare_id == "grok-4.5"
    assert classify_model("xai/grok-4.5") == "openai"


def test_bare_grok_model_gets_the_xai_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    provider = create_provider("grok-4.5", EngineConfig())

    assert provider.model == "grok-4.5"
    assert provider._base_url == XAI_BASE_URL
    # xAI's documented surface is chat completions, not Responses.
    assert provider._wire_api == "chat_completions"


def test_explicit_base_url_and_wire_api_win(monkeypatch):
    """Proxies / gateways must not be overridden by the xAI defaults."""
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    provider = create_provider(
        "grok-4.5",
        EngineConfig(openai_base_url="https://gw.internal/v1", openai_wire_api="responses"),
    )
    assert provider._base_url == "https://gw.internal/v1"
    assert provider._wire_api == "responses"


def test_grok_context_budget_profile():
    from clawagents.graph.model_profiles import resolve_long_context_threshold

    window, ratio = resolve_context_budget("grok-4.5", 128_000)
    assert window == 500_000
    assert ratio == pytest.approx(0.85)
    assert resolve_long_context_threshold("grok-4.5") == 200_000
    assert resolve_context_budget("grok-4.3", 128_000)[0] == 1_000_000
    assert resolve_context_budget("grok-build-0.1", 128_000)[0] == 256_000


def test_grok_advertises_reasoning_effort():
    assert model_supports_reasoning_effort("grok-4.5")


@pytest.mark.parametrize(
    "given,expected",
    [
        ("none", "low"),      # xAI rejects `none` outright — would 400
        ("minimal", "low"),   # not offered
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),     # not offered; nearest supported
    ],
)
def test_effort_is_clamped_to_levels_xai_offers(given, expected):
    assert clamp_reasoning_effort_for_model("grok-4.5", given) == expected


def test_clamping_does_not_touch_other_models():
    assert clamp_reasoning_effort_for_model("gpt-5.6-luna", "none") == "none"
    assert clamp_reasoning_effort_for_model("gpt-5.6-luna", "max") == "max"
    assert clamp_reasoning_effort_for_model("grok-4.5", None) is None


def test_xai_key_is_read_from_the_environment(monkeypatch):
    """Covers the workspace `.env` route: dotenv populates os.environ, and the
    provider must pick XAI_API_KEY up without any explicit config."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-from-env")
    provider = create_provider("grok-4.5", EngineConfig())
    assert getattr(provider.client, "api_key", "") == "xai-from-env"


def test_grok_api_key_alias_is_accepted(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "grok-alias-key")
    provider = create_provider("grok-4.5", EngineConfig())
    assert getattr(provider.client, "api_key", "") == "grok-alias-key"


def test_explicit_config_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "env-key")
    provider = create_provider("grok-4.5", EngineConfig(openai_api_key="explicit-key"))
    assert getattr(provider.client, "api_key", "") == "explicit-key"
