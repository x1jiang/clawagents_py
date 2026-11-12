"""Models added 2026-11: Gemini 4 Carbon, GPT-6 Cyber, hosted Muse (Meta Model API).

No network: profiles, request-contract helpers, and request construction
against a mocked OpenAI-compatible client.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clawagents.agent import create_claw_agent
from clawagents.graph.model_profiles import resolve_long_context_threshold, resolve_model_profile
from clawagents.provider_profiles import (
    META_API_BASE_URL,
    default_profile_for_model,
    resolve_provider_profile,
)
from clawagents.providers import llm


# ── context windows ──────────────────────────────────────────────────────


@pytest.mark.parametrize("model,window,output", [
    ("gemini-4-carbon", 1_048_576, 65_536),
    ("gpt-6-cyber", 1_050_000, 128_000),
    ("muse-spark-1.3", 1_048_576, None),
    ("muse-spark-1.3-contributor", 1_048_576, None),
    ("muse-spark-1.2", 1_048_576, None),
    ("muse-spark-1.2-contributor", 1_048_576, None),
    ("muse-spark-1.1", 1_048_576, None),
    ("muse-wonder-1.0", 1_048_576, None),
    ("muse-ember-1.0", 1_000_000, None),
    ("meta.muse-spark-1.3", 1_048_576, None),
])
def test_new_model_limits(model, window, output):
    profile = resolve_model_profile(model)
    assert profile is not None, model
    assert profile["max_input_tokens"] == window
    assert profile.get("max_output_tokens") == output


def test_long_context_thresholds():
    assert resolve_long_context_threshold("gpt-6-cyber") == 272_000
    assert resolve_long_context_threshold("muse-ember-1.0") == 200_000
    assert resolve_long_context_threshold("muse-spark-1.3") is None


def test_self_hosted_glimmer_profile_is_untouched():
    assert resolve_model_profile("Muse-Glimmer-30B")["max_input_tokens"] == 196_608
    assert not llm.is_hosted_muse_model("Muse-Glimmer-30B")


# ── Gemini 4 Carbon ──────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["gemini-4-carbon", "models/gemini-4-carbon", "gemini-4-argon"])
def test_gemini_4_family_uses_thinking_level_contract(model):
    assert llm._gemini_38_model(model)
    assert llm._gemini_4_fallback_model(model) == "gemini-3.8-flash"


def test_gemini_carbon_pattern_does_not_widen_to_older_models():
    assert not llm._gemini_38_model("gemini-3.7-flash")
    assert llm._gemini_4_fallback_model("gemini-3.8-flash") is None


# ── GPT-6 Cyber ──────────────────────────────────────────────────────────


def test_gpt6_cyber_request_contract_is_conservative():
    m = "gpt-6-cyber"
    assert llm._requires_responses_for_tools(m)
    assert llm._responses_only_model_label(m) == "GPT-6 Cyber"
    assert llm.model_supports_reasoning_effort(m)
    assert llm.prefers_responses_api(m, has_tools=True)
    # Effort set is not published; never send the two values Astra rejects.
    assert llm.clamp_reasoning_effort_for_model(m, "none") == "low"
    assert llm.clamp_reasoning_effort_for_model(m, "minimal") == "low"
    assert llm.clamp_reasoning_effort_for_model(m, "high") == "high"
    # Fast mode is not documented for it.
    assert not llm.supports_openai_fast_mode(m)


def test_existing_gpt6_contracts_unchanged():
    assert llm._requires_responses_for_tools("gpt-6.1-terra")
    assert not llm._requires_responses_for_tools("gpt-6-luna")
    assert llm._responses_only_model_label("gpt-6.1-sol") == "GPT-6.1 Sol"
    assert llm.clamp_reasoning_effort_for_model("gpt-6.1-terra", "none") == "none"
    assert llm.clamp_reasoning_effort_for_model("gpt-6-astra", "none") == "low"
    for m in ("gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra"):
        assert llm.model_supports_reasoning_effort(m)
        assert llm.supports_openai_fast_mode(m)


# ── hosted Muse / Meta Model API ─────────────────────────────────────────


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for var in ("MODEL_API_KEY", "META_MODEL_API_KEY", "META_MODEL_API_BASE_URL",
                "META_MODEL_API_MODEL", "PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-must-not-leak")
    return tmp_path


@pytest.mark.parametrize("model,expected", [
    ("muse-spark-1.3", "meta-api"), ("muse-spark-1.2-contributor", "meta-api"),
    ("muse-wonder-1.0", "meta-api"), ("muse-ember-1.0", "meta-api"),
    ("Muse-Glimmer-30B", None), ("gpt-6.1-terra", None), ("gemini-4-carbon", None), (None, None),
])
def test_default_profile_for_model(model, expected):
    assert default_profile_for_model(model) == expected


def test_meta_api_profile_defaults(clean_env, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "meta-key")
    p = resolve_provider_profile("meta-api")
    assert (p.provider, p.model, p.base_url, p.api_key) == (
        "openai", "muse-spark-1.3", META_API_BASE_URL, "meta-key")
    monkeypatch.setenv("META_MODEL_API_KEY", "preferred")
    assert resolve_provider_profile("meta-api").api_key == "preferred"
    p = resolve_provider_profile("meta-api", model="muse-wonder-1.0", api_key="explicit")
    assert (p.model, p.api_key) == ("muse-wonder-1.0", "explicit")


def test_meta_api_never_borrows_another_vendors_key(clean_env):
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        resolve_provider_profile("meta-api")
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        create_claw_agent(model="muse-spark-1.3", workspace=clean_env, skills=[], memory=[])


def test_bare_muse_model_routes_to_the_meta_endpoint(clean_env, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "meta-key")
    agent = create_claw_agent(model="muse-spark-1.3", workspace=clean_env, skills=[], memory=[])
    assert agent.llm.model == "muse-spark-1.3"
    assert str(agent.llm.client.base_url).rstrip("/") == META_API_BASE_URL
    assert agent.llm.client.api_key == "meta-key"
    # Chat Completions is the documented OpenAI-compatible path with tools.
    assert not agent.llm._should_use_responses(has_tools=True)


def test_explicit_base_url_is_respected_for_muse_ids(clean_env):
    """A gateway that serves Muse under its own URL must not be overridden."""
    agent = create_claw_agent(model="muse-spark-1.3", base_url="http://localhost:9/v1",
                              api_key="gw", workspace=clean_env, skills=[], memory=[])
    assert str(agent.llm.client.base_url).rstrip("/") == "http://localhost:9/v1"


@pytest.mark.parametrize("model,effort,expected", [
    ("muse-spark-1.3", "none", "minimal"),   # "none" is HTTP 400 on reasoning models
    ("muse-spark-1.3", "minimal", "minimal"),
    ("muse-spark-1.3", "max", "max"),
    ("muse-spark-1.3-contributor", "max", "xhigh"),  # Contributor has no "max"
    ("muse-spark-1.2-contributor", "high", "high"),
    ("muse-wonder-1.0", "xhigh", "xhigh"),
    ("muse-ember-1.0", "none", "minimal"),
])
def test_muse_reasoning_effort_clamp(model, effort, expected):
    assert llm.model_supports_reasoning_effort(model)
    assert llm.clamp_reasoning_effort_for_model(model, effort) == expected


def test_muse_models_keep_sampling_and_are_not_treated_as_gpt():
    assert not llm.openai_model_rejects_temperature("muse-spark-1.3")
    assert not llm._requires_responses_for_tools("muse-spark-1.3")
    assert not llm.supports_openai_fast_mode("muse-spark-1.3")


async def test_muse_chat_request_shape(clean_env, monkeypatch):
    """What actually goes on the wire for a Muse tool turn."""
    monkeypatch.setenv("MODEL_API_KEY", "meta-key")
    agent = create_claw_agent(model="muse-spark-1.3", workspace=clean_env, skills=[], memory=[])
    provider = agent.llm
    provider._reasoning_effort = "none"
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None, reasoning=None)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(total_tokens=5, prompt_tokens=3, completion_tokens=2,
                              prompt_tokens_details=None, completion_tokens_details=None),
        model="muse-spark-1.3",
    )
    create = AsyncMock(return_value=completion)
    monkeypatch.setattr(provider.client.chat.completions, "create", create)
    monkeypatch.setattr(llm, "_get_stream_breaker", lambda *a, **k: None, raising=False)
    tools = [llm.NativeToolSchema("check", "check", {"path": {"type": "string", "required": True}})]

    await provider.chat([llm.LLMMessage("user", "hello")], tools=tools)

    sent = create.call_args.kwargs
    assert sent["model"] == "muse-spark-1.3"
    assert "max_completion_tokens" in sent and "max_tokens" not in sent
    assert sent["tools"] and sent["tools"][0]["function"]["name"] == "check"
    # Meta rejects every tool_choice except "auto", and these parameters outright.
    assert sent.get("tool_choice", "auto") == "auto"
    for unsupported in ("stop", "frequency_penalty", "presence_penalty", "logprobs", "logit_bias", "n"):
        assert unsupported not in sent
    assert sent.get("reasoning_effort") == "minimal"
