"""Model-profile resolution: context windows, prefix stripping, fallbacks.

Context windows are pinned to the vendor docs (Anthropic context-windows
page, Google Gemini model cards, AWS Bedrock model cards). If a vendor
changes a window, update ``MODEL_PROFILES`` *and* the expectation here —
a silent mismatch either over-compacts (wasting context) or overflows
(``prompt is too long`` 400s).
"""

from __future__ import annotations

import pytest

from clawagents.graph.model_profiles import (
    MODEL_PROFILES,
    normalize_model_id,
    resolve_context_budget,
    resolve_long_context_threshold,
    resolve_model_profile,
)


# ─── Anthropic ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        # Dated / Bedrock / Mantle spellings must land on the same profile.
        "claude-opus-4-8-20260301",
        "anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-7-20250514-v1:0",
        "bedrock/global.anthropic.claude-sonnet-5-v1:0",
        "anthropic.claude-fable-5",
    ],
)
def test_claude_1m_models(model: str) -> None:
    window, _ = resolve_context_budget(model, 128_000)
    assert window == 1_000_000, model


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-haiku-4-5",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
    ],
)
def test_claude_200k_models(model: str) -> None:
    window, _ = resolve_context_budget(model, 1_000_000)
    assert window == 200_000, model


def test_claude_dotted_minor_beats_family_fallback() -> None:
    # "claude-opus-4" is a legitimate 200K family key; the 1M point releases
    # must be listed before it so prefix matching does not collapse them.
    keys = list(MODEL_PROFILES)
    for specific in ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6"):
        assert keys.index(specific) < keys.index("claude-opus-4")
    assert keys.index("claude-sonnet-4-6") < keys.index("claude-sonnet-4")
    assert keys.index("claude-sonnet-4-5") < keys.index("claude-sonnet-4")


# ─── Google ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.8-flash",
        "gemini-3.8-flash-preview",
        "gemini-3.7-flash",
        "gemini-2.5-flash",
        # Pro models are 1,048,576 input tokens per ai.google.dev — not 2M.
        "gemini-3.1-pro-preview",
        "gemini-3-pro",
        "gemini-2.5-pro",
    ],
)
def test_gemini_1m(model: str) -> None:
    window, ratio = resolve_context_budget(model, 128_000)
    assert window == 1_000_000
    assert ratio == pytest.approx(0.90)


# ─── Bedrock Mantle third-party ids ─────────────────────────────────────────


@pytest.mark.parametrize(
    "model, expected",
    [
        ("deepseek.v3.2", 164_000),
        ("deepseek.v3.1", 128_000),
        ("moonshotai.kimi-k2.5", 256_000),
        ("moonshot.kimi-k2-thinking", 256_000),
        ("zai.glm-5", 200_000),
        ("zai.glm-4.7", 200_000),
        ("openai.gpt-oss-120b", 128_000),
        ("openai.gpt-oss-20b", 128_000),
        ("openai.gpt-oss-safeguard-120b", 128_000),
        ("openai.gpt-oss-120b-1:0", 128_000),
    ],
)
def test_mantle_ids(model: str, expected: int) -> None:
    window, _ = resolve_context_budget(model, 1_000_000)
    assert window == expected, model


def test_gpt_oss_does_not_hit_gpt5_family() -> None:
    # "gpt-oss-120b" shares the "gpt-" stem with the GPT-5 keys; make sure it
    # does not prefix-match a 400K/1M frontier profile.
    assert resolve_model_profile("gpt-oss-120b") is MODEL_PROFILES["gpt-oss"]


# ─── normalize_model_id ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("GPT-5.6-Luna", "gpt-5.6-luna"),
        ("  openai.gpt-5.6-sol ", "gpt-5.6-sol"),
        ("bedrock/us.anthropic.claude-opus-4-8-20260301-v1:0", "claude-opus-4-8-20260301-v1"),
        ("eu.anthropic.claude-sonnet-5-v1:0", "claude-sonnet-5-v1"),
        ("xai.grok-4.3", "grok-4.3"),
        ("zai.glm-5", "glm-5"),
        ("moonshotai.kimi-k2.5", "kimi-k2.5"),
        # DeepSeek keeps its vendor dot — that *is* the model key on Mantle.
        ("deepseek.v3.2", "deepseek.v3.2"),
        ("us.deepseek.v3.2:0", "deepseek.v3.2"),
        ("", ""),
    ],
)
def test_normalize_model_id(raw: str, expected: str) -> None:
    assert normalize_model_id(raw) == expected


def test_geo_and_vendor_prefixes_both_stripped() -> None:
    # Regression: the old resolver stripped at most one prefix, so the most
    # common Bedrock spelling (geo + vendor) never matched a profile.
    assert resolve_model_profile("us.anthropic.claude-opus-4-8-20260301-v1:0") is not None
    assert resolve_model_profile("global.openai.gpt-5.6-luna") is MODEL_PROFILES["gpt-5.6-luna"]
    assert resolve_model_profile("apac.xai.grok-4.5") is MODEL_PROFILES["grok-4.5"]


# ─── Fallbacks ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", [None, "", "   ", "totally-unknown-model"])
def test_unknown_model_falls_back_to_caller_window(model: str | None) -> None:
    assert resolve_model_profile(model) is None
    window, ratio = resolve_context_budget(model or "", 123_456)
    assert window == 123_456
    assert ratio == pytest.approx(0.75)
    assert resolve_long_context_threshold(model) is None


def test_long_context_threshold_only_where_defined() -> None:
    assert resolve_long_context_threshold("gpt-5.6-luna") == 272_000
    assert resolve_long_context_threshold("xai.grok-4.5") == 200_000
    assert resolve_long_context_threshold("claude-opus-4-8") is None
    assert resolve_long_context_threshold("gemini-3.8-flash") is None


def test_every_profile_is_well_formed() -> None:
    for key, profile in MODEL_PROFILES.items():
        assert key == key.strip().lower(), key
        assert int(profile["max_input_tokens"]) > 0, key
        assert 0 < float(profile["budget_ratio"]) <= 1, key
        threshold = profile.get("long_context_threshold")
        if threshold is not None:
            assert 0 < int(threshold) < int(profile["max_input_tokens"]), key
