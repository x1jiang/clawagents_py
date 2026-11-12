"""``clawagents.providers.llm`` is a facade over per-provider modules.

It must keep exporting every name the single-file module had, and replacing a
name on it must reach the code that now lives elsewhere.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import TypeVar

import pytest

from clawagents.providers import llm

PARTS = ("base", "stream_retry", "tool_convert", "openai_provider", "gemini_provider",
         "anthropic_provider", "bedrock_provider", "factory")


def _part(name):
    return importlib.import_module(f"clawagents.providers.{name}")


@pytest.mark.parametrize("name", [
    # public API
    "LLMMessage", "LLMResponse", "LLMProvider", "NativeToolSchema", "NativeToolCall",
    "OpenAIProvider", "GeminiProvider", "AnthropicProvider", "BedrockProvider",
    "MantleAnthropicProvider", "BedrockConverseProvider", "create_provider",
    "prefers_responses_api", "clamp_reasoning_effort_for_model", "normalize_reasoning_effort",
    "supports_openai_fast_mode", "is_hosted_muse_model", "has_active_anthropic_tool_turn",
    "mantle_anthropic_base_url", "mantle_openai_base_url",
    # private names other modules and tests reach for
    "_get_stream_breaker", "_admit_stream_breaker", "_record_stream_breaker", "_stall_guarded_stream",
    "_StreamReplayGuard", "_repair_json", "_to_gemini_tools", "_mantle_gateway_key",
    "_gemini_38_model", "_requires_responses_for_tools", "_invoke_callback", "_is_retryable",
    # modules and flags that tests replace
    "genai", "types", "_HAS_GEMINI", "_anthropic_mod", "_HAS_ANTHROPIC", "logger", "asyncio", "json",
])
def test_names_are_still_importable_from_llm(name):
    assert hasattr(llm, name), name


def test_every_implementation_name_is_reexported():
    missing = []
    for part in PARTS:
        module = _part(part)
        for name, value in vars(module).items():
            if name.startswith("__") or getattr(value, "__module__", None) != module.__name__:
                continue  # only names the part itself defines
            if isinstance(value, TypeVar):
                continue  # each module declares its own ``T``; identity is irrelevant
            if getattr(llm, name, None) is not value:
                missing.append(f"{part}.{name}")
    assert missing == []


def test_implementation_modules_form_layers_without_cycles():
    order = {name: i for i, name in enumerate(PARTS)}
    for part in PARTS:
        text = Path(_part(part).__file__).read_text()
        for other in PARTS:
            if f"from clawagents.providers.{other} import" in text:
                assert order[other] < order[part], f"{part} imports {other}, which is a later layer"
        assert "clawagents.providers.llm import" not in text, f"{part} must not import the facade"


def test_assigning_on_the_facade_reaches_the_implementation(monkeypatch):
    sentinel = object()
    gemini = _part("gemini_provider")
    original = gemini.genai
    monkeypatch.setattr(llm, "genai", sentinel)
    assert gemini.genai is sentinel, "the Gemini code must see the replacement"
    monkeypatch.undo()
    assert gemini.genai is original and llm.genai is original


def test_patching_a_shared_helper_reaches_every_module_that_uses_it(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_get_stream_breaker", lambda *a, **k: calls.append(a) or None)
    for part in ("stream_retry", "openai_provider", "gemini_provider", "anthropic_provider"):
        module = _part(part)
        assert module._get_stream_breaker is llm._get_stream_breaker, part
    monkeypatch.undo()
    assert _part("openai_provider")._get_stream_breaker is _part("stream_retry")._get_stream_breaker


def test_mock_patch_by_dotted_path_is_restored():
    from unittest.mock import patch

    anthropic = _part("anthropic_provider")
    before = anthropic._HAS_ANTHROPIC
    with patch("clawagents.providers.llm._HAS_ANTHROPIC", not before):
        assert anthropic._HAS_ANTHROPIC is (not before)
    assert anthropic._HAS_ANTHROPIC is before and llm._HAS_ANTHROPIC is before


def test_plain_assignment_like_the_vscode_gemini_shim(monkeypatch):
    """The extension's legacy shim does ``llm_mod._to_gemini_tools = replacement``."""
    owner = _part("tool_convert")
    original = owner._to_gemini_tools
    try:
        llm._to_gemini_tools = lambda schemas: ["shim"]
        assert owner._to_gemini_tools([]) == ["shim"]
        assert _part("gemini_provider")._to_gemini_tools([]) == ["shim"]
    finally:
        llm._to_gemini_tools = original
    assert _part("gemini_provider")._to_gemini_tools is original


def test_new_attributes_do_not_leak_into_implementation_modules():
    llm._scratch_attribute_for_test = 1
    try:
        assert all(not hasattr(_part(p), "_scratch_attribute_for_test") for p in PARTS)
    finally:
        del llm._scratch_attribute_for_test


def test_loggers_keep_the_historic_name():
    for part in PARTS:
        logger = getattr(_part(part), "logger", None)
        if logger is not None:
            assert logger.name == "clawagents.providers.llm", part
