"""Gemini conversation turn hygiene (function-call ordering)."""

from __future__ import annotations

from clawagents.providers.llm import (
    _coalesce_gemini_contents,
    _ensure_gemini_function_pairs,
    _sanitize_gemini_contents,
)


def test_coalesce_merges_parallel_tool_responses_only():
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {
            "role": "model",
            "parts": [
                {"function_call": {"name": "a", "args": {}}},
                {"function_call": {"name": "b", "args": {}}},
            ],
        },
        {"role": "user", "parts": [{"function_response": {"name": "a", "response": {"result": "1"}}}]},
        {"role": "user", "parts": [{"function_response": {"name": "b", "response": {"result": "2"}}}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user", "model", "user"]
    assert len(out[2]["parts"]) == 2
    assert all("function_response" in p for p in out[2]["parts"])


def test_fr_not_mixed_with_following_user_text():
    """Gemini rejects function_response+plain text in the same user turn."""
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"function_call": {"name": "ask_user", "args": {"question": "?"}}}]},
        {"role": "user", "parts": [{"text": "hi again"}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user", "model", "user", "model", "user"]
    assert all("function_response" in p for p in out[2]["parts"])
    assert out[4]["parts"] == [{"text": "hi again"}]


def test_coalesce_drops_leading_model():
    raw = [
        {"role": "model", "parts": [{"text": "orphan"}]},
        {"role": "user", "parts": [{"text": "hi"}]},
    ]
    out = _coalesce_gemini_contents(raw)
    assert out == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_orphan_fr_after_text_model_dropped():
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "thinking only"}]},
        {"role": "user", "parts": [{"function_response": {"name": "x", "response": {"result": "1"}}}]},
    ]
    out = _sanitize_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user", "model"]
    assert out[1]["parts"] == [{"text": "thinking only"}]


def test_ensure_pairs_inserts_synthetic_fr():
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"function_call": {"name": "ask_user", "args": {}}}]},
    ]
    paired = _ensure_gemini_function_pairs(raw)
    assert paired[-1]["role"] == "user"
    assert "function_response" in paired[-1]["parts"][0]
