"""Gemini conversation turn hygiene (function-call ordering)."""

from __future__ import annotations

from clawagents.providers.llm import (
    _coalesce_gemini_contents,
    _ensure_gemini_function_pairs,
)


def test_coalesce_merges_parallel_tool_responses():
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
        {"role": "user", "parts": [{"text": "thanks"}]},
    ]
    out = _coalesce_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user", "model", "user"]
    assert len(out[2]["parts"]) == 3  # two FRs + thanks text


def test_ensure_pairs_inserts_synthetic_fr_before_plain_user():
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"function_call": {"name": "ask_user", "args": {"question": "?"}}}]},
        {"role": "user", "parts": [{"text": "hi again"}]},
    ]
    paired = _ensure_gemini_function_pairs(raw)
    out = _coalesce_gemini_contents(paired)
    assert [t["role"] for t in out] == ["user", "model", "user"]
    assert any("function_response" in p for p in out[2]["parts"])
    assert any(p.get("text") == "hi again" for p in out[2]["parts"])


def test_coalesce_drops_leading_model():
    raw = [
        {"role": "model", "parts": [{"text": "orphan"}]},
        {"role": "user", "parts": [{"text": "hi"}]},
    ]
    out = _coalesce_gemini_contents(raw)
    assert out == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_skipped_tool_plus_new_user_alternates():
    """Regression: Tool Skipped as bare user then new 'hi' must not leave user,user,user."""
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "user", "parts": [{"text": "[Tool Skipped] ask_user was not approved"}]},
        {"role": "user", "parts": [{"text": "hi"}]},
    ]
    out = _coalesce_gemini_contents(raw)
    assert [t["role"] for t in out] == ["user"]
    assert len(out[0]["parts"]) == 3
