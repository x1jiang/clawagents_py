"""Harden Gemini contents against google-genai pydantic extra_forbidden."""

from __future__ import annotations

import pytest

from clawagents.providers.llm import (
    _sanitize_gemini_contents,
    _scrub_gemini_contents,
    _scrub_gemini_function_response,
)


def _sdk_validate(contents: list[dict]) -> None:
    from google.genai import types

    for turn in contents:
        types.Content.model_validate(turn)
    types._GenerateContentParameters(model="gemini-3.7-flash", contents=contents)


def test_scrub_strips_illegal_call_id():
    dirty = {
        "name": "read_file",
        "response": {"result": "ok"},
        "id": "call_8652019",
        "call_id": "call_8652019",
    }
    clean = _scrub_gemini_function_response(dirty)
    assert clean == {
        "name": "read_file",
        "response": {"result": "ok"},
        "id": "call_8652019",
    }
    from google.genai import types

    types.FunctionResponse(**clean)
    with pytest.raises(Exception):
        types.FunctionResponse(**dirty)


def test_user_transcript_with_call_id_becomes_sdk_legal():
    """Replay the 6.20.62 crash: extra call_id after read_file."""
    raw = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "Hi. What need?"}]},
        {"role": "user", "parts": [{"text": "wha tis this project about?"}]},
        {
            "role": "model",
            "parts": [
                {
                    "thought_signature": b"sig",
                    "function_call": {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": "call_8652019",
                    },
                },
                {"text": ""},
            ],
        },
        {
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": "read_file",
                    "response": {"result": "# Project\n..."},
                    "id": "call_8652019",
                    "call_id": "call_8652019",
                }
            }],
        },
    ]
    cleaned = _sanitize_gemini_contents(raw)
    blob = str(cleaned)
    assert "call_id" not in blob
    assert cleaned[-1]["parts"][0]["function_response"]["id"] == "call_8652019"
    _sdk_validate(cleaned)


def test_scrub_alone_is_enough_for_dirty_payload():
    dirty = [
        {"role": "user", "parts": [{"text": "hi"}]},
        {
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": "read_file",
                    "response": {"result": "ok"},
                    "id": "c1",
                    "call_id": "c1",
                    "unknown_extra": True,
                }
            }],
        },
    ]
    cleaned = _scrub_gemini_contents(dirty)
    assert "call_id" not in str(cleaned)
    assert "unknown_extra" not in str(cleaned)
    from google.genai import types

    types.FunctionResponse(**cleaned[1]["parts"][0]["function_response"])
