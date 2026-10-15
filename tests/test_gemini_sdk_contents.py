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


def test_gemini_4_argon_prompt_block_falls_back_to_38_flash_in_stream():
    """Gemini 4 Argon BlockedReason.OTHER on bio-keyword prompts falls back to gemini-3.8-flash."""
    import asyncio
    from types import SimpleNamespace
    from clawagents.config.config import EngineConfig
    from clawagents.providers.llm import GeminiProvider, LLMMessage

    cfg = EngineConfig(gemini_api_key="test-key", gemini_model="gemini-4-argon")
    prov = GeminiProvider(cfg)
    called_models: list[str] = []

    async def fake_stream(*, model: str, contents, config):
        called_models.append(model)
        if model == "gemini-4-argon":
            async def _blocked():
                yield SimpleNamespace(
                    candidates=None,
                    prompt_feedback=SimpleNamespace(
                        block_reason=SimpleNamespace(name="OTHER")
                    ),
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=0,
                        candidates_token_count=0,
                        thoughts_token_count=0,
                        cached_content_token_count=0,
                    ),
                )
            return _blocked()

        async def _ok():
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        finish_reason=SimpleNamespace(name="STOP"),
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="Checked repo.", thought=False, function_call=None)]
                        ),
                    )
                ],
                prompt_feedback=None,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=120,
                    candidates_token_count=4,
                    thoughts_token_count=10,
                    cached_content_token_count=0,
                ),
            )
        return _ok()

    prov.client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=fake_stream)
        )
    )
    chunks: list[str] = []
    resp = asyncio.run(
        prov.chat(
            [LLMMessage(role="user", content="https://github.com/anthropics/uplifting-biomolecular-modeling")],
            on_chunk=chunks.append,
        )
    )
    assert called_models == ["gemini-4-argon", "gemini-3.8-flash"]
    assert resp.content == "Checked repo."
    assert resp.model == "gemini-3.8-flash"
    assert resp.finish_reason == "STOP"
    assert chunks == ["Checked repo."]


def test_gemini_prompt_block_raises_when_not_fallback_eligible():
    """Non-Gemini-4 prompt blocks raise a clear RuntimeError instead of returning empty done."""
    import asyncio
    from types import SimpleNamespace
    from clawagents.config.config import EngineConfig
    from clawagents.providers.llm import GeminiProvider, LLMMessage

    cfg = EngineConfig(gemini_api_key="test-key", gemini_model="gemini-3.8-flash")
    prov = GeminiProvider(cfg)

    async def fake_stream(*, model: str, contents, config):
        async def _blocked():
            yield SimpleNamespace(
                candidates=None,
                prompt_feedback=SimpleNamespace(
                    block_reason=SimpleNamespace(name="SAFETY")
                ),
                usage_metadata=None,
            )
        return _blocked()

    prov.client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=fake_stream)
        )
    )
    with pytest.raises(RuntimeError, match=r"Gemini blocked the request \(reason=SAFETY\)"):
        asyncio.run(
            prov.chat(
                [LLMMessage(role="user", content="blocked prompt")],
                on_chunk=lambda _c: None,
            )
        )

