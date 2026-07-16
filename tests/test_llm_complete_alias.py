"""Regression: LLMProvider.complete aliases chat; chat has no stream= kwarg."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse


class _Stub(LLMProvider):
    name = "stub"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
        self.calls.append(
            {
                "n": len(messages),
                "on_chunk": on_chunk,
                "tools": tools,
            }
        )
        return LLMResponse(content="ok", model="stub", tokens_used=1)


def test_complete_aliases_chat_and_ignores_stream():
    llm = _Stub()

    async def _run():
        return await llm.complete(
            [LLMMessage(role="user", content="hi")],
            stream=False,
        )

    resp = asyncio.run(_run())
    assert resp.content == "ok"
    assert len(llm.calls) == 1


def test_chat_rejects_stream_kwarg():
    llm = _Stub()

    async def _run():
        await llm.chat([LLMMessage(role="user", content="hi")], stream=False)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        asyncio.run(_run())


def test_chat_signature_has_no_stream():
    sig = inspect.signature(LLMProvider.chat)
    assert "stream" not in sig.parameters
