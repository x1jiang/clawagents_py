"""Tests for clawagents.transport."""

from __future__ import annotations

import asyncio

import pytest

from clawagents.transport import (
    LegacyChatTransport,
    Transport,
    TransportRegistry,
    TransportRequest,
    TransportResponse,
)


class _Stub(Transport):
    """Minimal Transport for tests."""

    name = "stub"

    def __init__(self) -> None:
        self.call_count = 0

    async def chat(self, request: TransportRequest) -> TransportResponse:
        self.call_count += 1
        return TransportResponse(
            text=f"echo:{request.model}",
            finish_reason="stop",
        )


@pytest.fixture(autouse=True)
def _clear_registry():
    TransportRegistry.clear()
    yield
    TransportRegistry.clear()


def test_request_defaults():
    req = TransportRequest(model="x", messages=[])
    assert req.tools is None
    assert req.tool_choice is None
    assert req.stream is False
    assert req.extra == {}


def test_response_defaults():
    res = TransportResponse(text="hi")
    assert res.tool_calls == []
    assert res.usage is None
    assert res.finish_reason is None


def test_chat_round_trip():
    stub = _Stub()
    res = asyncio.run(stub.chat(TransportRequest(model="m", messages=[])))
    assert res.text == "echo:m"
    assert stub.call_count == 1


def test_default_stream_yields_one_chunk():
    stub = _Stub()

    async def collect():
        out: list[TransportResponse] = []
        async for chunk in stub.stream(TransportRequest(model="m", messages=[])):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())
    assert len(chunks) == 1
    assert chunks[0].text == "echo:m"


def test_registry_register_and_get():
    stub = _Stub()
    TransportRegistry.register(stub)
    assert TransportRegistry.has("stub")
    assert TransportRegistry.get("stub") is stub


def test_registry_register_with_explicit_name():
    stub = _Stub()
    TransportRegistry.register(stub, name="my-stub")
    assert TransportRegistry.has("my-stub")
    assert not TransportRegistry.has("stub")


def test_registry_register_requires_name():
    class _Anon(Transport):
        async def chat(self, request: TransportRequest) -> TransportResponse:
            return TransportResponse(text="")

    with pytest.raises(ValueError):
        TransportRegistry.register(_Anon())


def test_registry_get_missing_raises():
    with pytest.raises(KeyError, match="bogus"):
        TransportRegistry.get("bogus")


def test_registry_list_is_sorted():
    TransportRegistry.register(_Stub(), name="b")
    TransportRegistry.register(_Stub(), name="a")
    assert TransportRegistry.list() == ["a", "b"]


def test_registry_unregister_and_clear():
    TransportRegistry.register(_Stub(), name="a")
    TransportRegistry.unregister("a")
    assert not TransportRegistry.has("a")
    TransportRegistry.register(_Stub(), name="x")
    TransportRegistry.register(_Stub(), name="y")
    TransportRegistry.clear()
    assert TransportRegistry.list() == []


def test_legacy_adapter_returns_transport_response_directly():
    async def fn(req: TransportRequest) -> TransportResponse:
        return TransportResponse(text=f"hi:{req.model}")

    t = LegacyChatTransport("legacy", fn)
    res = asyncio.run(t.chat(TransportRequest(model="g", messages=[])))
    assert res.text == "hi:g"
    assert t.name == "legacy"


def test_legacy_adapter_accepts_dict_return():
    async def fn(req: TransportRequest):
        return {"text": "ok", "finish_reason": "stop"}

    t = LegacyChatTransport("legacy", fn)
    res = asyncio.run(t.chat(TransportRequest(model="g", messages=[])))
    assert res.text == "ok"
    assert res.finish_reason == "stop"


def test_legacy_adapter_rejects_unknown_return_type():
    async def fn(req: TransportRequest):
        return "just a string"

    t = LegacyChatTransport("legacy", fn)
    with pytest.raises(TypeError, match="expected TransportResponse or dict"):
        asyncio.run(t.chat(TransportRequest(model="g", messages=[])))


def test_aclose_default_is_noop():
    stub = _Stub()
    asyncio.run(stub.aclose())  # should not raise
