"""Hermetic tests for the ACP adapter.

These tests must run without the optional ``acp`` package — they cover
the message dataclasses, the agent → ACP translation pipeline, and the
in-memory prompt runner.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from clawagents.acp import (
    AcpServer,
    AgentMessageChunk,
    AgentSession,
    AgentThoughtChunk,
    PermissionDecision,
    PermissionRequest,
    PromptRequest,
    StopReason,
    ToolCallComplete,
    ToolCallStart,
    decode_update,
    encode_update,
)


# ──────────────────────────────────────────────────────────────────────
# Message round-trips
# ──────────────────────────────────────────────────────────────────────


def test_prompt_request_from_text_blocks() -> None:
    payload = {
        "sessionId": "s1",
        "prompt": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
            {"type": "image", "data": "..."},
        ],
    }
    req = PromptRequest.from_dict(payload)
    assert req.session_id == "s1"
    assert req.text == "hello\nworld"
    assert len(req.blocks) == 3


def test_prompt_request_round_trip() -> None:
    req = PromptRequest(session_id="s2", text="hi", blocks=[{"type": "text", "text": "hi"}])
    assert PromptRequest.from_dict(req.to_dict()).text == "hi"


def test_message_chunk_round_trip() -> None:
    chunk = AgentMessageChunk(text="streaming")
    encoded = encode_update(chunk)
    assert encoded["sessionUpdate"] == "agent_message_chunk"
    assert encoded["content"] == {"type": "text", "text": "streaming"}
    decoded = decode_update(encoded)
    assert isinstance(decoded, AgentMessageChunk)
    assert decoded.text == "streaming"


def test_thought_chunk_round_trip() -> None:
    thought = AgentThoughtChunk(text="thinking")
    encoded = encode_update(thought)
    decoded = decode_update(encoded)
    assert isinstance(decoded, AgentThoughtChunk)
    assert decoded.text == "thinking"


def test_tool_call_round_trip_with_args() -> None:
    start = ToolCallStart.new(
        "write_file", arguments={"path": "/tmp/a.txt"}, label="Write a.txt"
    )
    encoded = encode_update(start)
    assert encoded["sessionUpdate"] == "tool_call"
    assert encoded["status"] == "in_progress"
    assert encoded["rawInput"] == {"path": "/tmp/a.txt"}
    decoded = decode_update(encoded)
    assert isinstance(decoded, ToolCallStart)
    assert decoded.name == "write_file"
    assert decoded.arguments == {"path": "/tmp/a.txt"}


def test_tool_call_complete_text_output() -> None:
    end = ToolCallComplete(
        tool_call_id="tc_x",
        name="read_file",
        output="contents",
    )
    encoded = encode_update(end)
    assert encoded["status"] == "completed"
    assert encoded["content"][0] == {"type": "text", "text": "contents"}


def test_tool_call_complete_json_output() -> None:
    end = ToolCallComplete(
        tool_call_id="tc_x", name="search", output={"hits": [1, 2, 3]}
    )
    encoded = encode_update(end)
    text = encoded["content"][0]["text"]
    assert json.loads(text) == {"hits": [1, 2, 3]}


def test_tool_call_complete_error() -> None:
    end = ToolCallComplete(tool_call_id="tc_x", name="exec", error="boom")
    encoded = encode_update(end)
    assert encoded["status"] == "failed"
    decoded = decode_update(encoded)
    assert isinstance(decoded, ToolCallComplete)
    assert decoded.error == "boom"


def test_decode_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        decode_update({"sessionUpdate": "bogus"})


def test_permission_decision_from_dict_allow() -> None:
    decision = PermissionDecision.from_dict(
        {"outcome": {"kind": "allow"}, "remember": True}
    )
    assert decision.allowed is True
    assert decision.one_time is False


def test_permission_decision_from_dict_deny() -> None:
    decision = PermissionDecision.from_dict({"outcome": {"kind": "denied"}})
    assert decision.allowed is False


# ──────────────────────────────────────────────────────────────────────
# AgentSession translation
# ──────────────────────────────────────────────────────────────────────


def test_session_dispatches_message_chunks() -> None:
    sink: List[Dict[str, Any]] = []
    sess = AgentSession(session_id="s1", sink=sink.append)
    sess.dispatch("llm.delta", {"text": "hello "})
    sess.dispatch("message_text", {"text": "world"})
    assert [u["sessionUpdate"] for u in sink] == [
        "agent_message_chunk",
        "agent_message_chunk",
    ]
    assert sink[0]["content"]["text"] == "hello "


def test_session_dispatches_thought_chunks() -> None:
    sink: List[Dict[str, Any]] = []
    sess = AgentSession(session_id="s1", sink=sink.append)
    sess.dispatch("reasoning", {"text": "hmm"})
    assert sink[0]["sessionUpdate"] == "agent_thought_chunk"


def test_session_pairs_tool_start_with_completion() -> None:
    sink: List[Dict[str, Any]] = []
    sess = AgentSession(session_id="s1", sink=sink.append)
    sess.dispatch("tool.started", {"name": "read_file", "arguments": {"path": "/x"}})
    sess.dispatch("tool.completed", {"name": "read_file", "output": "ok"})
    assert sink[0]["sessionUpdate"] == "tool_call"
    assert sink[1]["sessionUpdate"] == "tool_call_update"
    # Completion should reference the same tool_call_id as the start.
    assert sink[0]["toolCallId"] == sink[1]["toolCallId"]
    # And carry the original arguments through.
    assert sink[1]["rawInput"] == {"path": "/x"}


def test_session_pairs_concurrent_tool_calls_in_order() -> None:
    sink: List[Dict[str, Any]] = []
    sess = AgentSession(session_id="s1", sink=sink.append)
    sess.dispatch("tool.started", {"name": "fetch", "arguments": {"url": "a"}})
    sess.dispatch("tool.started", {"name": "fetch", "arguments": {"url": "b"}})
    sess.dispatch("tool.completed", {"name": "fetch", "output": "first"})
    sess.dispatch("tool.completed", {"name": "fetch", "output": "second"})
    starts = [u for u in sink if u["sessionUpdate"] == "tool_call"]
    ends = [u for u in sink if u["sessionUpdate"] == "tool_call_update"]
    assert starts[0]["toolCallId"] == ends[0]["toolCallId"]
    assert starts[1]["toolCallId"] == ends[1]["toolCallId"]
    assert ends[0]["content"][0]["text"] == "first"
    assert ends[1]["content"][0]["text"] == "second"


def test_session_records_stop_reason() -> None:
    sess = AgentSession(session_id="s1")
    sess.dispatch("run_finished", {"reason": "max_tokens"})
    assert sess.stop_reason == StopReason.MAX_TOKENS


def test_session_records_error_stop() -> None:
    sess = AgentSession(session_id="s1")
    sess.dispatch("run_error", {"error": "oops"})
    assert sess.stop_reason == StopReason.ERROR


def test_session_dispatch_rejects_async_sink() -> None:
    coros: List[Any] = []

    async def async_sink(_: Dict[str, Any]) -> None:
        return None

    def capturing_sink(payload: Dict[str, Any]) -> Any:
        c = async_sink(payload)
        coros.append(c)
        return c

    sess = AgentSession(session_id="s1", sink=capturing_sink)
    with pytest.raises(TypeError):
        sess.dispatch("message_text", {"text": "x"})
    for c in coros:
        c.close()


def test_session_adispatch_supports_async_sink() -> None:
    async def run() -> List[Dict[str, Any]]:
        sink: List[Dict[str, Any]] = []

        async def async_sink(payload: Dict[str, Any]) -> None:
            sink.append(payload)

        sess = AgentSession(session_id="s1", sink=async_sink)
        await sess.adispatch("message_text", {"text": "ok"})
        return sink

    sink = asyncio.run(run())
    assert len(sink) == 1
    assert sink[0]["content"]["text"] == "ok"


# ──────────────────────────────────────────────────────────────────────
# Permission gate
# ──────────────────────────────────────────────────────────────────────


def test_permission_default_allows() -> None:
    sess = AgentSession(session_id="s1")
    decision = asyncio.run(sess.request_permission("write_file"))
    assert decision.allowed is True


def test_permission_calls_requester() -> None:
    seen: List[PermissionRequest] = []

    async def requester(req: PermissionRequest) -> PermissionDecision:
        seen.append(req)
        return PermissionDecision(allowed=False, rationale="nope")

    sess = AgentSession(session_id="s1", permission_requester=requester)
    decision = asyncio.run(
        sess.request_permission("write_file", arguments={"path": "/etc/hosts"})
    )
    assert decision.allowed is False
    assert decision.rationale == "nope"
    assert seen and seen[0].name == "write_file"
    assert seen[0].arguments == {"path": "/etc/hosts"}


# ──────────────────────────────────────────────────────────────────────
# AcpServer.run_prompt() — integration without the optional package
# ──────────────────────────────────────────────────────────────────────


class _FakeAgent:
    """Minimal ClawAgent stand-in that emits a few events then returns."""

    def __init__(self) -> None:
        self.on_event = None
        self.received: List[str] = []

    async def arun(self, prompt: str) -> str:
        self.received.append(prompt)
        if self.on_event is not None:
            self.on_event("llm.delta", {"text": "hi"})
            self.on_event("tool.started", {"name": "read_file", "arguments": {"p": "/x"}})
            self.on_event("tool.completed", {"name": "read_file", "output": "ok"})
            self.on_event("run_finished", {"reason": "end_turn"})
        return "done"


def test_server_run_prompt_relays_events() -> None:
    agent = _FakeAgent()
    server = AcpServer(agent=agent)
    sink: List[Dict[str, Any]] = []

    async def async_sink(raw: Dict[str, Any]) -> None:
        sink.append(raw)

    async def run() -> StopReason:
        return await server.run_prompt(
            PromptRequest(session_id="s1", text="hello"), async_sink
        )

    stop = asyncio.run(run())
    assert stop == StopReason.END_TURN
    assert agent.received == ["hello"]
    kinds = [u["sessionUpdate"] for u in sink]
    assert kinds == [
        "agent_message_chunk",
        "tool_call",
        "tool_call_update",
    ]


def test_server_run_prompt_falls_back_to_final_message() -> None:
    """Agents that don't stream still produce one user-visible chunk."""

    class SilentAgent:
        on_event = None

        async def arun(self, prompt: str) -> str:
            return f"echo: {prompt}"

    server = AcpServer(agent=SilentAgent())
    sink: List[Dict[str, Any]] = []

    async def async_sink(raw: Dict[str, Any]) -> None:
        sink.append(raw)

    async def run() -> StopReason:
        return await server.run_prompt(
            PromptRequest(session_id="s2", text="say hi"), async_sink
        )

    stop = asyncio.run(run())
    assert stop == StopReason.END_TURN
    assert len(sink) == 1
    assert sink[0]["content"]["text"] == "echo: say hi"


def test_server_run_prompt_supports_invoke_only_agents() -> None:
    class InvokeOnlyAgent:
        async def invoke(self, prompt: str) -> str:
            return f"invoke: {prompt}"

    server = AcpServer(agent=InvokeOnlyAgent())
    sink: List[Dict[str, Any]] = []

    async def async_sink(raw: Dict[str, Any]) -> None:
        sink.append(raw)

    async def run() -> StopReason:
        return await server.run_prompt(
            PromptRequest(session_id="s-invoke", text="hello"), async_sink
        )

    stop = asyncio.run(run())
    assert stop == StopReason.END_TURN
    assert len(sink) == 1
    assert sink[0]["content"]["text"] == "invoke: hello"


def test_server_run_prompt_reports_runner_error() -> None:
    class BoomAgent:
        on_event = None

        async def arun(self, prompt: str) -> str:
            raise RuntimeError("kaboom")

    server = AcpServer(agent=BoomAgent())
    sink: List[Dict[str, Any]] = []

    async def async_sink(raw: Dict[str, Any]) -> None:
        sink.append(raw)

    async def run() -> StopReason:
        return await server.run_prompt(
            PromptRequest(session_id="s3", text="boom"), async_sink
        )

    stop = asyncio.run(run())
    assert stop == StopReason.ERROR


def test_server_serve_raises_without_acp_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """If `acp` isn't installed, .serve() raises a friendly error."""

    import importlib

    real_import = importlib.import_module

    def stub_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "acp":
            raise ImportError("no module named 'acp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", stub_import)

    from clawagents.acp.errors import MissingAcpDependencyError

    server = AcpServer(agent=_FakeAgent())
    with pytest.raises(MissingAcpDependencyError):
        server.serve()
