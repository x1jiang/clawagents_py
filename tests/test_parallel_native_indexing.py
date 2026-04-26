"""Regression test for the parallel-native-tool-call indexing bug.

When the LLM emits multiple native tool calls in one round AND a `before_tool`
hook either:
  (a) rejects one of them (skipping reduces approved length below tool_calls)
  (b) returns updated_args (which constructs a NEW ParsedToolCall instance)

…the OLD code computed `_approved_call_ids` by indexing
`native_tool_call_objects[approved_idx]`, which used the wrong NativeToolCall
when (a) happened. And it built `native_tc_map` via an `is`-identity check
against `approved_calls`, which silently dropped entries when (b) happened.

The fix tracks the original `tool_calls` index alongside each approved call
through the entire parallel branch.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from clawagents.graph.agent_loop import run_agent_graph, HookResult
from clawagents.providers.llm import (
    LLMProvider,
    LLMResponse,
    LLMMessage,
    NativeToolCall,
    NativeToolSchema,
)
from clawagents.run_context import RunContext
from clawagents.tools.registry import ToolRegistry, ToolResult


class _NativeMockLLM(LLMProvider):
    """LLM that emits a scripted sequence of native tool-call rounds."""

    name = "mock-native"

    def __init__(self, rounds: list[LLMResponse]):
        self._rounds = rounds
        self._idx = 0
        self.received_messages: list[list[LLMMessage]] = []

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk=None,
        cancel_event=None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        self.received_messages.append(list(messages))
        idx = min(self._idx, len(self._rounds) - 1)
        self._idx += 1
        return self._rounds[idx]


class _RecordingTool:
    description = "test tool"
    parameters = {"x": {"type": "string"}}

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict[str, Any]] = []

    async def execute(self, args):
        self.calls.append(dict(args))
        return ToolResult(success=True, output=f"{self.name}:ok")


def _build_registry(*tool_names: str) -> tuple[ToolRegistry, list[_RecordingTool]]:
    reg = ToolRegistry()
    tools = [_RecordingTool(n) for n in tool_names]
    for t in tools:
        reg.register(t)
    return reg, tools


def _last_tool_message_for_call_id(messages: list[LLMMessage], tool_call_id: str) -> LLMMessage | None:
    for m in reversed(messages):
        if m.role == "tool" and m.tool_call_id == tool_call_id:
            return m
    return None


@pytest.mark.asyncio
async def test_native_indexing_with_before_tool_rejecting_middle_call():
    """before_tool rejects the middle call; remaining tool messages must use the
    original native ids (id_a, id_c) — NOT id_a, id_b due to off-by-one."""
    rounds = [
        LLMResponse(
            content="",
            model="mock",
            tokens_used=1,
            tool_calls=[
                NativeToolCall("alpha", {"x": "1"}, tool_call_id="id_a"),
                NativeToolCall("beta", {"x": "2"}, tool_call_id="id_b"),
                NativeToolCall("gamma", {"x": "3"}, tool_call_id="id_c"),
            ],
        ),
        LLMResponse(content="done", model="mock", tokens_used=1),
    ]
    llm = _NativeMockLLM(rounds)
    reg, _tools = _build_registry("alpha", "beta", "gamma")

    def hook(name, args):
        # Reject only the middle call
        if name == "beta":
            return HookResult(allowed=False, reason="blocked beta")
        return HookResult(allowed=True)

    state = await run_agent_graph(
        "task",
        llm,
        tools=reg,
        streaming=False,
        on_event=lambda k, d: None,
        before_tool=hook,
        use_native_tools=True,
        max_iterations=3,
    )

    # The second LLM call should have received tool messages tagged with the
    # ORIGINAL native ids for the surviving calls (id_a, id_c). The bug
    # manifested as id_a, id_b — i.e. native_tool_call_objects[approved_idx]
    # over-indexed past the rejected entry.
    second_round = llm.received_messages[1]
    assert _last_tool_message_for_call_id(second_round, "id_a") is not None, \
        f"expected tool message with id_a in second round; got {[(m.role, getattr(m, 'tool_call_id', None)) for m in second_round]}"
    assert _last_tool_message_for_call_id(second_round, "id_c") is not None, \
        "expected tool message with id_c in second round"
    # id_b was rejected — must NOT appear as a tool message
    assert _last_tool_message_for_call_id(second_round, "id_b") is None, \
        "rejected call id_b must not appear in tool messages"


@pytest.mark.asyncio
async def test_native_indexing_with_before_tool_modifying_args():
    """before_tool returns updated_args; the resulting (new) ParsedToolCall must
    still be mapped to the original NativeToolCall id, not fallback_*.

    The OLD bug used `tc is approved_calls[i]` to match — that fails after
    HookResult.updated_args constructs a new ParsedToolCall, so native_tc_map
    silently dropped entries.
    """
    rounds = [
        LLMResponse(
            content="",
            model="mock",
            tokens_used=1,
            tool_calls=[
                NativeToolCall("alpha", {"x": "raw"}, tool_call_id="id_a"),
                NativeToolCall("beta", {"x": "raw"}, tool_call_id="id_b"),
            ],
        ),
        LLMResponse(content="done", model="mock", tokens_used=1),
    ]
    llm = _NativeMockLLM(rounds)
    reg, tools = _build_registry("alpha", "beta")

    def hook(name, args):
        # Modify args for both — this constructs a new ParsedToolCall internally
        return HookResult(allowed=True, updated_args={"x": f"sanitised:{args.get('x')}"})

    state = await run_agent_graph(
        "task",
        llm,
        tools=reg,
        streaming=False,
        on_event=lambda k, d: None,
        before_tool=hook,
        use_native_tools=True,
        max_iterations=3,
    )

    # Both tool messages should use original native ids — not fallback_0 / fallback_1.
    second_round = llm.received_messages[1]
    assert _last_tool_message_for_call_id(second_round, "id_a") is not None, \
        f"expected tool message with id_a; got {[(m.role, getattr(m, 'tool_call_id', None)) for m in second_round]}"
    assert _last_tool_message_for_call_id(second_round, "id_b") is not None, \
        "expected tool message with id_b"
    # Args must reflect the hook's modification (proves the new call was used)
    assert tools[0].calls and tools[0].calls[0]["x"] == "sanitised:raw"
    assert tools[1].calls and tools[1].calls[0]["x"] == "sanitised:raw"


@pytest.mark.asyncio
async def test_parallel_native_calls_honor_run_context_rejection():
    rounds = [
        LLMResponse(
            content="",
            model="mock",
            tokens_used=1,
            tool_calls=[
                NativeToolCall("alpha", {"x": "1"}, tool_call_id="id_a"),
                NativeToolCall("beta", {"x": "2"}, tool_call_id="id_b"),
            ],
        ),
        LLMResponse(content="done", model="mock", tokens_used=1),
    ]
    llm = _NativeMockLLM(rounds)
    reg, tools = _build_registry("alpha", "beta")
    ctx = RunContext()
    ctx.reject_tool("id_b", tool_name="beta", reason="blocked beta")

    await run_agent_graph(
        "task",
        llm,
        tools=reg,
        streaming=False,
        on_event=lambda k, d: None,
        use_native_tools=True,
        max_iterations=3,
        run_context=ctx,
    )

    assert tools[0].calls == [{"x": "1"}]
    assert tools[1].calls == []
