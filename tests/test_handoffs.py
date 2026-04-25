"""Tests for v6.4 Handoffs + Agent.as_tool.

Uses a hand-rolled mock LLM (similar to ``_NativeMockLLM`` in
``test_parallel_native_indexing.py``) so we can drive the agent loop
deterministically without real provider calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from clawagents.agent import ClawAgent
from clawagents.graph.agent_loop import run_agent_graph
from clawagents.handoffs import Handoff, HandoffInputData, handoff
from clawagents.handoff_filters import remove_all_tools
from clawagents.lifecycle import RunHooks
from clawagents.providers.llm import (
    LLMProvider,
    LLMResponse,
    LLMMessage,
    NativeToolCall,
    NativeToolSchema,
)
from clawagents.run_context import RunContext
from clawagents.stream_events import HandoffOccurredEvent, StreamEvent
from clawagents.tools.registry import ToolRegistry, ToolResult


# ─── Mock LLM ────────────────────────────────────────────────────────────


class _NativeMockLLM(LLMProvider):
    """LLM that emits a scripted sequence of native tool-call rounds.

    Records the ``tools`` list seen by every chat call so tests can
    assert what schemas the loop synthesised.
    """

    name = "mock-native"

    def __init__(self, rounds: list[LLMResponse]):
        self._rounds = rounds
        self._idx = 0
        self.received_messages: list[list[LLMMessage]] = []
        self.received_tools: list[list[NativeToolSchema] | None] = []

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: Any = None,
        cancel_event: Any = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        self.received_messages.append(list(messages))
        self.received_tools.append(list(tools) if tools else None)
        idx = min(self._idx, len(self._rounds) - 1)
        self._idx += 1
        return self._rounds[idx]


# ─── Helpers ────────────────────────────────────────────────────────────


def _build_target_agent(
    text: str = "child final answer",
) -> tuple[ClawAgent, _NativeMockLLM]:
    """Return a ClawAgent that immediately replies ``text`` and stops."""
    target_llm = _NativeMockLLM([
        LLMResponse(content=text, model="mock", tokens_used=1),
    ])
    target = ClawAgent(
        llm=target_llm,
        tools=ToolRegistry(),
        system_prompt="You are the billing specialist.",
        streaming=False,
        use_native_tools=True,
        max_iterations=2,
        name="billing_specialist",
    )
    return target, target_llm


# ─── Handoff schema surfacing ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handoff_tool_appears_in_native_schemas():
    """The synthetic ``transfer_to_<name>`` tool must reach the LLM's tools list."""
    target, _ = _build_target_agent()
    h = handoff(target)

    parent_llm = _NativeMockLLM([
        LLMResponse(content="done", model="mock", tokens_used=1),
    ])
    await run_agent_graph(
        "task",
        parent_llm,
        tools=ToolRegistry(),
        streaming=False,
        on_event=lambda k, d: None,
        use_native_tools=True,
        max_iterations=2,
        handoffs=[h],
    )

    # First chat call should expose the handoff as a NativeToolSchema.
    assert parent_llm.received_tools[0] is not None
    names = [s.name for s in parent_llm.received_tools[0] or []]
    assert h.name in names
    # Default name has the prefix.
    assert h.name.startswith("transfer_to_")
    # Description carries the agent label.
    desc_match = next(s for s in parent_llm.received_tools[0] or [] if s.name == h.name)
    assert "billing_specialist" in desc_match.description


# ─── Dispatch transfers control ────────────────────────────────────────


@pytest.mark.asyncio
async def test_handoff_call_runs_target_agent():
    target, target_llm = _build_target_agent("billing answer")
    h = handoff(target)

    parent_llm = _NativeMockLLM([
        LLMResponse(
            content="",
            model="mock",
            tokens_used=1,
            tool_calls=[NativeToolCall(h.name, {"reason": "user asked about invoice"}, tool_call_id="call_1")],
        ),
        # The parent LLM should NEVER be called again because the loop
        # transferred to the child. Add a safety response anyway.
        LLMResponse(content="parent fallback (should not appear)", model="mock", tokens_used=1),
    ])

    state = await run_agent_graph(
        "Help me with my bill",
        parent_llm,
        tools=ToolRegistry(),
        streaming=False,
        on_event=lambda k, d: None,
        use_native_tools=True,
        max_iterations=4,
        handoffs=[h],
    )

    # The child agent's terminal text wins.
    assert state.result == "billing answer"
    assert state.status == "done"
    # The child LLM was indeed called (i.e. transfer happened).
    assert len(target_llm.received_messages) == 1
    # The parent LLM was called exactly once — it issued the handoff and didn't loop further.
    assert len(parent_llm.received_messages) == 1


# ─── Input filter is invoked ────────────────────────────────────────


@pytest.mark.asyncio
async def test_input_filter_receives_payload_and_filters_messages():
    target, target_llm = _build_target_agent("filtered child")
    captured: dict[str, Any] = {}

    def custom_filter(data: HandoffInputData) -> HandoffInputData:
        captured["payload"] = data
        # Drop assistant + tool messages on the way to the child.
        kept = [m for m in data.input_history if m.role in ("system", "user")]
        return HandoffInputData(
            input_history=kept,
            pre_handoff_items=data.pre_handoff_items,
            new_items=data.new_items,
            run_context=data.run_context,
        )

    h = handoff(target, input_filter=custom_filter)

    parent_llm = _NativeMockLLM([
        LLMResponse(
            content="",
            model="mock",
            tokens_used=1,
            tool_calls=[NativeToolCall(h.name, {"reason": "transfer"}, tool_call_id="call_1")],
        ),
    ])

    await run_agent_graph(
        "Original user task",
        parent_llm,
        tools=ToolRegistry(),
        streaming=False,
        on_event=lambda k, d: None,
        use_native_tools=True,
        max_iterations=4,
        handoffs=[h],
    )

    assert "payload" in captured
    payload: HandoffInputData = captured["payload"]
    assert isinstance(payload, HandoffInputData)
    assert payload.run_context is not None
    # The trigger (handoff tool call assistant + tool ack) is in the history.
    roles = [m.role for m in payload.input_history]
    assert "assistant" in roles
    assert "tool" in roles


# ─── remove_all_tools strips tool exchanges ────────────────────────────


def test_remove_all_tools_strips_tool_messages():
    history = [
        LLMMessage(role="system", content="you are helpful"),
        LLMMessage(role="user", content="hi"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "x", "name": "ls", "args": {}}],
        ),
        LLMMessage(role="tool", content="result", tool_call_id="x"),
        LLMMessage(role="user", content="[Tool Result] foo"),
        LLMMessage(role="assistant", content="great"),
    ]
    data = HandoffInputData(input_history=history)

    out = remove_all_tools(data)
    out_roles = [(m.role, m.content if isinstance(m.content, str) else "") for m in out.input_history]
    assert ("tool", "result") not in out_roles
    assert all(not (r == "user" and c.startswith("[Tool Result]")) for r, c in out_roles)
    # System/user/assistant text stays.
    assert ("system", "you are helpful") in out_roles
    assert ("user", "hi") in out_roles
    assert ("assistant", "great") in out_roles
    # The native-tool-call assistant message (with tool_calls_meta) is dropped.
    assert all(getattr(m, "tool_calls_meta", None) is None for m in out.input_history)


# ─── RunHooks.on_handoff fires ────────────────────────────────────


@pytest.mark.asyncio
async def test_runhooks_on_handoff_fires():
    target, _ = _build_target_agent("ok")
    h = handoff(target)

    seen: list[tuple[str, str]] = []

    class Capture(RunHooks):
        async def on_handoff(self, context: RunContext, from_agent: str, to_agent: str) -> None:
            seen.append((from_agent, to_agent))

    parent_llm = _NativeMockLLM([
        LLMResponse(
            content="",
            model="mock",
            tokens_used=1,
            tool_calls=[NativeToolCall(h.name, {"reason": "transfer"}, tool_call_id="call_1")],
        ),
    ])

    await run_agent_graph(
        "task",
        parent_llm,
        tools=ToolRegistry(),
        streaming=False,
        on_event=lambda k, d: None,
        use_native_tools=True,
        max_iterations=4,
        handoffs=[h],
        hooks=Capture(),
        agent_name="parent_agent",
    )

    assert seen == [("parent_agent", "billing_specialist")]


# ─── HandoffOccurredEvent is emitted ────────────────────────────


@pytest.mark.asyncio
async def test_handoff_occurred_event_emitted():
    target, _ = _build_target_agent("ok")
    h = handoff(target)

    events: list[StreamEvent] = []
    parent_llm = _NativeMockLLM([
        LLMResponse(
            content="",
            model="mock",
            tokens_used=1,
            tool_calls=[NativeToolCall(h.name, {"reason": "test reason"}, tool_call_id="call_1")],
        ),
    ])

    await run_agent_graph(
        "task",
        parent_llm,
        tools=ToolRegistry(),
        streaming=False,
        on_event=lambda k, d: None,
        use_native_tools=True,
        max_iterations=4,
        handoffs=[h],
        on_stream_event=events.append,
        agent_name="parent_agent",
    )

    handoff_evts = [e for e in events if isinstance(e, HandoffOccurredEvent)]
    assert len(handoff_evts) == 1
    e = handoff_evts[0]
    assert e.from_agent == "parent_agent"
    assert e.to_agent == "billing_specialist"
    assert e.tool_name == h.name
    assert e.reason == "test reason"


# ─── Agent.as_tool: callable, runs wrapped agent, returns state.result ──


@pytest.mark.asyncio
async def test_agent_as_tool_runs_wrapped_agent():
    target, target_llm = _build_target_agent("wrapped child output")

    tool = target.as_tool(
        tool_name="ask_billing",
        tool_description="Ask the billing agent",
    )

    assert tool.name == "ask_billing"
    assert tool.description == "Ask the billing agent"
    assert "task" in tool.parameters
    assert tool.parameters["task"].get("required") is True

    result = await tool.execute({"task": "Where is my refund?"})
    assert result.success is True
    assert result.output == "wrapped child output"
    assert len(target_llm.received_messages) == 1


@pytest.mark.asyncio
async def test_agent_as_tool_uses_custom_extractor():
    target, _ = _build_target_agent("default would be this")

    def extract(state: Any) -> str:
        return f"custom:{state.tool_calls}:{state.result}"

    tool = target.as_tool(
        tool_name="ask_billing",
        tool_description="Ask the billing agent",
        custom_output_extractor=extract,
    )

    result = await tool.execute({"task": "anything"})
    assert result.success is True
    # ``state.result`` is the child's terminal text; ``state.tool_calls`` is 0
    # because the mock LLM stops on the first round.
    assert result.output == "custom:0:default would be this"


@pytest.mark.asyncio
async def test_agent_as_tool_missing_task_arg():
    target, _ = _build_target_agent()
    tool = target.as_tool(tool_name="ask", tool_description="Ask")
    result = await tool.execute({"task": ""})
    assert result.success is False
    assert "missing" in (result.error or "").lower()
