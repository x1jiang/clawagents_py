"""ClawAgents Simulated Test Suite

Covers all major subsystems using mock LLMs — no real API keys needed.

Run:  python tests/simulated_test.py
"""

import asyncio
import math
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clawagents.tools.registry import (
    ToolRegistry, ToolResult, ParsedToolCall, truncate_tool_output,
)
from clawagents.graph.agent_loop import (
    run_agent_graph as _run_agent_graph, AgentState, EventKind, OnEvent,
    _estimate_tokens, _truncate_old_tool_args,
)
from functools import partial

async def run_agent_graph(*args, use_native_tools=False, **kwargs):
    return await _run_agent_graph(*args, use_native_tools=use_native_tools, **kwargs)
from clawagents.providers.llm import LLMProvider, LLMMessage, LLMResponse

# ━━━ Test Harness ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_passed = 0
_failed = 0


def check(condition: bool, label: str):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✓ {label}")
    else:
        _failed += 1
        print(f"  ✗ {label}")


def section(name: str):
    print(f"\n━━━ {name} ━━━")


# ━━━ Mock LLM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MockLLM(LLMProvider):
    name = "mock"

    def __init__(self, responses: list[str]):
        self.responses = responses
        self._idx = 0
        self.call_count = 0
        self.last_messages: list[LLMMessage] = []

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None) -> LLMResponse:
        self.call_count += 1
        self.last_messages = list(messages)
        content = self.responses[self._idx] if self._idx < len(self.responses) else "I'm done."
        if self._idx < len(self.responses) - 1:
            self._idx += 1

        if on_chunk:
            for ch in content:
                if asyncio.iscoroutinefunction(on_chunk):
                    await on_chunk(ch)
                else:
                    on_chunk(ch)

        return LLMResponse(content=content, model="mock", tokens_used=math.ceil(len(content) / 4))

    def reset(self):
        self._idx = 0
        self.call_count = 0


# ━━━ Mock Tools ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MathTool:
    name = "calculate"
    description = "Evaluate a math expression"
    parameters = {"expression": {"type": "string", "description": "Math expression", "required": True}}

    async def execute(self, args):
        try:
            result = eval(str(args.get("expression", "0")))
            return ToolResult(success=True, output=str(result))
        except Exception:
            return ToolResult(success=False, output="", error="Invalid expression")


class SlowTool:
    name = "slow_op"
    description = "A slow operation"
    parameters = {}

    def __init__(self, delay_s: float = 100):
        self._delay = delay_s

    async def execute(self, args):
        await asyncio.sleep(self._delay)
        return ToolResult(success=True, output="done")


class FailingTool:
    name = "unstable"
    description = "Throws errors"
    parameters = {}

    async def execute(self, args):
        raise RuntimeError("Boom!")


class CounterTool:
    description = "Increments a counter"
    parameters = {}

    def __init__(self, name="counter"):
        self.name = name
        self.count = 0

    async def execute(self, args):
        self.count += 1
        return ToolResult(success=True, output=f"count={self.count}")


def collect_events() -> tuple[list[dict], OnEvent]:
    events: list[dict] = []
    def handler(kind: EventKind, data: dict[str, Any]):
        events.append({"kind": kind, "data": data})
    return events, handler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    print("ClawAgents (Python) Simulated Test Suite\n")

    # ━━━ 1. Tool Registry ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("1. Tool Registry")

    reg = ToolRegistry(tool_timeout_s=3)
    math_tool = MathTool()
    reg.register(math_tool)

    check(reg.get("calculate") is math_tool, "register + get works")
    check(len(reg.list()) == 1, "list returns registered tools")
    check(reg.get("nonexistent") is None, "get returns None for unknown tool")

    d1 = reg.describe_for_llm()
    d2 = reg.describe_for_llm()
    check(d1 is d2, "describe_for_llm returns cached string (same ref)")
    check("calculate" in d1, "description includes tool name")

    counter = CounterTool()
    reg.register(counter)
    d3 = reg.describe_for_llm()
    check(d3 is not d1, "cache invalidated after new register")
    check("counter" in d3, "new description includes new tool")

    calc_result = await reg.execute_tool("calculate", {"expression": "2 + 3"})
    check(calc_result.success and calc_result.output == "5", "execute_tool succeeds with correct result")

    unknown_result = await reg.execute_tool("nope", {})
    check(not unknown_result.success and "Unknown" in (unknown_result.error or ""), "unknown tool returns error")

    # Timeout
    timeout_reg = ToolRegistry(tool_timeout_s=1)
    timeout_reg.register(SlowTool(100))
    t0 = time.monotonic()
    timeout_result = await timeout_reg.execute_tool("slow_op", {})
    elapsed = time.monotonic() - t0
    check(not timeout_result.success and "timed out" in (timeout_result.error or ""), "tool timeout fires")
    check(elapsed < 3, f"timeout completed in reasonable time ({elapsed:.1f}s)")

    # Error handling
    reg.register(FailingTool())
    fail_result = await reg.execute_tool("unstable", {})
    check(not fail_result.success and "Boom" in (fail_result.error or ""), "throwing tool caught gracefully")

    # Truncation
    long_output = "x" * 20000
    truncated = truncate_tool_output(long_output)
    check(len(truncated) < len(long_output), "truncate_tool_output shortens long output")
    check("truncated" in truncated, "truncation marker present")
    check(truncate_tool_output("short") == "short", "short output unchanged")

    # ━━━ 2. Tool Call Parsing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("2. Tool Call Parsing")

    parser = ToolRegistry()
    parser.register(MathTool())

    calls1 = parser.parse_tool_calls('Here:\n```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```')
    check(len(calls1) == 1 and calls1[0].tool_name == "calculate", "parses single fenced tool call")

    calls2 = parser.parse_tool_calls('```json\n[{"tool": "calculate", "args": {"expression": "1+1"}}, {"tool": "calculate", "args": {"expression": "2+2"}}]\n```')
    check(len(calls2) == 2, "parses array of tool calls")

    calls3 = parser.parse_tool_calls("The answer is 42.")
    check(len(calls3) == 0, "no tool calls in plain text")

    calls4 = parser.parse_tool_calls('{"tool": "calculate", "args": {"expression": "5*5"}}')
    check(len(calls4) == 1, "parses bare JSON tool call")

    calls5 = parser.parse_tool_calls('```python\nprint("hello")\n```\n```json\n{"tool": "calculate", "args": {}}\n```')
    check(len(calls5) == 1 and calls5[0].tool_name == "calculate", "skips non-JSON code blocks")

    calls6 = parser.parse_tool_calls('```json\n{broken json\n```')
    check(len(calls6) == 0, "gracefully handles malformed JSON")

    # ━━━ 3. Parallel Execution ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("3. Parallel Execution")

    p_reg = ToolRegistry(tool_timeout_s=5)
    c_a = CounterTool("counter_a")
    c_b = CounterTool("counter_b")
    p_reg.register(c_a)
    p_reg.register(c_b)

    p_results = await p_reg.execute_tools_parallel([
        ParsedToolCall("counter_a", {}),
        ParsedToolCall("counter_b", {}),
        ParsedToolCall("counter_a", {}),
    ])
    check(len(p_results) == 3, "parallel returns correct number of results")
    check(all(r.success for r in p_results), "all parallel calls succeed")
    check(c_a.count == 2 and c_b.count == 1, "parallel calls actually executed")

    p_reg.register(FailingTool())
    p_results2 = await p_reg.execute_tools_parallel([
        ParsedToolCall("counter_a", {}),
        ParsedToolCall("unstable", {}),
    ])
    check(p_results2[0].success and not p_results2[1].success, "parallel isolates failures")

    # ━━━ 4. Agent Loop — Simple Completion ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("4. Agent Loop — Simple Completion")

    simple_llm = MockLLM(["The answer to your question is 42."])
    e1, h1 = collect_events()
    state1 = await run_agent_graph("What is the meaning of life?", simple_llm, on_event=h1, streaming=False)

    check(state1.status == "done", "simple completion: status is done")
    check("42" in state1.result, "simple completion: result contains answer")
    check(state1.tool_calls == 0, "simple completion: no tool calls")
    check(any(e["kind"] == "final_content" for e in e1), "final_content event emitted")
    check(any(e["kind"] == "agent_done" for e in e1), "agent_done event emitted")

    # ━━━ 5. Agent Loop — Tool Usage ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("5. Agent Loop — Tool Usage")

    tool_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "6 * 7"}}\n```',
        "The result of 6 * 7 is 42.",
    ])
    tool_reg = ToolRegistry()
    tool_reg.register(MathTool())
    e2, h2 = collect_events()
    state2 = await run_agent_graph("What is 6*7?", tool_llm, tools=tool_reg, on_event=h2, streaming=False)

    check(state2.status == "done", "tool usage: status is done")
    check(state2.tool_calls == 1, "tool usage: 1 tool call made")
    check("42" in state2.result, "tool usage: final answer correct")
    check(any(e["kind"] == "tool_call" for e in e2), "tool_call event emitted")
    check(any(e["kind"] == "tool_result" for e in e2), "tool_result event emitted")

    # ━━━ 6. Agent Loop — Multi-step Tool Chain ━━━━━━━━━━━━━━━━━━━━━━━━
    section("6. Agent Loop — Multi-step Tool Chain")

    chain_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "10 + 20"}}\n```',
        '```json\n{"tool": "calculate", "args": {"expression": "30 * 2"}}\n```',
        "First I got 30, then 60. Done.",
    ])
    chain_reg = ToolRegistry()
    chain_reg.register(MathTool())
    state3 = await run_agent_graph("Two calcs", chain_llm, tools=chain_reg, streaming=False, on_event=lambda k, d: None)

    check(state3.tool_calls == 2, "multi-step: 2 tool calls")
    check(state3.status == "done", "multi-step: completed")

    # ━━━ 7. Agent Loop — Parallel Tool Calls ━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("7. Agent Loop — Parallel Tool Calls")

    par_llm = MockLLM([
        '```json\n[{"tool": "calculate", "args": {"expression": "1+1"}}, {"tool": "calculate", "args": {"expression": "2+2"}}]\n```',
        "Got 2 and 4.",
    ])
    par_reg = ToolRegistry()
    par_reg.register(MathTool())
    e4, h4 = collect_events()
    state4 = await run_agent_graph("Both", par_llm, tools=par_reg, on_event=h4, streaming=False)

    check(state4.tool_calls == 2, "parallel loop: 2 tool calls")
    check(sum(1 for e in e4 if e["kind"] == "tool_call") == 2, "parallel loop: 2 tool_call events")

    # ━━━ 8. Tool Loop Detection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("8. Tool Loop Detection")

    loop_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```',
    ] * 12)
    loop_reg = ToolRegistry()
    loop_reg.register(MathTool())
    e5, h5 = collect_events()
    state5 = await run_agent_graph("Loop", loop_llm, tools=loop_reg, on_event=h5, streaming=False, max_iterations=20)

    check("loop" in state5.result.lower(), "loop detection: result mentions loop")
    check(any(e["kind"] == "warn" and "loop" in str(e["data"].get("message", "")) for e in e5), "loop detection: warn event")

    # ━━━ 9. Error Handling ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("9. Error Handling")

    err_llm = MockLLM([
        '```json\n{"tool": "unstable", "args": {}}\n```',
        "The tool failed, but I recovered.",
    ])
    err_reg = ToolRegistry()
    err_reg.register(FailingTool())
    state6 = await run_agent_graph("Use unstable", err_llm, tools=err_reg, streaming=False, on_event=lambda k, d: None)

    check(state6.status == "done", "error recovery: agent still completes")
    check("recovered" in state6.result, "error recovery: agent adapts after tool error")

    # ━━━ 10. Event System — Complete Coverage ━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("10. Event System — Complete Coverage")

    ev_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```',
        "Done! The answer is 2.",
    ])
    ev_reg = ToolRegistry()
    ev_reg.register(MathTool())
    e_all, h_all = collect_events()
    await run_agent_graph("Compute 1+1", ev_llm, tools=ev_reg, on_event=h_all, streaming=False)

    kinds = {e["kind"] for e in e_all}
    check("tool_call" in kinds, "event: tool_call present")
    check("tool_result" in kinds, "event: tool_result present")
    check("final_content" in kinds, "event: final_content present")
    check("agent_done" in kinds, "event: agent_done present")

    done_evt = next(e for e in e_all if e["kind"] == "agent_done")
    check(isinstance(done_evt["data"]["elapsed"], float), "agent_done has elapsed")
    check(done_evt["data"]["tool_calls"] == 1, "agent_done has correct tool_calls count")

    # ━━━ 11. Streaming Mode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("11. Streaming Mode")

    stream_llm = MockLLM(["Streamed response!"])
    streamed_chars = 0

    original_chat = stream_llm.chat

    async def counting_chat(messages, on_chunk=None, cancel_event=None, tools=None):
        nonlocal streamed_chars
        if on_chunk:
            def wrapped(ch):
                nonlocal streamed_chars
                streamed_chars += len(ch)
                on_chunk(ch)
            return await original_chat(messages, on_chunk=wrapped, cancel_event=cancel_event, tools=tools)
        return await original_chat(messages, on_chunk=on_chunk, cancel_event=cancel_event, tools=tools)

    stream_llm.chat = counting_chat
    await run_agent_graph("Stream test", stream_llm, streaming=True, on_event=lambda k, d: None)
    check(streamed_chars > 0, f"streaming: received {streamed_chars} chars via on_chunk")

    # ━━━ 12. Custom System Prompt ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("12. Custom System Prompt")

    custom_llm = MockLLM(["Hola, el resultado es 42."])
    await run_agent_graph("Hello", custom_llm, system_prompt="Always respond in Spanish.", streaming=False, on_event=lambda k, d: None)
    sys_msg = next((m for m in custom_llm.last_messages if m.role == "system"), None)
    check(sys_msg is not None and "Spanish" in sys_msg.content, "custom system prompt injected")

    # ━━━ 13. Hook: before_llm ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("13. Hook: before_llm")

    hook_llm = MockLLM(["Hooked!"])
    before_llm_called = False

    def my_before_llm(msgs):
        nonlocal before_llm_called
        before_llm_called = True
        return [*msgs, LLMMessage(role="user", content="[injected]")]

    await run_agent_graph("Hook test", hook_llm, streaming=False, on_event=lambda k, d: None, before_llm=my_before_llm)
    check(before_llm_called, "before_llm hook was called")
    check(any("[injected]" in m.content for m in hook_llm.last_messages), "before_llm injected message")

    # ━━━ 14. Hook: before_tool (block) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("14. Hook: before_tool (block)")

    block_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```',
        "The tool was blocked.",
    ])
    block_reg = ToolRegistry()
    block_reg.register(MathTool())
    e_block, h_block = collect_events()
    state_block = await run_agent_graph(
        "Blocked", block_llm, tools=block_reg, streaming=False, on_event=h_block,
        before_tool=lambda name, args: False,
    )
    check(state_block.tool_calls == 0, "before_tool block: no tool calls executed")

    # ━━━ 15. Hook: after_tool (modify result) ━━━━━━━━━━━━━━━━━━━━━━━━━
    section("15. Hook: after_tool (modify result)")

    after_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "2+2"}}\n```',
        "Modified result received.",
    ])
    after_reg = ToolRegistry()
    after_reg.register(MathTool())
    after_tool_saw = ""

    def my_after_tool(name, args, result):
        nonlocal after_tool_saw
        after_tool_saw = result.output
        return ToolResult(success=result.success, output="MODIFIED", error=result.error)

    await run_agent_graph("After", after_llm, tools=after_reg, streaming=False, on_event=lambda k, d: None, after_tool=my_after_tool)
    check(after_tool_saw == "4", "after_tool received original result")
    check(any("MODIFIED" in m.content for m in after_llm.last_messages), "after_tool modification applied")

    # ━━━ 16. Token Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("16. Token Estimation")

    check(_estimate_tokens("a" * 400, 1.0) == 100, "base token estimation correct")
    check(_estimate_tokens("a" * 400, 2.0) == 200, "multiplied token estimation correct")
    check(_estimate_tokens("", 1.0) == 0, "empty string = 0 tokens")

    # ━━━ 17. Arg Truncation in Old Messages ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("17. Arg Truncation in Old Messages")

    msgs = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="assistant", content='{"tool": "write_file", "args": {"path": "x.txt", "content": "' + "A" * 3000 + '"}}'),
        LLMMessage(role="user", content="[Tool Result] ok"),
    ] + [LLMMessage(role="user", content=f"recent {i}") for i in range(21)]

    result_msgs = _truncate_old_tool_args(msgs, protect_recent=20)
    truncated_msg = result_msgs[1]
    check("...(argument truncated)" in truncated_msg.content, "old write_file args truncated")
    check(len(truncated_msg.content) < len(msgs[1].content), "truncated message is shorter")

    for m in result_msgs[-20:]:
        check("...(argument truncated)" not in m.content, "recent messages NOT truncated")
        break  # just check one to avoid spam

    # non-write_file assistant messages should NOT be truncated
    msgs2 = [
        LLMMessage(role="assistant", content='{"tool": "calculate", "args": {"expression": "' + "1" * 3000 + '"}}'),
    ] + [LLMMessage(role="user", content=f"r{i}") for i in range(21)]
    result2 = _truncate_old_tool_args(msgs2)
    check("...(argument truncated)" not in result2[0].content, "non-write_file args preserved")

    # ━━━ 18. Path Traversal Blocking ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("18. Path Traversal Blocking")

    from clawagents.sandbox.local import LocalBackend
    _local_sb = LocalBackend()

    check(_local_sb.safe_path("src/test.py").startswith(_local_sb.cwd), "safe path within root OK")
    try:
        _local_sb.safe_path("../../../etc/passwd")
        check(False, "traversal should throw")
    except ValueError:
        check(True, "path traversal blocked")

    # ━━━ 19. Exec Safety ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("19. Exec Safety")

    from clawagents.tools.exec import ExecTool
    exec_tool = ExecTool(_local_sb)

    echo_result = await exec_tool.execute({"command": "echo hello"})
    check(echo_result.success and "hello" in echo_result.output, "exec: echo works")

    blocked_result = await exec_tool.execute({"command": "rm -rf /"})
    check(not blocked_result.success and "Blocked" in (blocked_result.error or ""), "exec: dangerous command blocked")

    empty_result = await exec_tool.execute({"command": ""})
    check(not empty_result.success, "exec: empty command fails")

    # ━━━ 20. ClawAgent Class ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("20. ClawAgent Class")

    from clawagents.agent import ClawAgent

    agent_llm = MockLLM(["Agent response: 42."])
    agent_reg = ToolRegistry()
    agent_reg.register(MathTool())

    agent = ClawAgent(llm=agent_llm, tools=agent_reg, streaming=False, use_native_tools=False)
    e_agent, h_agent = collect_events()
    state_agent = await agent.invoke("What is 42?", on_event=h_agent)
    check(state_agent.status == "done", "ClawAgent.invoke completes")
    check("42" in state_agent.result, "ClawAgent.invoke returns correct result")

    # inject_context
    agent.inject_context("Always say hello first")
    agent_llm2 = MockLLM(["Hello! The answer is 42."])
    agent2 = ClawAgent(llm=agent_llm2, tools=agent_reg, streaming=False, use_native_tools=False)
    agent2.inject_context("Always say hello first")
    await agent2.invoke("Test", on_event=lambda k, d: None)
    check(any("[Context]" in m.content for m in agent_llm2.last_messages), "inject_context adds context message")

    # block_tools
    agent3_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```',
        "Blocked.",
    ])
    agent3 = ClawAgent(llm=agent3_llm, tools=agent_reg, streaming=False, use_native_tools=False)
    agent3.block_tools("calculate")
    e3, h3 = collect_events()
    state3b = await agent3.invoke("Calc", on_event=h3)
    check(state3b.tool_calls == 0, "block_tools prevents tool execution")

    # allow_only_tools
    agent4_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```',
        "Allowed.",
    ])
    agent4 = ClawAgent(llm=agent4_llm, tools=agent_reg, streaming=False, use_native_tools=False)
    agent4.allow_only_tools("calculate")
    state4b = await agent4.invoke("Calc", on_event=lambda k, d: None)
    check(state4b.tool_calls == 1, "allow_only_tools permits allowed tool")

    # ━━━ 21. Stable Key Ordering in ToolCallTracker ━━━━━━━━━━━━━━━━━━━━
    section("21. Stable Key Ordering in ToolCallTracker")

    from clawagents.graph.agent_loop import _ToolCallTracker
    tracker_key = _ToolCallTracker(window_size=12, soft_limit=2, hard_limit=4)
    tracker_key.record("read_file", {"path": "a.txt", "mode": "r"})
    tracker_key.record("read_file", {"mode": "r", "path": "a.txt"})
    check(tracker_key.is_soft_looping("read_file", {"path": "a.txt", "mode": "r"}), "tracker: same args different order = same key (sort_keys)")
    check(tracker_key.is_soft_looping("read_file", {"mode": "r", "path": "a.txt"}), "tracker: reverse order also matches")

    # ━━━ 22. ToolCallTracker Serialization Safety ━━━━━━━━━━━━━━━━━━━━━
    section("22. ToolCallTracker Serialization Safety")

    tracker_safe = _ToolCallTracker(window_size=12, soft_limit=2, hard_limit=4)
    try:
        tracker_safe.record("test", {"data": {1, 2, 3}})  # set is not JSON-serializable
        check(True, "tracker: non-serializable args handled without crash")
    except Exception:
        check(False, "tracker: non-serializable args should not crash")

    # ━━━ 23. Hook Exception Safety ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("23. Hook Exception Safety")

    hook_err_llm = MockLLM([
        '```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```',
        "Final after hook error.",
    ])
    hook_err_reg = ToolRegistry()
    hook_err_reg.register(MathTool())
    hook_err_evts, hook_err_h = collect_events()

    def _throwing_before_llm(msgs):
        raise RuntimeError("beforeLLM boom")

    def _throwing_before_tool(name, args):
        raise RuntimeError("beforeTool boom")

    def _throwing_after_tool(name, args, result):
        raise RuntimeError("afterTool boom")

    state_hook_err = await run_agent_graph(
        "Hook errors", hook_err_llm, hook_err_reg,
        streaming=False, on_event=hook_err_h,
        before_llm=_throwing_before_llm,
        before_tool=_throwing_before_tool,
        after_tool=_throwing_after_tool,
    )
    check(state_hook_err.status != "error", "hook errors: agent completed without error status")
    check(any(e["kind"] == "warn" for e in hook_err_evts), "hook errors: warn events emitted for hook failures")

    # ━━━ 24. Max Rounds Sets state.result ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("24. Max Rounds Sets state.result")

    max_responses = [f'```json\n{{"tool": "calculate", "args": {{"expression": "{i}+1"}}}}\n```' for i in range(20)]
    max_llm = MockLLM(max_responses)
    max_reg = ToolRegistry()
    max_reg.register(MathTool())
    state_max = await run_agent_graph(
        "Max rounds", max_llm, max_reg,
        streaming=False, on_event=lambda k, d: None,
        max_iterations=5,
    )
    check("maximum" in state_max.result.lower() or "reached" in state_max.result.lower(), "max rounds: state.result set")
    check(state_max.status == "done", "max rounds: status is done")

    # ━━━ 25. Empty Compaction Summary Handled ━━━━━━━━━━━━━━━━━━━━━━━━━
    section("25. Empty Compaction Summary Handled")

    compact_responses = ['```json\n{"tool": "calculate", "args": {"expression": "1+1"}}\n```'] * 5
    compact_responses.append("Final answer after compaction.")
    compact_llm = MockLLM(compact_responses)
    compact_reg = ToolRegistry()
    compact_reg.register(MathTool())
    state_compact = await run_agent_graph(
        "Compaction test", compact_llm, compact_reg,
        streaming=False, context_window=500, on_event=lambda k, d: None,
    )
    check(state_compact.status in ("done", "error"), "compaction: agent completed without crash")

    # ━━━ 26. Invalid Numeric Args Safety ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("26. Invalid Numeric Args Safety")

    from clawagents.tools.exec import ExecTool
    exec_tool_instance = ExecTool(_local_sb)
    exec_invalid = await exec_tool_instance.execute({"command": "echo ok", "timeout": "not_a_number"})
    check(exec_invalid.success, "exec: invalid timeout falls back to default")

    # ━━━ 27. max_iterations Is Respected ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    section("27. max_iterations Is Respected")

    limit_responses = [f'```json\n{{"tool": "calculate", "args": {{"expression": "{i}+10"}}}}\n```' for i in range(20)]
    limit_llm = MockLLM(limit_responses)
    limit_reg = ToolRegistry()
    limit_reg.register(MathTool())
    state_limit = await run_agent_graph(
        "Limit test", limit_llm, limit_reg,
        max_iterations=4, streaming=False, on_event=lambda k, d: None,
    )
    check(state_limit.tool_calls <= 4, f"max_iterations=4: tool calls ({state_limit.tool_calls}) <= 4")
    check("4" in state_limit.result and "maximum" in state_limit.result.lower(), "max_iterations=4: result message correct")

    # ━━━ 28. ParsedToolCall __eq__ and __hash__ ━━━━━━━━━━━━━━━━━━━━━━━
    section("28. ParsedToolCall __eq__ and __hash__")

    ptc1 = ParsedToolCall("read_file", {"path": "a.txt"})
    ptc2 = ParsedToolCall("read_file", {"path": "a.txt"})
    ptc3 = ParsedToolCall("read_file", {"path": "b.txt"})
    check(ptc1 == ptc2, "ParsedToolCall: equal args are equal")
    check(ptc1 != ptc3, "ParsedToolCall: different args are not equal")
    check(hash(ptc1) == hash(ptc2), "ParsedToolCall: equal objects have same hash")
    check(ptc1 in [ptc2, ptc3], "ParsedToolCall: 'in' works with equality")

    # ━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'━' * 50}")
    print(f"Results: {_passed} passed, {_failed} failed, {_passed + _failed} total")
    if _failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
