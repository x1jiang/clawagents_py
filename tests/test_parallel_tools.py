"""
Unit tests for ToolRegistry — parallel tool parsing & execution.
Run with: python -m pytest tests/test_parallel_tools.py -v
"""

import asyncio
import json
import time
from typing import Any, Dict

import pytest

from clawagents.tools.registry import ParsedToolCall, ToolRegistry, ToolResult


# ─── Helpers ──────────────────────────────────────────────────────────────


class MockTool:
    def __init__(self, name: str, delay_s: float = 0, fail: bool = False):
        self.name = name
        self.description = f"Test tool: {name}"
        self.parameters: Dict[str, Dict[str, Any]] = {}
        self._delay = delay_s
        self._fail = fail

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._fail:
            return ToolResult(success=False, output="", error=f"{self.name} failed")
        return ToolResult(success=True, output=f"{self.name}:{json.dumps(args)}")


@pytest.fixture
def registry():
    r = ToolRegistry()
    return r


@pytest.fixture
def loaded_registry():
    r = ToolRegistry()
    r.register(MockTool("fast_tool", delay_s=0.01))
    r.register(MockTool("slow_tool", delay_s=0.05))
    r.register(MockTool("fail_tool", fail=True))
    return r


# ─── parse_tool_calls ─────────────────────────────────────────────────────


class TestParseToolCalls:
    def test_single_fenced_json(self, registry):
        response = '```json\n{"tool": "read_file", "args": {"path": "a.txt"}}\n```'
        calls = registry.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"
        assert calls[0].args == {"path": "a.txt"}

    def test_single_bare_json(self, registry):
        response = '{"tool": "ls", "args": {"dir": "."}}'
        calls = registry.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].tool_name == "ls"

    def test_array_fenced_json(self, registry):
        response = """```json
[
  {"tool": "read_file", "args": {"path": "a.txt"}},
  {"tool": "read_file", "args": {"path": "b.txt"}},
  {"tool": "ls", "args": {"dir": "src"}}
]
```"""
        calls = registry.parse_tool_calls(response)
        assert len(calls) == 3
        assert calls[0].tool_name == "read_file"
        assert calls[1].tool_name == "read_file"
        assert calls[2].tool_name == "ls"
        assert calls[2].args == {"dir": "src"}

    def test_array_bare_json(self, registry):
        response = '[{"tool": "write_file", "args": {"path": "x.txt", "content": "hello"}}, {"tool": "ls"}]'
        calls = registry.parse_tool_calls(response)
        assert len(calls) == 2
        assert calls[0].tool_name == "write_file"
        assert calls[1].tool_name == "ls"
        assert calls[1].args == {}

    def test_non_json_response(self, registry):
        calls = registry.parse_tool_calls("I think we should read the file first.")
        assert len(calls) == 0

    def test_json_without_tool_key(self, registry):
        calls = registry.parse_tool_calls('{"action": "read", "file": "a.txt"}')
        assert len(calls) == 0

    def test_filters_invalid_entries_in_array(self, registry):
        response = '[{"tool": "ls", "args": {}}, {"not_a_tool": true}, {"tool": "read_file", "args": {"path": "x"}}]'
        calls = registry.parse_tool_calls(response)
        assert len(calls) == 2
        assert calls[0].tool_name == "ls"
        assert calls[1].tool_name == "read_file"

    def test_legacy_parse_tool_call_returns_first(self, registry):
        response = '[{"tool": "ls", "args": {}}, {"tool": "read_file", "args": {"path": "x"}}]'
        call = registry.parse_tool_call(response)
        assert call is not None
        assert call["toolName"] == "ls"

    def test_legacy_parse_tool_call_returns_none(self, registry):
        call = registry.parse_tool_call("Just text, no tool calls")
        assert call is None


# ─── execute_tools_parallel ───────────────────────────────────────────────


class TestExecuteToolsParallel:
    @pytest.mark.asyncio
    async def test_single_call(self, loaded_registry):
        results = await loaded_registry.execute_tools_parallel([
            ParsedToolCall("fast_tool", {"x": 1}),
        ])
        assert len(results) == 1
        assert results[0].success is True
        assert "fast_tool" in results[0].output

    @pytest.mark.asyncio
    async def test_empty_input(self, loaded_registry):
        results = await loaded_registry.execute_tools_parallel([])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_parallel_execution(self, loaded_registry):
        start = time.monotonic()
        results = await loaded_registry.execute_tools_parallel([
            ParsedToolCall("fast_tool", {"a": 1}),
            ParsedToolCall("slow_tool", {"b": 2}),
            ParsedToolCall("fast_tool", {"c": 3}),
        ])
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.success for r in results)
        # If truly parallel, total time should be closer to max(0.05) than sum(0.01+0.05+0.01=0.07)
        assert elapsed < 0.15, f"Expected parallel execution, but took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_preserves_order(self, loaded_registry):
        results = await loaded_registry.execute_tools_parallel([
            ParsedToolCall("slow_tool", {"order": "first"}),
            ParsedToolCall("fast_tool", {"order": "second"}),
        ])
        assert len(results) == 2
        assert "slow_tool" in results[0].output
        assert "fast_tool" in results[1].output

    @pytest.mark.asyncio
    async def test_failure_isolation(self, loaded_registry):
        results = await loaded_registry.execute_tools_parallel([
            ParsedToolCall("fast_tool", {}),
            ParsedToolCall("fail_tool", {}),
            ParsedToolCall("fast_tool", {}),
        ])
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert "fail_tool failed" in results[1].error
        assert results[2].success is True

    @pytest.mark.asyncio
    async def test_unknown_tool(self, loaded_registry):
        results = await loaded_registry.execute_tools_parallel([
            ParsedToolCall("nonexistent_tool", {}),
            ParsedToolCall("fast_tool", {}),
        ])
        assert len(results) == 2
        assert results[0].success is False
        assert "Unknown tool" in results[0].error
        assert results[1].success is True


# ─── describe_for_llm ────────────────────────────────────────────────────


class TestDescribeForLLM:
    def test_includes_parallel_syntax(self, loaded_registry):
        desc = loaded_registry.describe_for_llm()
        assert "multiple independent" in desc
        assert "array" in desc
        assert "parallel" in desc

    def test_empty_registry(self, registry):
        assert registry.describe_for_llm() == ""


# ─── execute_tool (single) ───────────────────────────────────────────────


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_registered_tool(self):
        r = ToolRegistry()
        r.register(MockTool("echo"))
        result = await r.execute_tool("echo", {"msg": "hi"})
        assert result.success is True
        assert "echo" in result.output

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        r = ToolRegistry()
        result = await r.execute_tool("ghost", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_tool_exception(self):
        class ThrowTool:
            name = "thrower"
            description = "throws"
            parameters: Dict[str, Dict[str, Any]] = {}

            async def execute(self, args: Dict[str, Any]) -> ToolResult:
                raise RuntimeError("boom")

        r = ToolRegistry()
        r.register(ThrowTool())
        result = await r.execute_tool("thrower", {})
        assert result.success is False
        assert "boom" in result.error
