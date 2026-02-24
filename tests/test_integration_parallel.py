"""
Integration tests for parallel tool parsing, execution, and agent-loop wiring.

Run with: python -m pytest tests/test_integration_parallel.py -v
"""

import asyncio
import json
import time
from typing import Any, Dict

import pytest

from clawagents.tools.registry import ParsedToolCall, ToolRegistry, ToolResult


# ─── Mock tools ──────────────────────────────────────────────────────────


class MockTool:
    def __init__(self, name: str, delay_s: float = 0, fail: bool = False, output_size: int = 0):
        self.name = name
        self.description = f"Test tool: {name}"
        self.parameters: Dict[str, Dict[str, Any]] = {
            "path": {"type": "string", "description": "a path", "required": True}
        }
        self._delay = delay_s
        self._fail = fail
        self._output_size = output_size

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._fail:
            return ToolResult(success=False, output="", error=f"{self.name} failed")
        output = f"{self.name}:{json.dumps(args)}"
        if self._output_size > 0:
            output = "x" * self._output_size
        return ToolResult(success=True, output=output)


# ─── Skill loading sanity check ──────────────────────────────────────────


class TestSkillLoading:
    def test_create_claw_agent_with_nonexistent_skills_path(self):
        """create_claw_agent with skills= param registers skill management tools
        even when the directory doesn't exist."""

        from clawagents.agent import create_claw_agent
        from clawagents.providers.llm import LLMProvider, LLMMessage, LLMResponse

        class FakeLLM(LLMProvider):
            name = "fake"
            async def chat(self, messages, on_chunk=None, cancel_event=None):
                return LLMResponse(content="test", model="fake", tokens_used=0)

        agent = create_claw_agent(
            model=FakeLLM(),
            tools=[],
            skills=["/nonexistent/path/that/should/not/crash"],
        )

        tool_names = [t.name for t in agent.tools.list()]
        assert "list_skills" in tool_names
        assert "use_skill" in tool_names


# ─── Multi-fenced-block parsing ──────────────────────────────────────────


class TestMultiFencedBlockParsing:
    def test_non_json_block_then_json_tool_call(self):
        """Parser should skip non-JSON fenced blocks and find the JSON tool call."""
        registry = ToolRegistry()

        response = """Here's an explanation:
```python
print("hello")
```

Now let me read the file:
```json
{"tool": "read_file", "args": {"path": "test.txt"}}
```"""

        calls = registry.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"

    def test_text_block_then_json_array(self):
        """Array tool calls after a non-JSON fenced block."""
        registry = ToolRegistry()

        response = """I'll read both files at once:
```text
Some explanation here
```

```json
[
  {"tool": "read_file", "args": {"path": "a.txt"}},
  {"tool": "read_file", "args": {"path": "b.txt"}}
]
```"""

        calls = registry.parse_tool_calls(response)
        assert len(calls) == 2


# ─── Large parallel result concatenation ─────────────────────────────────


class TestLargeResultConcatenation:
    @pytest.mark.asyncio
    async def test_large_output_passes_through(self):
        registry = ToolRegistry()
        registry.register(MockTool("big_output", output_size=5000))

        calls = [ParsedToolCall("big_output", {}) for _ in range(3)]
        results = await registry.execute_tools_parallel(calls)

        summaries = [
            f"{calls[i].tool_name}({json.dumps(calls[i].args)}) => {r.output}"
            for i, r in enumerate(results)
        ]
        last_result = "\n".join(summaries)

        assert len(results) == 3
        assert len(last_result) > 10000


# ─── Full parallel flow integration ─────────────────────────────────────


class TestFullParallelFlow:
    @pytest.mark.asyncio
    async def test_parse_execute_parallel(self):
        registry = ToolRegistry()
        registry.register(MockTool("read_file", delay_s=0.02))
        registry.register(MockTool("ls", delay_s=0.01))

        llm_response = """```json
[
  {"tool": "read_file", "args": {"path": "a.txt"}},
  {"tool": "read_file", "args": {"path": "b.txt"}},
  {"tool": "ls", "args": {"path": "."}}
]
```"""

        calls = registry.parse_tool_calls(llm_response)
        assert len(calls) == 3

        start = time.monotonic()
        results = await registry.execute_tools_parallel(calls)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.success for r in results)
        assert elapsed < 0.1, f"Should be parallel but took {elapsed:.3f}s"
