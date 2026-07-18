"""Registry must preserve full tool output on ToolResult.raw_output."""

from __future__ import annotations

import asyncio

from clawagents.tools.registry import ToolRegistry, ToolResult, truncate_tool_output


class _BigTool:
    name = "big_dump"
    description = "emits a large string"
    keywords = ["big"]
    parameters: dict = {}

    async def execute(self, args: dict) -> ToolResult:
        return ToolResult(success=True, output="HEAD" + ("Z" * 50_000) + "TAIL")


def test_truncate_shortens_but_raw_output_keeps_full():
    full = "HEAD" + ("Z" * 50_000) + "TAIL"
    preview = truncate_tool_output(full)
    assert len(preview) < len(full)
    assert "TAIL" in preview


def test_registry_execute_preserves_raw_output():
    reg = ToolRegistry()
    reg.register(_BigTool())

    async def _run():
        return await reg.execute_tool("big_dump", {})

    result = asyncio.run(_run())
    assert result.success
    assert isinstance(result.output, str)
    assert len(result.output) < 50_000
    assert isinstance(result.raw_output, str)
    assert len(result.raw_output) > 50_000
    assert result.raw_output.endswith("TAIL")
    assert result.raw_output.startswith("HEAD")
