from __future__ import annotations

import re
from typing import Any, Iterable

from clawagents.tools.registry import ToolRegistry, ToolResult


_DEFAULT_ALLOWED_TOOLS = {
    "ls", "read_file", "grep", "glob", "tree", "diff", "web_fetch", "echo",
}
_SUB_RE = re.compile(r"\$\{([A-Za-z0-9_.-]+)\.output\}")


def _substitute(value: Any, results: dict[str, ToolResult]) -> Any:
    if isinstance(value, str):
        return _SUB_RE.sub(lambda m: str(results.get(m.group(1)).output) if results.get(m.group(1)) else "", value)
    if isinstance(value, list):
        return [_substitute(item, results) for item in value]
    if isinstance(value, dict):
        return {k: _substitute(v, results) for k, v in value.items()}
    return value


class ToolProgramTool:
    name = "tool_program"
    description = (
        "Run a bounded read-only sequence of tool calls with ${step.output} substitutions. "
        "Use this for deterministic multi-step lookups without returning every intermediate result."
    )
    parameters = {
        "steps": {
            "type": "array",
            "description": "Ordered steps: {id?: string, tool: string, args?: object}.",
            "required": True,
            "items": {"type": "object"},
        }
    }

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_tools: Iterable[str] | None = None,
        max_steps: int = 8,
    ):
        self._registry = registry
        self._allowed_tools = set(allowed_tools or _DEFAULT_ALLOWED_TOOLS)
        self._max_steps = max(1, max_steps)

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        steps = args.get("steps")
        if not isinstance(steps, list):
            return ToolResult(success=False, output="", error="tool_program requires a steps array")
        if len(steps) > self._max_steps:
            return ToolResult(success=False, output="", error=f"tool_program supports at most {self._max_steps} steps")

        results: dict[str, ToolResult] = {}
        last = ToolResult(success=True, output="")
        for idx, raw_step in enumerate(steps):
            if not isinstance(raw_step, dict) or not isinstance(raw_step.get("tool"), str):
                return ToolResult(success=False, output="", error=f"Step {idx + 1} is missing a tool name")
            tool_name = raw_step["tool"]
            if tool_name == self.name or tool_name not in self._allowed_tools:
                return ToolResult(success=False, output="", error=f"Step {idx + 1} uses disallowed tool: {tool_name}")

            effective_args = _substitute(dict(raw_step.get("args") or {}), results)
            last = await self._registry.execute_tool(tool_name, effective_args)
            key = str(raw_step.get("id") or idx)
            results[key] = last
            results[str(idx)] = last
            if not last.success:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Step {idx + 1}/{len(steps)} ({tool_name}) failed: {last.error or 'unknown error'}",
                )

        return last


def create_tool_program_tool(
    registry: ToolRegistry,
    *,
    allowed_tools: Iterable[str] | None = None,
    max_steps: int = 8,
) -> ToolProgramTool:
    return ToolProgramTool(registry, allowed_tools=allowed_tools, max_steps=max_steps)
