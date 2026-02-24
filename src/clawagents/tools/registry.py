"""ClawAgents Tool System

Optimizations learned from deepagents/openclaw:
- Tool description caching (invalidated on register)
- Per-execution timeout (120s default, configurable)
- Head+tail truncation with per-tool context budget
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Protocol


class ToolResult:
    __slots__ = ("success", "output", "error")

    def __init__(self, success: bool, output: str | list[dict[str, Any]], error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error


class Tool(Protocol):
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        ...


class ParsedToolCall:
    __slots__ = ("tool_name", "args")

    def __init__(self, tool_name: str, args: Dict[str, Any]):
        self.tool_name = tool_name
        self.args = args

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedToolCall):
            return NotImplemented
        return self.tool_name == other.tool_name and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.tool_name, frozenset(self.args.items()) if self.args else 0))


# ─── Constants (aligned with deepagents/openclaw) ─────────────────────────

MAX_TOOL_OUTPUT_CHARS = 12_000
_TRUNCATION_HEAD = 5_000
_TRUNCATION_TAIL = 2_000
DEFAULT_TOOL_TIMEOUT_S = 120


def truncate_tool_output(output: str | list[dict[str, Any]], max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str | list[dict[str, Any]]:
    if not isinstance(output, str):
        return output
    if len(output) <= max_chars:
        return output
    head = output[:_TRUNCATION_HEAD]
    tail = output[-_TRUNCATION_TAIL:]
    dropped = len(output) - _TRUNCATION_HEAD - _TRUNCATION_TAIL
    return f"{head}\n\n[… truncated {dropped} characters …]\n\n{tail}"


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```")


class ToolRegistry:
    def __init__(self, tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S):
        self.tools: Dict[str, Tool] = {}
        self._description_cache: Optional[str] = None
        self._tool_timeout_s = tool_timeout_s

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        self._description_cache = None

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list(self) -> List[Tool]:
        return list(self.tools.values())

    def describe_for_llm(self) -> str:
        if self._description_cache is not None:
            return self._description_cache

        tools = self.list()
        if not tools:
            self._description_cache = ""
            return ""

        parts = [
            "## Available Tools\n",
            "You can call tools by responding with a JSON block. For a **single** tool call:",
            '```json\n{"tool": "tool_name", "args": {"param": "value"}}\n```\n',
            "For **multiple independent** tool calls that can run in parallel, use an array:",
            "```json\n[\n"
            '  {"tool": "read_file", "args": {"path": "a.txt"}},\n'
            '  {"tool": "read_file", "args": {"path": "b.txt"}}\n'
            "]\n```\n",
            "Use the array form when the calls are independent (no call depends on another's result).\n",
        ]

        for tool in tools:
            parts.append(f"### {tool.name}\n{tool.description}")
            params = tool.parameters
            if params:
                parts.append("Parameters:")
                for pname, info in params.items():
                    req = " (required)" if info.get("required") else ""
                    parts.append(f"- `{pname}` ({info.get('type')}{req}): {info.get('description')}")
            parts.append("")

        self._description_cache = "\n".join(parts)
        return self._description_cache

    def to_native_schemas(self):
        """Convert registered tools into NativeToolSchema list for native function calling."""
        from clawagents.providers.llm import NativeToolSchema
        return [
            NativeToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in self.list()
        ]

    def parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        calls = self.parse_tool_calls(response)
        if not calls:
            return None
        c = calls[0]
        return {"toolName": c.tool_name, "args": c.args}

    def parse_tool_calls(self, response: str) -> List[ParsedToolCall]:
        def try_parse(text: str) -> List[ParsedToolCall]:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []

            if isinstance(parsed, list):
                return [
                    ParsedToolCall(tool_name=item["tool"], args=item.get("args") or {})
                    for item in parsed
                    if isinstance(item, dict) and isinstance(item.get("tool"), str)
                ]
            if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
                return [ParsedToolCall(tool_name=parsed["tool"], args=parsed.get("args") or {})]
            return []

        for m in _FENCE_RE.finditer(response):
            calls = try_parse(m.group(1))
            if calls:
                return calls

        calls = try_parse(response.strip())
        if calls:
            return calls

        return []

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")
        try:
            result = await asyncio.wait_for(
                tool.execute(args),
                timeout=self._tool_timeout_s,
            )
            return ToolResult(
                success=result.success,
                output=truncate_tool_output(result.output),
                error=result.error,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False, output="",
                error=(
                    f'Tool "{tool_name}" timed out after {self._tool_timeout_s}s. '
                    "For long-running commands, consider using a timeout parameter."
                ),
            )
        except Exception as err:
            return ToolResult(success=False, output="", error=f"Tool error: {str(err)}")

    async def execute_tools_parallel(self, calls: List[ParsedToolCall]) -> List[ToolResult]:
        if not calls:
            return []
        if len(calls) == 1:
            return [await self.execute_tool(calls[0].tool_name, calls[0].args)]

        async def _safe_exec(call: ParsedToolCall) -> ToolResult:
            try:
                return await self.execute_tool(call.tool_name, call.args)
            except Exception as err:
                return ToolResult(success=False, output="", error=f"Tool error: {str(err)}")

        return list(await asyncio.gather(*[_safe_exec(c) for c in calls]))
