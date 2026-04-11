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

    # Optional attribute — set ``cacheable = True`` to enable result caching.
    # Not required by Protocol; checked via getattr at runtime.


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


# ─── File Snapshots (learned from Claude Code: fileHistoryMakeSnapshot) ────
# Before write tools modify a file, snapshot it for undo/rollback capability.

_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "create_file", "replace_in_file",
    "insert_in_file", "patch_file",
})


def _snapshot_before_write(tool_name: str, args: Dict[str, Any]) -> None:
    """Snapshot a file before a write tool modifies it."""
    from clawagents.config.features import is_enabled
    if not is_enabled("file_snapshots"):
        return
    if tool_name not in _WRITE_TOOLS:
        return

    import shutil
    import time
    from pathlib import Path

    # Extract file path from common arg names
    path_str = args.get("path") or args.get("file_path") or args.get("target_path") or ""
    if not path_str:
        return

    file_path = Path(path_str)
    if not file_path.exists() or not file_path.is_file():
        return

    try:
        snap_dir = Path.cwd() / ".clawagents" / "snapshots" / str(int(time.time()))
        snap_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(file_path), str(snap_dir / file_path.name))
    except Exception:
        pass  # Snapshot failure should never block tool execution


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


class LazyTool:
    """Deferred tool — the backing module is imported only on first execute()."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        module_path: str,
        class_name: str,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._module_path = module_path
        self._class_name = class_name
        self._resolved: Optional[Tool] = None

    async def execute(self, args: Dict[str, Any]):
        if self._resolved is None:
            import importlib
            mod = importlib.import_module(self._module_path)
            cls = getattr(mod, self._class_name)
            self._resolved = cls()
        return await self._resolved.execute(args)


class ToolRegistry:
    def __init__(
        self,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        cache_max_size: int = 256,
        cache_ttl_s: float = 60.0,
        validate_args: bool = True,
    ):
        self.tools: Dict[str, Tool] = {}
        self._description_cache: Optional[str] = None
        self._tool_timeout_s = tool_timeout_s
        self._validate_args = validate_args

        from clawagents.tools.cache import ResultCacheManager
        self._result_cache = ResultCacheManager(max_size=cache_max_size, default_ttl_s=cache_ttl_s)

    @property
    def result_cache(self):
        return self._result_cache

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        self._description_cache = None

    def register_lazy(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        module_path: str,
        class_name: str,
    ) -> None:
        """Register a tool that will be imported only when first executed."""
        lazy = LazyTool(
            name=name,
            description=description,
            parameters=parameters,
            module_path=module_path,
            class_name=class_name,
        )
        self.tools[name] = lazy  # type: ignore[assignment]
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

        # Parameter validation with lenient coercion
        effective_args = args
        if self._validate_args:
            from clawagents.tools.validate import validate_tool_args, format_validation_errors
            validation = validate_tool_args(tool, args)
            if not validation.valid:
                return ToolResult(
                    success=False, output="",
                    error=f"Invalid parameters:\n{format_validation_errors(validation.errors)}",
                )
            effective_args = validation.coerced

        # Cache lookup for cacheable tools
        is_cacheable = getattr(tool, "cacheable", False)
        if is_cacheable:
            cached = self._result_cache.get(tool_name, effective_args)
            if cached is not None:
                return cached

        try:
            # File snapshot before write tools (Claude Code pattern: fileHistoryMakeSnapshot)
            _snapshot_before_write(tool_name, effective_args)

            result = await asyncio.wait_for(
                tool.execute(effective_args),
                timeout=self._tool_timeout_s,
            )
            truncated = ToolResult(
                success=result.success,
                output=truncate_tool_output(result.output),
                error=result.error,
            )

            # Cache successful results for cacheable tools
            if is_cacheable and truncated.success:
                self._result_cache.set(tool_name, effective_args, truncated)

            return truncated
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
