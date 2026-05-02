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
    keywords: List[str]
    parameters: Dict[str, Dict[str, Any]]

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        ...

    # Optional attribute — set ``cacheable = True`` to enable result caching.
    # Not required by Protocol; checked via getattr at runtime.
    # Optional attribute — set ``keywords = [...]`` to improve compact discovery.


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


# ─── Parallel-execution policy (learned from Hermes) ──────────────────────
# A tool is run concurrently with siblings only when it is parallel-safe AND
# its path scope (if any) does not collide with another call's path scope.
#
# Tools may opt-in by setting a ``parallel_safe = True`` class attribute, and
# may declare a ``path_scoped_arg`` naming the argument whose value identifies
# the resource they touch (so two ``read_file`` calls on different paths can
# run concurrently, but two ``write_file`` calls on the same path cannot).
#
# Tools listed in ``_NEVER_PARALLEL_TOOLS`` are always run alone, regardless
# of parallel-safe flags — interactive tools must serialize.

_NEVER_PARALLEL_TOOLS: frozenset[str] = frozenset({
    "ask_user", "clarify", "confirm", "approve_action",
})

# Tools that read filesystem-like resources are safe to fan out as long as
# their target paths are independent. Write-side tools are intentionally
# excluded — they run sequentially via the snapshot path.
_DEFAULT_PARALLEL_SAFE: frozenset[str] = frozenset({
    "read_file", "list_dir", "glob", "search_files", "grep",
    "web_fetch", "shell",  # stateless reads
})

# Default declarations for path-scoped tools. Tools may override by setting
# their own ``path_scoped_arg`` attribute.
_DEFAULT_PATH_SCOPED_ARGS: Dict[str, str] = {
    "read_file": "path",
    "list_dir": "path",
    "glob": "path",
    "search_files": "path",
    "grep": "path",
    "web_fetch": "url",
}

MAX_PARALLEL_TOOL_WORKERS = 8


def _is_parallel_safe(tool: "Tool") -> bool:
    if tool.name in _NEVER_PARALLEL_TOOLS:
        return False
    flag = getattr(tool, "parallel_safe", None)
    if flag is True:
        return True
    if flag is False:
        return False
    return tool.name in _DEFAULT_PARALLEL_SAFE


def _path_scope_of(tool: "Tool", args: Dict[str, Any]) -> Optional[str]:
    arg_name = getattr(tool, "path_scoped_arg", None) or _DEFAULT_PATH_SCOPED_ARGS.get(tool.name)
    if not arg_name:
        return None
    val = args.get(arg_name)
    if val is None:
        return None
    return str(val)


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
    marker_budget = 40
    payload_budget = max(20, max_chars - marker_budget)
    head_chars = min(_TRUNCATION_HEAD, max(1, int(payload_budget * 0.7)))
    tail_chars = min(_TRUNCATION_TAIL, max(1, payload_budget - head_chars))
    head = output[:head_chars]
    tail = output[-tail_chars:]
    dropped = max(0, len(output) - len(head) - len(tail))
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
        keywords: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.keywords = list(keywords or [])
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
        result_cache: Any = None,
    ):
        self.tools: Dict[str, Tool] = {}
        self._description_cache: Optional[str] = None
        self._tool_timeout_s = tool_timeout_s
        self._validate_args = validate_args

        if result_cache is not None:
            self._result_cache = result_cache
        else:
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
        self.tools[name] = lazy
        self._description_cache = None

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list(self) -> List[Tool]:
        return list(self.tools.values())

    def inspect_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "keywords": list(getattr(tool, "keywords", [])),
                "cacheable": getattr(tool, "cacheable", False) is True,
                "parallel_safe": _is_parallel_safe(tool),
                "path_scoped_arg": (
                    getattr(tool, "path_scoped_arg", None)
                    or _DEFAULT_PATH_SCOPED_ARGS.get(tool.name)
                ),
            }
            for tool in self.list()
        ]

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

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        run_context: Any = None,
    ) -> ToolResult:
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

        # Plan-mode gate: refuse write-class tools when run_context is in PLAN mode.
        # Kept at the registry level (not in agent_loop) so all execution paths
        # see the same gate, including parallel dispatch.
        from clawagents.permissions.mode import (
            PermissionMode,
            evaluate_tool_permission,
        )

        if run_context is not None:
            mode = getattr(run_context, "permission_mode", PermissionMode.DEFAULT)
            file_path = (
                args.get("path")
                or args.get("file_path")
                or args.get("filePath")
            )
            decision = evaluate_tool_permission(
                tool_name,
                mode=mode,
                file_path=file_path if isinstance(file_path, str) else None,
                command=args.get("command") if isinstance(args.get("command"), str) else None,
            )
            if not decision.allowed and not decision.requires_confirmation:
                return ToolResult(
                    success=False, output="",
                    error=(
                        f"Refused: '{tool_name}' is a write-class tool and you are in "
                        "plan mode. Call exit_plan_mode first, or restrict yourself "
                        "to read-only tools while planning."
                    ),
                )

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

            # Route run_context to tools that declare it in their execute signature.
            execute_awaitable = _call_tool_execute(tool, effective_args, run_context)
            result = await asyncio.wait_for(
                execute_awaitable,
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

    async def execute_tools_parallel(
        self,
        calls: List[ParsedToolCall],
        *,
        run_context: Any = None,
    ) -> List[ToolResult]:
        """Execute calls with Hermes-style path-scoped parallelism.

        Calls are partitioned into ordered batches:
        * Never-parallel tools and unsafe tools form singleton batches.
        * Parallel-safe tools are merged into the trailing batch when their
          path scope (if any) does not collide with the existing scopes.
        * Each batch runs concurrently with at most ``MAX_PARALLEL_TOOL_WORKERS``
          in flight; batches run strictly in order to preserve write semantics.
        """
        if not calls:
            return []
        if len(calls) == 1:
            return [await self.execute_tool(calls[0].tool_name, calls[0].args, run_context=run_context)]

        async def _safe_exec(call: ParsedToolCall) -> ToolResult:
            try:
                return await self.execute_tool(call.tool_name, call.args, run_context=run_context)
            except Exception as err:
                return ToolResult(success=False, output="", error=f"Tool error: {str(err)}")

        # Build batches. Each batch is a list of (original_index, call).
        batches: List[List[tuple[int, ParsedToolCall]]] = []
        scopes_per_batch: List[set[str]] = []

        for idx, call in enumerate(calls):
            tool = self.tools.get(call.tool_name)
            psafe = bool(tool and _is_parallel_safe(tool))
            scope = _path_scope_of(tool, call.args) if tool else None

            if not psafe:
                batches.append([(idx, call)])
                scopes_per_batch.append(set())
                continue

            if batches:
                # Try to merge into the trailing batch only if it is itself
                # parallel-safe (i.e. all members were parallel-safe and no
                # path-scope collision exists).
                last = batches[-1]
                last_scopes = scopes_per_batch[-1]
                last_tool = self.tools.get(last[0][1].tool_name)
                last_psafe = bool(last_tool and _is_parallel_safe(last_tool))
                if (
                    last_psafe
                    and (scope is None or scope not in last_scopes)
                    and len(last) < MAX_PARALLEL_TOOL_WORKERS
                ):
                    last.append((idx, call))
                    if scope is not None:
                        last_scopes.add(scope)
                    continue

            batches.append([(idx, call)])
            scopes_per_batch.append({scope} if scope is not None else set())

        results: List[Optional[ToolResult]] = [None] * len(calls)
        for batch in batches:
            if len(batch) == 1:
                idx, call = batch[0]
                results[idx] = await _safe_exec(call)
            else:
                tasks = [_safe_exec(c) for _, c in batch]
                done = await asyncio.gather(*tasks)
                for (idx, _), res in zip(batch, done):
                    results[idx] = res

        return [r if r is not None else ToolResult(success=False, output="", error="missing result") for r in results]


def _call_tool_execute(tool: Tool, args: Dict[str, Any], run_context: Any):
    """Invoke ``tool.execute`` with ``run_context`` only when it is accepted.

    This keeps the public Tool protocol minimal: legacy tools that define
    ``async def execute(self, args)`` work as-is, while new tools can opt in
    to the typed context by declaring ``run_context``.
    """
    if run_context is None:
        return tool.execute(args)

    from clawagents.function_tool import (
        FunctionTool,
        _tool_signature_accepts_run_context,
    )

    if isinstance(tool, FunctionTool):
        return tool.execute(args, run_context=run_context)

    accepts, param_name = _tool_signature_accepts_run_context(tool)
    if accepts and param_name:
        try:
            return tool.execute(args, **{param_name: run_context})
        except TypeError:
            return tool.execute(args)
    return tool.execute(args)
