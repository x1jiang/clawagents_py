from __future__ import annotations

import json
from typing import Any, Literal

from clawagents.permissions.mode import is_write_class_tool
from clawagents.tools.registry import ToolRegistry, ToolResult

ToolProfileName = Literal["minimal", "read-only", "write", "full"]

_DISCOVERY_TOOLS = {"tool_discover", "tool_describe", "tool_profile"}


def _tool_keywords(tool: Any) -> list[str]:
    raw = getattr(tool, "keywords", [])
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return []


def names_for_tool_profile(
    registry: ToolRegistry,
    profile: str = "full",
) -> list[str]:
    names = [tool.name for tool in registry.list()]
    if profile == "minimal":
        return [name for name in names if name in _DISCOVERY_TOOLS]
    if profile == "read-only":
        return [name for name in names if name in _DISCOVERY_TOOLS or not is_write_class_tool(name)]
    if profile == "write":
        return [name for name in names if name in _DISCOVERY_TOOLS or is_write_class_tool(name)]
    return names


class _ToolDiscover:
    name = "tool_discover"
    description = "Search the compact tool catalog by name, description, or named profile."
    parameters = {
        "query": {"type": "string", "description": "Optional case-insensitive search text."},
        "profile": {"type": "string", "description": "Tool profile: minimal, read-only, write, or full."},
        "limit": {"type": "number", "description": "Maximum results to return."},
    }
    parallel_safe = True

    def __init__(self, registry: ToolRegistry, max_results: int = 25, allowed_names=None):
        self._registry = registry
        self._max_results = max(1, max_results)
        self._allowed_names = allowed_names or (lambda profile: set(names_for_tool_profile(registry, profile)))

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip().lower()
        profile = str(args.get("profile") or "full")
        limit = max(1, int(args.get("limit") or self._max_results))
        allowed = self._allowed_names(profile)
        rows = []
        for tool in self._registry.list():
            if tool.name not in allowed:
                continue
            if profile != "minimal" and tool.name in _DISCOVERY_TOOLS:
                continue
            keywords = _tool_keywords(tool)
            if (
                query
                and query not in tool.name.lower()
                and query not in tool.description.lower()
                and not any(query in keyword.lower() for keyword in keywords)
            ):
                continue
            row = {"name": tool.name, "description": tool.description}
            if keywords:
                row["keywords"] = keywords
            rows.append(row)
            if len(rows) >= limit:
                break
        return ToolResult(True, json.dumps(rows))


class _ToolDescribe:
    name = "tool_describe"
    description = "Return the full schema for one registered tool."
    parameters = {"name": {"type": "string", "description": "Tool name to describe.", "required": True}}
    parallel_safe = True

    def __init__(self, registry: ToolRegistry, allowed_names=None):
        self._registry = registry
        self._allowed_names = allowed_names or (lambda profile: set(names_for_tool_profile(registry, profile)))

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        name = str(args.get("name") or "")
        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(False, "", f"Unknown tool: {name}")
        if name not in self._allowed_names("full"):
            return ToolResult(False, "", f"Tool is outside discovery profile: {name}")
        return ToolResult(True, json.dumps({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "keywords": _tool_keywords(tool),
            "cacheable": getattr(tool, "cacheable", False) is True,
            "parallel_safe": getattr(tool, "parallel_safe", False) is True,
        }))


class _ToolProfile:
    name = "tool_profile"
    description = "List tool names included in a compact profile."
    parameters = {"profile": {"type": "string", "description": "Profile: minimal, read-only, write, or full.", "required": True}}
    parallel_safe = True

    def __init__(self, registry: ToolRegistry, allowed_names=None):
        self._registry = registry
        self._allowed_names = allowed_names or (lambda profile: set(names_for_tool_profile(registry, profile)))

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(True, json.dumps(list(self._allowed_names(str(args.get("profile") or "full")))))


def create_tool_discovery_tools(
    registry: ToolRegistry,
    *,
    max_results: int = 25,
    max_profile: ToolProfileName = "full",
):
    def allowed_names(profile: str) -> set[str]:
        boundary = set(names_for_tool_profile(registry, max_profile))
        return {name for name in names_for_tool_profile(registry, profile) if name in boundary}

    return [
        _ToolDiscover(registry, max_results=max_results, allowed_names=allowed_names),
        _ToolDescribe(registry, allowed_names=allowed_names),
        _ToolProfile(registry, allowed_names=allowed_names),
    ]
