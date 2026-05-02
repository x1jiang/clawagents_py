"""Static readiness preview for the ClawAgents CLI."""

from __future__ import annotations

from typing import Any

from clawagents.provider_profiles import resolve_provider_profile


def build_dry_run_preview(
    *,
    task: str = "",
    profile: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_provider_profile(
        profile,
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
    )
    catalog = _tool_catalog()
    matching_tools = _matching_tools(task, catalog)

    ready = bool(resolved.base_url or resolved.api_key or not resolved.profile)
    status = "ready" if ready else "blocked"
    next_actions = ["run the prompt directly"] if ready else ["set an API key or choose a local/base-url profile"]

    return {
        "dry_run": True,
        "status": status,
        "provider": {
            "profile": resolved.profile,
            "provider": resolved.provider,
            "model": resolved.model,
            "base_url": resolved.base_url or "",
            "api_version": resolved.api_version or "",
            "auth": "configured" if resolved.api_key or resolved.base_url else "missing",
        },
        "task": task,
        "tool_count": len(catalog),
        "matching_tools": matching_tools,
        "next_actions": next_actions,
    }


def _tool_catalog() -> list[dict[str, Any]]:
    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.advanced_fs import create_advanced_fs_tools
    from clawagents.tools.background_task import create_background_task_tools
    from clawagents.tools.catalog import create_tool_discovery_tools
    from clawagents.tools.exec import create_exec_tools
    from clawagents.tools.filesystem import create_filesystem_tools
    from clawagents.tools.interactive import interactive_tools
    from clawagents.tools.registry import ToolRegistry
    from clawagents.tools.think import think_tools
    from clawagents.tools.todolist import todolist_tools
    from clawagents.tools.web import web_tools

    sb = LocalBackend()
    registry = ToolRegistry()
    for tool in [
        *todolist_tools,
        *think_tools,
        *interactive_tools,
        *create_filesystem_tools(sb),
        *create_exec_tools(sb),
        *create_advanced_fs_tools(sb),
        *[t for t in web_tools if t.name == "web_fetch"],
        *create_background_task_tools(),
    ]:
        registry.register(tool)
    for tool in create_tool_discovery_tools(registry):
        registry.register(tool)
    return registry.inspect_tools()


def _matching_tools(task: str, catalog: list[dict[str, Any]]) -> list[str]:
    tokens = [token for token in task.lower().replace("_", " ").replace("-", " ").split() if len(token) > 2]
    matches = ["tool_discover"]
    for entry in catalog:
        name = str(entry.get("name", ""))
        haystack = " ".join([
            name,
            str(entry.get("description", "")),
            *[str(k) for k in entry.get("keywords", [])],
        ]).lower()
        if any(token in haystack for token in tokens) and name not in matches:
            matches.append(name)
        if len(matches) >= 10:
            break
    return matches

