"""Marketplace install tool."""

from __future__ import annotations

import json
from typing import Any, Dict

from clawagents.config.features import is_enabled
from clawagents.marketplace import install_from_source, list_installed
from clawagents.tools.registry import Tool, ToolResult


class MarketplaceInstallTool:
    name = "marketplace_install"
    description = (
        "Install a skill or plugin from a local path or git URL into "
        ".clawagents/skills or .clawagents/plugins."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "source": {
            "type": "string",
            "description": "Local path or git URL",
            "required": True,
        },
        "kind": {
            "type": "string",
            "description": "skill | plugin (auto-detected when omitted)",
        },
        "name": {
            "type": "string",
            "description": "Optional install name override",
        },
    }

    def __init__(self, workspace: str | None = None):
        self._workspace = workspace

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not is_enabled("marketplace"):
            return ToolResult(success=False, output="", error="marketplace feature disabled")
        source = args.get("source")
        if not source:
            return ToolResult(success=False, output="", error="source required")
        kind = args.get("kind")
        if kind not in (None, "skill", "plugin"):
            return ToolResult(success=False, output="", error="kind must be skill or plugin")
        result = install_from_source(
            str(source),
            kind=kind,  # type: ignore[arg-type]
            workspace=self._workspace,
            name=str(args["name"]) if args.get("name") else None,
        )
        return ToolResult(
            success=result.ok,
            output=json.dumps(result.to_dict(), ensure_ascii=False),
            error=result.error,
        )


class MarketplaceListTool:
    name = "marketplace_list"
    description = "List skills/plugins installed via marketplace_install."
    parameters: Dict[str, Dict[str, Any]] = {}

    def __init__(self, workspace: str | None = None):
        self._workspace = workspace

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        rows = list_installed(self._workspace)
        return ToolResult(success=True, output=json.dumps(rows, ensure_ascii=False, indent=2))


def create_marketplace_tools(workspace: str | None = None) -> list[Tool]:
    return [
        MarketplaceInstallTool(workspace),
        MarketplaceListTool(workspace),
    ]
