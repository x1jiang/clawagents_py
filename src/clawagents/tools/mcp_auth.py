"""Tool for updating MCP auth and reconnecting active sessions."""

from __future__ import annotations

from typing import Any

from clawagents.mcp.manager import MCPServerManager
from clawagents.tools.registry import ToolResult


class MCPAuthTool:
    name = "mcp_auth"
    description = "Configure auth for an MCP server and reconnect active sessions when possible."
    keywords = ["mcp", "auth", "bearer", "header", "env", "reconnect"]
    parameters = {
        "server_name": {"type": "string", "description": "Configured MCP server name.", "required": True},
        "mode": {"type": "string", "description": "Auth mode: bearer, header, or env.", "required": True},
        "value": {"type": "string", "description": "Secret value to apply.", "required": True},
        "key": {"type": "string", "description": "Header or env key override."},
    }

    def __init__(self, manager: MCPServerManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            await self._manager.update_server_auth(
                str(args.get("server_name") or ""),
                mode=str(args.get("mode") or ""),
                value=str(args.get("value") or ""),
                key=(str(args["key"]) if args.get("key") is not None else None),
                reconnect=True,
            )
        except Exception as exc:
            return ToolResult(False, "", str(exc))
        return ToolResult(True, f"Saved MCP auth for {args.get('server_name')}")

