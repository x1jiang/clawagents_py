"""retrieve_tool_result — hydrate a previously crushed/offloaded tool output."""

from __future__ import annotations

import json
import os
from typing import Any

from clawagents.tool_output_artifacts import load_tool_artifact, search_tool_artifacts
from clawagents.tools.registry import Tool, ToolResult


class RetrieveToolResultTool:
    name = "retrieve_tool_result"
    description = (
        "Fetch the full text of a tool output that was crushed or offloaded to "
        "save context. Pass the artifact id from a [Crushed tool output … id=…] "
        "or [Tool output truncated] message. "
        "Alternatively pass query= to search stored artifacts locally."
    )
    parameters = {
        "id": {
            "type": "string",
            "description": "Artifact id from a crushed/offloaded tool result",
            "required": False,
        },
        "query": {
            "type": "string",
            "description": "Search stored tool artifacts by substring (local only)",
            "required": False,
        },
        "max_chars": {
            "type": "integer",
            "description": "Optional cap on returned characters (default 100000)",
            "required": False,
        },
        "limit": {
            "type": "integer",
            "description": "Max search hits when using query (default 20)",
            "required": False,
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if query and not str(args.get("id") or args.get("artifact_id") or "").strip():
            try:
                limit = int(args.get("limit") or 20)
            except (TypeError, ValueError):
                limit = 20
            hits = search_tool_artifacts(query, workspace=self._workspace, limit=limit)
            if not hits:
                return ToolResult(success=True, output=f"No tool artifacts matched query={query!r}")
            return ToolResult(
                success=True,
                output=json.dumps({"query": query, "hits": hits}, indent=2),
            )

        artifact_id = str(args.get("id") or args.get("artifact_id") or "").strip()
        if not artifact_id:
            return ToolResult(success=False, output="", error="id or query is required")
        try:
            max_chars = int(args.get("max_chars") or 100_000)
        except (TypeError, ValueError):
            max_chars = 100_000
        max_chars = max(1_000, min(max_chars, 500_000))
        ok, text, meta = load_tool_artifact(
            artifact_id, workspace=self._workspace, max_chars=max_chars
        )
        if not ok:
            return ToolResult(success=False, output="", error=text)
        header = ""
        if meta:
            header = (
                f"tool={meta.get('tool_name', '?')} kind={meta.get('kind', '?')} "
                f"chars={meta.get('chars', '?')}\n\n"
            )
        return ToolResult(success=True, output=header + text)


def create_retrieve_tool_result_tool(workspace: str | None = None) -> Tool:
    return RetrieveToolResultTool(workspace)
