"""retrieve_tool_result — hydrate a previously crushed/offloaded tool output."""

from __future__ import annotations

import json
import os
from typing import Any

from clawagents.tool_output_artifacts import load_tool_artifact_page, search_tool_artifacts
from clawagents.tools.registry import Tool, ToolResult


class RetrieveToolResultTool:
    name = "retrieve_tool_result"
    description = (
        "Fetch a page of a tool output that was crushed or offloaded to "
        "save context (default 16000 characters / 400 lines). Continue with "
        "offset=next_offset from the returned header. Pass the artifact id from a [Crushed tool output … id=…] "
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
            "description": "Page character ceiling (default 16000, maximum 500000)",
            "required": False,
        },
        "offset": {
            "type": "integer",
            "description": "Zero-based Unicode character offset, normally prior next_offset; mutually exclusive with line_start",
            "required": False,
        },
        "line_start": {
            "type": "integer",
            "description": "First line to read, 1-based (LF-delimited); mutually exclusive with offset",
            "required": False,
        },
        "line_count": {
            "type": "integer",
            "description": "Maximum lines in this page (default 400, maximum 10000); max_chars may split a long line",
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
            page_args = {}
            for name in ("offset", "line_start", "line_count", "max_chars"):
                if args.get(name) is not None:
                    value = args[name]
                    if isinstance(value, bool) or isinstance(value, float):
                        raise ValueError(name)
                    page_args[name] = int(value)
        except (TypeError, ValueError):
            return ToolResult(success=False, output="", error="Page ranges must be integers")
        ok, text, meta = load_tool_artifact_page(
            artifact_id, workspace=self._workspace, **page_args
        )
        if not ok:
            return ToolResult(success=False, output="", error=text)
        meta = meta or {}
        header = (
            f"[artifact id={meta.get('id', artifact_id)} offset={meta['offset']} next_offset={meta['next_offset']} eof={str(meta['eof']).lower()}]\n"
            f"tool={meta.get('tool_name', '?')} kind={meta.get('kind', '?')} line_start={meta['line_start']}; continue with next_offset\n"
        )
        return ToolResult(success=True, output=header + text)


def create_retrieve_tool_result_tool(workspace: str | None = None) -> Tool:
    return RetrieveToolResultTool(workspace)
