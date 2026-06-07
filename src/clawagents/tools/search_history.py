"""search_history tool — cross-session raw message recall."""

from __future__ import annotations

import os
from typing import Any

from clawagents.session.history_search import format_search_history_response, search_history
from clawagents.tools.registry import Tool, ToolResult


class SearchHistoryTool:
    name = "search_history"
    description = (
        "Search raw messages from past agent sessions (cross-session archive). "
        "Returns actual prior user/assistant/tool content snippets, not summaries. "
        "For the current chat only, use the session backend search instead."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "Text to search for",
            "required": True,
        },
        "limit": {
            "type": "integer",
            "description": "Max hits (1-50)",
            "required": False,
        },
        "session_id": {
            "type": "string",
            "description": "Optional: restrict to one archived session id",
            "required": False,
        },
        "include_jsonl": {
            "type": "boolean",
            "description": "Also search JSONL session event logs",
            "required": False,
        },
        "format": {
            "type": "string",
            "description": "Response format: text (default) or json",
            "required": False,
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(success=False, output="", error="query is required")
        limit = int(args.get("limit") or 20)
        session_id = args.get("session_id") or None
        include_jsonl = bool(args.get("include_jsonl", True))
        hits = search_history(
            query,
            limit=min(max(limit, 1), 50),
            session_id=session_id,
            workspace=self._workspace,
            include_jsonl=include_jsonl,
        )
        as_json = str(args.get("format", "")).lower() == "json"
        return ToolResult(
            success=True,
            output=format_search_history_response(query, hits, as_json=as_json),
        )


def create_search_history_tool(workspace: str | None = None) -> Tool:
    return SearchHistoryTool(workspace)
