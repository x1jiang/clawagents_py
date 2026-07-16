"""memory_search tool — hybrid FTS5 + MMR recall from smart memory store."""

from __future__ import annotations

import os
from typing import Any

from clawagents.memory.smart_store import MemorySearchConfig, search_memories
from clawagents.tools.registry import Tool, ToolResult


class MemorySearchTool:
    name = "memory_search"
    description = (
        "Search durable project memories (smart memory store): facts, decisions, "
        "flush notes, and curated MEMORY.md chunks. Uses FTS5 + Jaccard MMR "
        "(exact blake2 dedup; no vector/cosine embeddings)."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "Natural-language search query",
            "required": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Max hits (1-20, default 8)",
            "required": False,
        },
        "min_score": {
            "type": "number",
            "description": "Minimum relevance score 0-1 (default 0.15)",
            "required": False,
        },
    }

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(success=False, output="", error="query is required")
        max_results = int(args.get("max_results") or 8)
        min_score = float(args.get("min_score") if args.get("min_score") is not None else 0.15)
        cfg = MemorySearchConfig(
            max_results=min(max(max_results, 1), 20),
            min_score=max(0.0, min(min_score, 1.0)),
        )
        hits = search_memories(query, workspace=self._workspace, config=cfg)
        if not hits:
            return ToolResult(success=True, output="No matching memories.")
        lines = [f"## memory_search: {query!r} ({len(hits)} hit(s))", ""]
        for i, h in enumerate(hits, 1):
            lines.append(
                f"{i}. [{h.score:.2f}] {h.path} ({h.source})\n   {h.snippet.strip()}"
            )
        return ToolResult(success=True, output="\n".join(lines))


def create_memory_search_tool(workspace: str | None = None) -> Tool:
    return MemorySearchTool(workspace)
