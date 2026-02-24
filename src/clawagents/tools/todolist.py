"""TodoList planning tools for structured multi-step task execution.

Provides write_todos and update_todo tools that let the agent plan
before acting. Stores state per-invocation (module-level).
"""

import json
from typing import Any, Dict, List

from clawagents.tools.registry import Tool, ToolResult


# Module-level state (reset on import / new process)
_todos: List[Dict[str, Any]] = []


class WriteTodosTool:
    name = "write_todos"
    description = (
        "Create or replace a todo list for the current task. "
        "Use this at the start of a complex task to plan your approach. "
        "Pass a JSON array of strings describing each step."
    )
    parameters = {
        "todos": {
            "type": "array",
            "items": {"type": "string"},
            "description": "JSON array of todo strings, e.g. [\"Read file\", \"Fix bug\", \"Test\"]",
            "required": True,
        }
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        global _todos
        raw = args.get("todos", [])

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return ToolResult(success=False, output="", error="Invalid JSON array")

        if not isinstance(raw, list):
            return ToolResult(success=False, output="", error="Expected a JSON array of strings")

        _todos = [{"text": str(item), "done": False} for item in raw]
        return ToolResult(success=True, output=_format_todos())


class UpdateTodoTool:
    name = "update_todo"
    description = (
        "Mark a todo item as completed by its index (0-based). "
        "Use after finishing a planned step."
    )
    parameters = {
        "index": {
            "type": "number",
            "description": "0-based index of the todo to mark as complete",
            "required": True,
        }
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        global _todos
        try:
            idx = int(args.get("index", -1))
        except (TypeError, ValueError):
            idx = -1

        if not _todos:
            return ToolResult(success=False, output="", error="No todo list exists. Use write_todos first.")
        if idx < 0 or idx >= len(_todos):
            return ToolResult(success=False, output="", error=f"Index {idx} out of range (0-{len(_todos) - 1})")

        _todos[idx]["done"] = True
        return ToolResult(success=True, output=_format_todos())


def _format_todos() -> str:
    if not _todos:
        return "(no todos)"
    lines = []
    done = sum(1 for t in _todos if t["done"])
    lines.append(f"## Progress: {done}/{len(_todos)} complete\n")
    for i, t in enumerate(_todos):
        mark = "[x]" if t["done"] else "[ ]"
        lines.append(f"{i}. {mark} {t['text']}")
    return "\n".join(lines)


def reset_todos():
    """Reset todo state (for testing)."""
    global _todos
    _todos = []


todolist_tools: List[Tool] = [WriteTodosTool(), UpdateTodoTool()]
