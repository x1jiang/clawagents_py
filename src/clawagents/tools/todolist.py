"""TodoList planning tools for structured multi-step task execution.

Provides write_todos and update_todo tools that let the agent plan
before acting. Agent runs own their todo state; direct calls retain a legacy store.
"""

import json
from typing import Any, Dict, List

from clawagents.tools.registry import Tool, ToolResult
from clawagents.run_context import RunContext


# Module-level state (reset on import / new process)
_todos: List[Dict[str, Any]] = []


class WriteTodosTool:
    name = "write_todos"
    description = (
        "Create or replace a todo list for the current task. "
        "Use this at the start of a broad or long-running task to plan your approach; "
        "skip it for short lookup, read, compare, or JSON-report tasks. "
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

    async def execute(self, args: Dict[str, Any], run_context: RunContext | None = None) -> ToolResult:
        global _todos
        raw = args.get("todos", [])

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return ToolResult(success=False, output="", error="Invalid JSON array")

        if not isinstance(raw, list):
            return ToolResult(success=False, output="", error="Expected a JSON array of strings")

        todos = [{"text": str(item), "done": False} for item in raw]
        if run_context is None:
            _todos = todos
        else:
            run_context.todos = todos
        return ToolResult(success=True, output=_format_todos(todos))


class UpdateTodoTool:
    name = "update_todo"
    description = (
        "Mark a todo item as completed by its index (0-based). "
        "Use after finishing a planned step, but do not call it only to mark "
        "completion when you already have enough evidence to answer."
    )
    parameters = {
        "index": {
            "type": "number",
            "description": "0-based index of the todo to mark as complete",
            "required": True,
        }
    }

    async def execute(self, args: Dict[str, Any], run_context: RunContext | None = None) -> ToolResult:
        todos = _todos if run_context is None else run_context.todos
        try:
            idx = int(args.get("index", -1))
        except (TypeError, ValueError):
            idx = -1

        if not todos:
            return ToolResult(success=False, output="", error="No todo list exists. Use write_todos first.")
        if idx < 0 or idx >= len(todos):
            return ToolResult(success=False, output="", error=f"Index {idx} out of range (0-{len(todos) - 1})")

        todos[idx]["done"] = True
        return ToolResult(success=True, output=_format_todos(todos))


def pending_todos(run_context: RunContext | None) -> list[str]:
    """Only the current run can supply compaction state."""
    return [
        str(todo["text"])
        for todo in getattr(run_context, "todos", ())
        if not todo["done"]
    ]


def _format_todos(todos: list[dict[str, Any]] | None = None) -> str:
    todos = _todos if todos is None else todos
    if not todos:
        return "(no todos)"
    lines = []
    done = sum(1 for t in todos if t["done"])
    lines.append(f"## Progress: {done}/{len(todos)} complete\n")
    for i, t in enumerate(todos):
        mark = "[x]" if t["done"] else "[ ]"
        lines.append(f"{i}. {mark} {t['text']}")
    return "\n".join(lines)


def reset_todos():
    """Reset todo state (for testing)."""
    global _todos
    _todos = []


todolist_tools: List[Tool] = [WriteTodosTool(), UpdateTodoTool()]
