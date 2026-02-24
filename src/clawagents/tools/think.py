"""Think Tool — lets the agent reason without side effects.

This is a no-op tool: the agent's "thought" is recorded and returned.
Reduces unnecessary tool calls by giving the agent a structured place
to plan, reason, or reflect before acting.
"""

from typing import Any, Dict, List

from clawagents.tools.registry import Tool, ToolResult


class ThinkTool:
    name = "think"
    description = (
        "Use this tool to think, plan, or reason about the task without taking any action. "
        "Great for breaking down complex problems, evaluating options, or reflecting on results. "
        "Your thought is recorded but has no side effects."
    )
    parameters = {
        "thought": {
            "type": "string",
            "description": "Your reasoning, plan, or analysis",
            "required": True,
        }
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        thought = str(args.get("thought", ""))
        if not thought:
            return ToolResult(success=False, output="", error="No thought provided")
        return ToolResult(success=True, output=f"[Thought recorded]\n{thought}")


think_tools: List[Tool] = [ThinkTool()]
