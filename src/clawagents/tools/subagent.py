"""Sub-agent delegation via the `task` tool.

Spawns an isolated ClawAgent.invoke() with a fresh context window.
Only the final result is returned to the parent agent.

Supports typed SubAgentSpec for per-agent configuration (name, prompt, etc.)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clawagents.providers.llm import LLMProvider
from clawagents.tools.registry import Tool, ToolRegistry, ToolResult
from clawagents.process.command_queue import enqueue_command_in_lane
from clawagents.process.lanes import CommandLane


@dataclass
class SubAgentSpec:
    """Specification for a named sub-agent with its own configuration.

    When the parent dispatches a task with a matching ``agent`` name,
    these settings override the defaults.
    """

    name: str
    """Unique name for this sub-agent type (e.g., 'researcher', 'coder')."""

    description: str
    """Human-readable description of what this sub-agent does."""

    system_prompt: Optional[str] = None
    """System prompt for this sub-agent."""

    max_iterations: int = 5
    """Max tool rounds. Default: 5."""

    use_native_tools: bool = True
    """Whether to use native tool calling for this sub-agent."""


class TaskTool:
    name = "task"

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        subagents: Optional[List[SubAgentSpec]] = None,
        use_queue: bool = False,
    ):
        self._llm = llm
        self._tools = tools
        self._subagents = subagents or []
        self._use_queue = use_queue

        agent_names = [s.name for s in self._subagents]
        agent_list = f" Available specialized agents: {', '.join(agent_names)}." if agent_names else ""
        self.description = (
            "Delegate a task to a sub-agent with its own isolated context window. "
            "Use for complex sub-tasks that would clutter your main context. "
            "The sub-agent has access to the same tools but a fresh conversation."
            + agent_list
        )
        self.parameters = {
            "description": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
                "required": True,
            },
            "agent": {
                "type": "string",
                "description": f"Optional: name of a specialized sub-agent to use.{' Options: ' + ', '.join(agent_names) if agent_names else ''}",
            },
            "max_iterations": {
                "type": "number",
                "description": "Max tool rounds for the sub-agent. Default: 5",
            },
        }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        from clawagents.graph.agent_loop import run_agent_graph

        description = str(args.get("description", ""))
        agent_name = args.get("agent")
        try:
            max_iter = max(1, int(args.get("max_iterations", 5)))
        except (TypeError, ValueError):
            max_iter = 5

        if not description:
            return ToolResult(success=False, output="", error="No task description provided")

        spec: Optional[SubAgentSpec] = None
        if agent_name:
            spec = next((s for s in self._subagents if s.name == str(agent_name)), None)

        effective_max_iter = spec.max_iterations if spec else max_iter
        effective_prompt = spec.system_prompt if spec else None
        effective_native_tools = spec.use_native_tools if spec else True

        async def do_run() -> ToolResult:
            state = await run_agent_graph(
                task=description,
                llm=self._llm,
                tools=self._tools,
                system_prompt=effective_prompt,
                max_iterations=effective_max_iter,
                streaming=False,
                use_native_tools=effective_native_tools,
            )

            if state.status == "error":
                return ToolResult(
                    success=False,
                    output=state.result or "",
                    error=f"Sub-agent failed: {state.result}",
                )

            agent_label = f"Sub-agent [{spec.name}]" if spec else "Sub-agent"
            return ToolResult(
                success=True,
                output=f"[{agent_label} completed: {state.tool_calls} tool calls, {state.iterations} iterations]\n\n{state.result}",
            )

        try:
            if self._use_queue:
                return await enqueue_command_in_lane(CommandLane.Subagent.value, do_run)
            return await do_run()
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Sub-agent error: {str(e)}")


def create_task_tool(
    llm: LLMProvider,
    tools: ToolRegistry,
    subagents: Optional[List[SubAgentSpec]] = None,
    use_queue: bool = False,
) -> Tool:
    """Factory function to create a TaskTool with the parent's LLM and tools."""
    return TaskTool(llm=llm, tools=tools, subagents=subagents, use_queue=use_queue)
