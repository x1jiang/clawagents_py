"""ComposeTool — Deterministic multi-tool pipelines without LLM in the loop.

Chains multiple tool calls programmatically, passing results between steps.
Lighter-weight than sub-agents for predictable, deterministic workflows.

Inspired by ToolUniverse's ComposeTool pattern.

Example::

    from clawagents.tools.compose import create_compose_tool

    pipeline = create_compose_tool(
        name="read_and_grep",
        description="Read a file then search for a pattern in it",
        parameters={
            "path": {"type": "string", "description": "File path", "required": True},
            "pattern": {"type": "string", "description": "Search pattern", "required": True},
        },
        steps=lambda args, call: [
            lambda prev: call("read_file", {"path": args["path"]}),
            lambda prev: call("grep", {"pattern": args["pattern"], "content": prev.output}),
        ],
        registry=registry,
    )
    registry.register(pipeline)
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional

from clawagents.tools.registry import ToolRegistry, ToolResult


CallTool = Callable[[str, Dict[str, Any]], Awaitable[ToolResult]]
PipelineStep = Callable[[Optional[ToolResult]], Awaitable[ToolResult]]
StepBuilder = Callable[[Dict[str, Any], CallTool], List[PipelineStep]]


class ComposeTool:
    """A tool that chains other tools in a deterministic pipeline."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        step_builder: StepBuilder,
        registry: ToolRegistry,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._step_builder = step_builder
        self._registry = registry

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        async def call_tool(tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
            return await self._registry.execute_tool(tool_name, tool_args)

        steps = self._step_builder(args, call_tool)
        if not steps:
            return ToolResult(success=False, output="", error="ComposeTool has no steps")

        prev: Optional[ToolResult] = None
        outputs: list[str] = []

        for i, step in enumerate(steps):
            try:
                prev = await step(prev)
            except Exception as e:
                return ToolResult(
                    success=False,
                    output="\n---\n".join(outputs),
                    error=f"Step {i + 1}/{len(steps)} failed: {e}",
                )

            if not prev.success:
                return ToolResult(
                    success=False,
                    output="\n---\n".join(outputs),
                    error=f"Step {i + 1}/{len(steps)} failed: {prev.error or 'unknown error'}",
                )

            if isinstance(prev.output, str):
                outputs.append(prev.output)

        return prev if prev else ToolResult(success=True, output="\n---\n".join(outputs))


def create_compose_tool(
    name: str,
    description: str,
    parameters: Dict[str, Dict[str, Any]],
    steps: StepBuilder,
    registry: ToolRegistry,
) -> ComposeTool:
    """Create a deterministic multi-tool pipeline."""
    return ComposeTool(
        name=name,
        description=description,
        parameters=parameters,
        step_builder=steps,
        registry=registry,
    )
