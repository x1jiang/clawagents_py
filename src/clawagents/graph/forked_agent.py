"""Forked Agent Pattern (learned from Claude Code).

Provides the ability to run sandboxed sub-agents that share the parent's context
but operate with restricted tool sets and limited turn budgets. This enables:

1. Background research without polluting the main context
2. Parallel exploration of different approaches
3. Memory extraction via a forked lightweight agent

Usage:
    from clawagents.graph.forked_agent import run_forked_agent

    result = await run_forked_agent(
        fork_prompt="Analyze this error and suggest fixes",
        llm=llm,
        parent_messages=messages[:1],  # share system prompt
        allowed_tools=["read_file", "grep", "execute"],
        max_turns=5,
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def run_forked_agent(
    fork_prompt: str,
    llm: Any,
    parent_messages: list[Any] | None = None,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    max_turns: int = 5,
    context_window: int = 200_000,
    streaming: bool = False,
    on_event: Any = None,
    tools: Any = None,
) -> Any:
    """Run a sandboxed sub-agent that shares the parent's context.

    This creates a restricted execution environment using the existing
    run_agent_graph infrastructure but with tool-level isolation.

    Args:
        fork_prompt: The task for the forked agent
        llm: LLM provider instance (shared with parent for cache benefits)
        parent_messages: Optional parent messages to share (typically just system prompt)
        allowed_tools: If set, only these tools are available to the fork
        blocked_tools: If set, these tools are blocked from the fork
        max_turns: Maximum iterations for the fork
        context_window: Context window for the fork
        streaming: Whether to stream responses
        on_event: Event callback
        tools: Tool registry (can be the parent's registry or a filtered one)

    Returns:
        AgentState from the forked run
    """
    from clawagents.config.features import is_enabled
    if not is_enabled("forked_agents"):
        raise RuntimeError("Forked agents feature is not enabled. Set CLAW_FEATURE_FORKED_AGENTS=1")

    from clawagents.graph.agent_loop import run_agent_graph
    from clawagents.run_context import RunContext
    from clawagents.tools.registry import ToolRegistry

    # Create restricted tool registry if filtering is needed
    fork_registry = None
    if tools and (allowed_tools or blocked_tools):
        fork_registry = ToolRegistry()
        original_tools = tools.list() if hasattr(tools, "list") else []

        for tool in original_tools:
            if allowed_tools and tool.name not in allowed_tools:
                continue
            if blocked_tools and tool.name in blocked_tools:
                continue
            fork_registry.register(tool)
    else:
        fork_registry = tools

    # Build system prompt from parent context
    system_prompt = None
    if parent_messages and len(parent_messages) > 0:
        first_msg = parent_messages[0]
        if first_msg.role == "system":
            system_prompt = first_msg.content if isinstance(first_msg.content, str) else str(first_msg.content)

    # Suppress event logging for forked agents by default
    _noop = lambda *_a, **_kw: None

    # Forks are isolated like sub-agents: skip parent memory and lessons.
    # We do NOT bump depth — forks are typically used for memory extraction
    # and similar background tasks, not user-visible delegation, so they
    # should not consume the recursive-delegation budget. The `task` tool
    # remains the only path that increments RunContext.depth.
    #
    # Each fork also gets its own IterationBudget so a runaway research
    # fork cannot starve the parent's remaining turns.
    from clawagents.iteration_budget import IterationBudget
    fork_ctx: RunContext = RunContext(
        skip_memory=True,
        iteration_budget=IterationBudget(max(1, int(max_turns))),
    )

    state = await run_agent_graph(
        task=fork_prompt,
        llm=llm,
        tools=fork_registry,
        max_iterations=max_turns,
        context_window=context_window,
        streaming=streaming,
        on_event=on_event or _noop,
        system_prompt=system_prompt,
        run_context=fork_ctx,
        # Disable PTRL for forks (they shouldn't learn or modify lessons)
        trajectory=False,
        learn=False,
        rethink=False,
        session_end_tail=False,
    )

    return state


async def run_forked_agent_background(
    fork_prompt: str,
    llm: Any,
    **kwargs: Any,
) -> asyncio.Task:
    """Run a forked agent as a background task.

    Returns an asyncio.Task that can be awaited later for the result.
    """
    task = asyncio.create_task(
        run_forked_agent(fork_prompt=fork_prompt, llm=llm, **kwargs)
    )
    return task
