"""Goal autopilot product API — Grok /goal-class long-horizon driver.

Wraps planner (fail-closed) → agent execute →
majority verifier → fail-open strategist.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from clawagents.config.features import is_enabled
from clawagents.goal import (
    GoalOrchestrator,
    GoalState,
    GoalTracker,
    attach_goal_to_run_context,
)
from clawagents.run_context import RunContext


ExecuteFn = Callable[[GoalState, RunContext], Awaitable[str]]
LLMComplete = Callable[[str], Awaitable[str]]


async def run_goal(
    goal: str,
    *,
    workspace: str,
    llm_complete: LLMComplete,
    execute_fn: ExecuteFn,
    run_context: RunContext | None = None,
    skeptics: int = 3,
    auto_plan: bool = True,
    max_verify_rounds: int = 3,
) -> GoalState:
    """Drive a goal through plan → execute → verify (with strategist on miss).

    ``execute_fn`` typically invokes a ClawAgent turn that uses ``update_goal``.
    The final majority verifier still runs here so completion is fail-closed.
    """
    if not is_enabled("goal_autopilot"):
        tracker = GoalTracker(workspace)
        st = tracker.start(goal)
        tracker.mark_failed("goal_autopilot feature disabled")
        return tracker.state  # type: ignore[return-value]

    ctx = run_context or RunContext()
    ctx._metadata.setdefault("workspace", workspace)
    tracker = GoalTracker(workspace)
    attach_goal_to_run_context(ctx, tracker)
    ctx._metadata["goal_llm_complete"] = llm_complete

    state = tracker.start(goal)
    orch = GoalOrchestrator(tracker, llm_complete, skeptics=skeptics)

    if auto_plan:
        await orch.plan()
    else:
        state.status = state.status  # planning pending for caller

    evidence = ""
    for _ in range(max(1, max_verify_rounds)):
        if tracker.state is None:
            break
        if tracker.state.status.value == "paused":
            break
        evidence = await execute_fn(tracker.state, ctx)
        tracker.note(str(evidence)[:2000])
        ok, st = await orch.verify(str(evidence)[:6000])
        if ok:
            return st
        # Strategist already ran inside verify on consecutive misses
    if tracker.state and tracker.state.status.value not in ("done", "failed"):
        tracker.mark_failed("verify budget exhausted")
    return tracker.state  # type: ignore[return-value]


__all__ = ["run_goal", "ExecuteFn"]
