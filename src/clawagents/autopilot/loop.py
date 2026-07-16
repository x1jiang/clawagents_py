"""Autopilot plan → execute → verify loop (Grok /goal inspired)."""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Optional

from clawagents.autopilot import (
    AutopilotPhase,
    AutopilotRegistry,
    AutopilotTask,
    DEFAULT_AUTOPILOT_REGISTRY,
)
from clawagents.config.features import is_enabled
from clawagents.permissions.plan_approval import (
    PlanApprovalAction,
    PlanApprovalCallback,
    PlanApprovalDecision,
    await_plan_approval,
)
from clawagents.run_context import RunContext
from clawagents.tools.auto_verify import run_verify


PlanFn = Callable[[AutopilotTask], Awaitable[list[str] | str]]
ExecuteFn = Callable[[AutopilotTask], Awaitable[str]]
VerifyFn = Callable[[AutopilotTask], Awaitable[str]]


async def run_autopilot(
    goal: str,
    *,
    workspace: str,
    plan_fn: PlanFn,
    execute_fn: ExecuteFn,
    verify_fn: VerifyFn | None = None,
    approve_plan: PlanApprovalCallback | None = None,
    registry: AutopilotRegistry | None = None,
    task_id: str | None = None,
    auto_approve: bool = False,
) -> AutopilotTask:
    """Drive a goal through PLANNING → (approve) → EXECUTING → VERIFYING → DONE.

    For the full Grok /goal product (majority verifier + strategist + disk
    plan.md), prefer ``clawagents.goal.run_goal`` / ``create_claw_agent(goal_mode=True)``.
    This helper remains the thin library loop for custom plan/execute hooks.

    Callers supply ``plan_fn`` / ``execute_fn`` (typically wrapping an agent).
    ``approve_plan`` gates the transition out of planning unless
    ``auto_approve=True``.
    """
    if not is_enabled("autopilot_loop"):
        task = AutopilotTask(
            id=task_id or uuid.uuid4().hex[:10],
            goal=goal,
            workspace=workspace,
            phase=AutopilotPhase.FAILED,
        )
        task.notes.append("autopilot_loop feature disabled")
        return task

    reg = registry or DEFAULT_AUTOPILOT_REGISTRY
    task = AutopilotTask(
        id=task_id or uuid.uuid4().hex[:10],
        goal=goal,
        workspace=workspace,
        phase=AutopilotPhase.PLANNING,
    )
    reg.register(task.id, lambda t: _noop(t))  # discoverability

    try:
        plan_result = await plan_fn(task)
        if isinstance(plan_result, str):
            task.plan = [line.strip() for line in plan_result.splitlines() if line.strip()]
            task.metadata["plan_text"] = plan_result
        else:
            task.plan = [str(x) for x in plan_result]
            task.metadata["plan_text"] = "\n".join(task.plan)

        ctx = RunContext()
        ctx._metadata["workspace"] = workspace
        ctx._metadata["pending_plan_text"] = task.metadata["plan_text"]

        if auto_approve or approve_plan is None:
            decision = PlanApprovalDecision(PlanApprovalAction.APPROVE)
        else:
            decision = await await_plan_approval(
                task.metadata["plan_text"],
                ctx,
                callback=approve_plan,
            )

        if not decision.approved:
            task.phase = AutopilotPhase.FAILED
            task.notes.append(f"plan not approved: {decision.action.value}")
            if decision.comment:
                task.notes.append(decision.comment)
            return task

        task.phase = AutopilotPhase.EXECUTING
        exec_out = await execute_fn(task)
        task.notes.append(str(exec_out)[:4000])
        task.metadata["execute_output"] = str(exec_out)

        task.phase = AutopilotPhase.VERIFYING
        verifier = verify_fn or _default_verify
        verify_out = await verifier(task)
        task.metadata["verify_output"] = verify_out
        if verify_out:
            task.notes.append(verify_out[:4000])
            # Heuristic failure: non-empty verify with "exit " failures
            if "→ exit " in verify_out and "→ ok" not in verify_out.split("→ exit ")[0][-20:]:
                # soft: still DONE but flag
                task.metadata["verify_failed"] = True

        task.phase = AutopilotPhase.DONE
        return task
    except Exception as exc:
        task.phase = AutopilotPhase.FAILED
        task.notes.append(f"error: {exc}")
        return task


async def _default_verify(task: AutopilotTask) -> str:
    return run_verify(task.workspace)


async def _noop(task: AutopilotTask) -> dict[str, Any]:
    return {"id": task.id, "phase": task.phase.value}


__all__ = ["run_autopilot"]
