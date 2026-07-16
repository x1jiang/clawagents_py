"""Goal tools: start_goal / update_goal / pause_goal / resume_goal / goal_status."""

from __future__ import annotations

import os
from typing import Any

from clawagents.config.features import is_enabled
from clawagents.goal import (
    GoalOrchestrator,
    GoalPauseReason,
    GoalTracker,
    attach_goal_to_run_context,
    get_goal_tracker,
)
from clawagents.tools.registry import Tool, ToolResult


def _workspace(run_context: Any) -> str:
    if run_context is not None:
        meta = getattr(run_context, "_metadata", None)
        if isinstance(meta, dict) and isinstance(meta.get("workspace"), str):
            return meta["workspace"]
    return os.getcwd()


def _tracker(run_context: Any) -> GoalTracker:
    existing = get_goal_tracker(run_context)
    if existing is not None:
        return existing
    tracker = GoalTracker(_workspace(run_context))
    attach_goal_to_run_context(run_context, tracker)
    return tracker


def _llm_from_context(run_context: Any):
    """Best-effort LLM complete callable from run metadata."""

    async def _fallback(prompt: str) -> str:
        raise RuntimeError("no LLM bound for goal roles")

    if run_context is None:
        return _fallback
    meta = getattr(run_context, "_metadata", None)
    if not isinstance(meta, dict):
        return _fallback
    fn = meta.get("goal_llm_complete")
    if callable(fn):
        return fn
    return _fallback


class StartGoalTool:
    name = "start_goal"
    description = (
        "Start a long-horizon GOAL with planner→execute→verifier→strategist "
        "orchestration (Grok /goal style). Prefer this over claiming a multi-step "
        "project is done in one turn. Writes .clawagents/goal/plan.md."
    )
    parameters = {
        "goal": {
            "type": "string",
            "description": "Clear success-oriented goal statement",
            "required": True,
        },
        "auto_plan": {
            "type": "boolean",
            "description": "Run planner immediately (default true)",
            "required": False,
        },
    }

    async def execute(self, args: dict[str, Any], run_context: Any = None) -> ToolResult:
        if not is_enabled("goal_autopilot"):
            return ToolResult(success=False, output="", error="goal_autopilot feature disabled")
        goal = str(args.get("goal") or "").strip()
        if not goal:
            return ToolResult(success=False, output="", error="goal required")
        tracker = _tracker(run_context)
        state = tracker.start(goal)
        auto_plan = args.get("auto_plan", True)
        if auto_plan in (False, "false", "0", 0):
            return ToolResult(
                success=True,
                output=f"Goal {state.id} started (planning pending).\n{goal}",
            )
        orch = GoalOrchestrator(tracker, _llm_from_context(run_context))
        try:
            await orch.plan()
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Planner fail-closed: {exc}",
            )
        plan_preview = (tracker.state.plan_text if tracker.state else "")[:2000]
        return ToolResult(
            success=True,
            output=(
                f"Goal {state.id} ACTIVE.\nPlan written to .clawagents/goal/plan.md\n\n"
                f"{plan_preview}"
            ),
        )


class UpdateGoalTool:
    name = "update_goal"
    description = (
        "Report goal progress. Set completed=true only when success criteria are met; "
        "a verifier panel must confirm before the run can finish."
    )
    parameters = {
        "message": {
            "type": "string",
            "description": "Progress note / evidence summary",
            "required": False,
        },
        "completed": {
            "type": "boolean",
            "description": "Claim the goal is complete (triggers verifier)",
            "required": False,
        },
        "blocked_reason": {
            "type": "string",
            "description": "If blocked, why",
            "required": False,
        },
    }

    async def execute(self, args: dict[str, Any], run_context: Any = None) -> ToolResult:
        if not is_enabled("goal_autopilot"):
            return ToolResult(success=False, output="", error="goal_autopilot feature disabled")
        tracker = _tracker(run_context)
        if tracker.state is None:
            return ToolResult(success=False, output="", error="no active goal — call start_goal")
        msg = str(args.get("message") or "").strip()
        if msg:
            tracker.note(msg)
        blocked = str(args.get("blocked_reason") or "").strip()
        if blocked:
            tracker.state.blocked_reason = blocked
            tracker.pause(GoalPauseReason.BLOCKED, blocked)
            return ToolResult(success=True, output=f"Goal paused (blocked): {blocked}")

        completed = args.get("completed") in (True, "true", "1", 1)
        if not completed:
            return ToolResult(
                success=True,
                output=f"Progress noted. Status={tracker.state.status.value}",
            )

        evidence = msg or "\n".join(tracker.state.messages[-8:])
        orch = GoalOrchestrator(tracker, _llm_from_context(run_context))
        ok, state = await orch.verify(evidence)
        if ok:
            return ToolResult(
                success=True,
                output="Verifier majority ACCEPTED. Goal DONE.",
            )
        votes = state.metadata.get("last_verify_votes") or []
        preview = "\n---\n".join(str(v)[:400] for v in votes[:3])
        strategy = (
            f"\n\nStrategy note updated:\n{state.strategy_text[:1000]}"
            if state.strategy_text
            else ""
        )
        return ToolResult(
            success=True,
            output=(
                "Verifier majority REJECTED completion. Continue working the plan."
                f"{strategy}\n\nVotes:\n{preview}"
            ),
        )


class PauseGoalTool:
    name = "pause_goal"
    description = "Pause the active goal (user or agent)."
    parameters = {
        "message": {"type": "string", "description": "Pause reason", "required": False},
    }

    async def execute(self, args: dict[str, Any], run_context: Any = None) -> ToolResult:
        tracker = _tracker(run_context)
        if tracker.state is None:
            return ToolResult(success=False, output="", error="no active goal")
        tracker.pause(GoalPauseReason.USER, str(args.get("message") or ""))
        return ToolResult(success=True, output="Goal paused.")


class ResumeGoalTool:
    name = "resume_goal"
    description = "Resume a paused goal."
    parameters = {}

    async def execute(self, args: dict[str, Any], run_context: Any = None) -> ToolResult:
        tracker = _tracker(run_context)
        if tracker.state is None:
            return ToolResult(success=False, output="", error="no active goal")
        tracker.resume()
        return ToolResult(
            success=True,
            output=f"Goal resumed. Status={tracker.state.status.value if tracker.state else '?'}",
        )


class GoalStatusTool:
    name = "goal_status"
    description = "Show active goal status, plan path, and recent notes."
    parameters = {}

    async def execute(self, args: dict[str, Any], run_context: Any = None) -> ToolResult:
        tracker = _tracker(run_context)
        if tracker.state is None:
            return ToolResult(success=True, output="No goal started.")
        import json

        return ToolResult(
            success=True,
            output=json.dumps(tracker.state.to_dict(), indent=2),
        )


def create_goal_tools() -> list[Tool]:
    return [
        StartGoalTool(),
        UpdateGoalTool(),
        PauseGoalTool(),
        ResumeGoalTool(),
        GoalStatusTool(),
    ]


__all__ = ["create_goal_tools"]
