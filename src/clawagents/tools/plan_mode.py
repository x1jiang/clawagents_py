"""enter_plan_mode / exit_plan_mode built-in tools.

Plan mode is a soft-readonly stance the model can opt into to design an
approach before mutating state. While in plan mode, the registry refuses
write-class tools (see :mod:`clawagents.permissions.mode`).

``exit_plan_mode`` optionally waits for a host approval callback before
unlocking writes (Grok Build parity). With no callback registered it
auto-approves for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from clawagents.config.features import is_enabled
from clawagents.permissions.mode import PermissionMode
from clawagents.permissions.plan_approval import (
    PLAN_APPROVAL_META_KEY,
    PlanApprovalAction,
    PlanApprovalCallback,
    await_plan_approval,
    load_plan_text,
)
from clawagents.run_context import RunContext
from clawagents.tools.registry import ToolResult


_ENTER_REMINDER = (
    "<system-reminder>\n"
    "You are now in PLAN MODE. Until you call exit_plan_mode, the registry "
    "will refuse write-class tools (write_file, edit_file, execute, ...).\n"
    "Do:\n"
    "  - Read the codebase, search, gather context.\n"
    "  - Design a concrete plan with steps and impacted files.\n"
    "  - Write the plan via write_plan, then call exit_plan_mode when ready.\n"
    "Don't:\n"
    "  - Write or edit files.\n"
    "  - Run shell commands that modify state.\n"
    "</system-reminder>"
)

_EXIT_REMINDER = (
    "<system-reminder>\n"
    "You have exited plan mode. Permission mode is back to DEFAULT. "
    "Write-class tools are unblocked.\n"
    "If you drafted a plan, call write_plan(content=...) so Act mode can load "
    ".clawagents/plan.md on the next turns.\n"
    "</system-reminder>"
)

_REJECT_REMINDER = (
    "<system-reminder>\n"
    "Plan exit was rejected. You remain in PLAN MODE. Revise the plan "
    "(write_plan) and call exit_plan_mode again when ready.\n"
    "</system-reminder>"
)

_CHANGES_REMINDER = (
    "<system-reminder>\n"
    "The host requested changes to the plan. You remain in PLAN MODE. "
    "Incorporate the feedback, update write_plan, then exit_plan_mode again.\n"
    "</system-reminder>"
)


class EnterPlanModeTool:
    name = "enter_plan_mode"
    description = (
        "Enter PLAN MODE — a read-only exploration phase. While in plan mode, "
        "write-class tools (write_file, edit_file, execute, ...) are refused. "
        "Use this before non-trivial implementation tasks to design an approach "
        "before touching files. Call exit_plan_mode when ready to implement."
    )
    parameters: Dict[str, Dict[str, Any]] = {}

    async def execute(
        self,
        args: Dict[str, Any],
        run_context: RunContext | None = None,
    ) -> ToolResult:
        if run_context is None:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "enter_plan_mode requires a RunContext to mutate the "
                    "permission mode; this run does not propagate one."
                ),
            )
        run_context.permission_mode = PermissionMode.PLAN
        return ToolResult(success=True, output=_ENTER_REMINDER)


class ExitPlanModeTool:
    name = "exit_plan_mode"
    description = (
        "Exit PLAN MODE and return to DEFAULT permission mode. When a host "
        "approval callback is registered, the plan is presented for Approve / "
        "Request changes / Reject before writes unlock."
    )
    parameters: Dict[str, Dict[str, Any]] = {}

    def __init__(self, on_exit_plan_mode: Optional[PlanApprovalCallback] = None):
        self._on_exit_plan_mode = on_exit_plan_mode

    async def execute(
        self,
        args: Dict[str, Any],
        run_context: RunContext | None = None,
    ) -> ToolResult:
        if run_context is None:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "exit_plan_mode requires a RunContext to mutate the "
                    "permission mode; this run does not propagate one."
                ),
            )

        if run_context.permission_mode != PermissionMode.PLAN:
            run_context.permission_mode = PermissionMode.DEFAULT
            return ToolResult(success=True, output=_EXIT_REMINDER)

        workspace = None
        if isinstance(run_context._metadata.get("workspace"), str):
            workspace = run_context._metadata["workspace"]
        plan_text = load_plan_text(run_context, workspace=workspace)

        callback = self._on_exit_plan_mode
        if callback is None and is_enabled("plan_approval"):
            raw = run_context._metadata.get(PLAN_APPROVAL_META_KEY)
            if callable(raw):
                callback = raw  # type: ignore[assignment]

        # Fire lifecycle hooks (best-effort) before awaiting host decision.
        for h in run_context._metadata.get("hooks") or []:
            fn = getattr(h, "on_exit_plan_mode", None)
            if fn is None:
                continue
            try:
                result = fn(run_context, plan_text)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass

        decision = await await_plan_approval(
            plan_text,
            run_context,
            callback=callback,
        )

        if decision.action == PlanApprovalAction.APPROVE:
            run_context.permission_mode = PermissionMode.DEFAULT
            note = _EXIT_REMINDER
            if decision.comment:
                note += f"\nHost comment: {decision.comment}"
            return ToolResult(success=True, output=note)

        # Stay in PLAN for reject / request_changes
        run_context.permission_mode = PermissionMode.PLAN
        if decision.action == PlanApprovalAction.REQUEST_CHANGES:
            body = _CHANGES_REMINDER
            if decision.comment:
                body += f"\nFeedback: {decision.comment}"
            return ToolResult(success=False, output=body, error="plan_changes_requested")

        body = _REJECT_REMINDER
        if decision.comment:
            body += f"\nReason: {decision.comment}"
        return ToolResult(success=False, output=body, error="plan_rejected")


# ─── Public API ──────────────────────────────────────────────────────────

enter_plan_mode_tool = EnterPlanModeTool()
exit_plan_mode_tool = ExitPlanModeTool()


def create_plan_mode_tools(
    on_exit_plan_mode: Optional[PlanApprovalCallback] = None,
) -> list:
    """Return the [enter_plan_mode, exit_plan_mode] tool pair."""
    return [
        EnterPlanModeTool(),
        ExitPlanModeTool(on_exit_plan_mode=on_exit_plan_mode),
    ]
