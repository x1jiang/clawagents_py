"""enter_plan_mode / exit_plan_mode built-in tools.

Plan mode is a soft-readonly stance the model can opt into to design an
approach before mutating state. While in plan mode, the registry refuses
write-class tools (see :mod:`clawagents.permissions.mode`).

Each tool produces a tool result whose ``output`` is a system-generated
reminder describing the new mode — so the model sees its constraints in
the next observation, not in a system-prompt flag.

These tools require the typed :class:`~clawagents.run_context.RunContext`;
without a run-context they refuse cleanly so they can never silently no-op.
Mode mutations happen only via these tools — the registry never auto-flips
the bit.
"""

from __future__ import annotations

from typing import Any, Dict

from clawagents.permissions.mode import PermissionMode
from clawagents.run_context import RunContext
from clawagents.tools.registry import ToolResult


_ENTER_REMINDER = (
    "<system-reminder>\n"
    "You are now in PLAN MODE. Until you call exit_plan_mode, the registry "
    "will refuse write-class tools (write_file, edit_file, execute, ...).\n"
    "Do:\n"
    "  - Read the codebase, search, gather context.\n"
    "  - Design a concrete plan with steps and impacted files.\n"
    "  - When ready, call exit_plan_mode and resume normal operation.\n"
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
        "Exit PLAN MODE and return to DEFAULT permission mode. Write-class "
        "tools are unblocked after this call."
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
                    "exit_plan_mode requires a RunContext to mutate the "
                    "permission mode; this run does not propagate one."
                ),
            )
        run_context.permission_mode = PermissionMode.DEFAULT
        return ToolResult(success=True, output=_EXIT_REMINDER)


# ─── Public API ──────────────────────────────────────────────────────────

enter_plan_mode_tool = EnterPlanModeTool()
exit_plan_mode_tool = ExitPlanModeTool()


def create_plan_mode_tools() -> list:
    """Return the [enter_plan_mode, exit_plan_mode] tool pair."""
    return [enter_plan_mode_tool, exit_plan_mode_tool]
