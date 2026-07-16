"""Human plan-approval gate for exit_plan_mode (Grok Build parity).

When a host registers ``on_exit_plan_mode``, exiting plan mode waits for an
explicit Approve / Request changes / Reject decision before write-class tools
unlock. With no callback registered the gate auto-approves so library and
headless tests keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional


class PlanApprovalAction(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


@dataclass(frozen=True)
class PlanApprovalDecision:
    action: PlanApprovalAction
    comment: str = ""

    @property
    def approved(self) -> bool:
        return self.action == PlanApprovalAction.APPROVE


PlanApprovalCallback = Callable[
    [str, Any],
    Awaitable[PlanApprovalDecision | bool | str],
]

PLAN_APPROVAL_META_KEY = "on_exit_plan_mode"
PLAN_TEXT_META_KEY = "pending_plan_text"


def normalize_plan_decision(value: Any) -> PlanApprovalDecision:
    """Coerce callback return values into a PlanApprovalDecision."""
    if isinstance(value, PlanApprovalDecision):
        return value
    if isinstance(value, bool):
        return PlanApprovalDecision(
            PlanApprovalAction.APPROVE if value else PlanApprovalAction.REJECT
        )
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key in ("approve", "approved", "a", "yes", "ok", "true", "1"):
            return PlanApprovalDecision(PlanApprovalAction.APPROVE)
        if key in ("request_changes", "changes", "revise", "s"):
            return PlanApprovalDecision(PlanApprovalAction.REQUEST_CHANGES)
        if key in ("reject", "quit", "q", "no", "false", "0"):
            return PlanApprovalDecision(PlanApprovalAction.REJECT)
        return PlanApprovalDecision(PlanApprovalAction.REQUEST_CHANGES, comment=value)
    return PlanApprovalDecision(PlanApprovalAction.REJECT, comment="invalid decision")


async def await_plan_approval(
    plan_text: str,
    run_context: Any,
    *,
    callback: Optional[PlanApprovalCallback] = None,
) -> PlanApprovalDecision:
    """Resolve an exit-plan decision.

    Precedence: explicit ``callback`` → ``run_context._metadata[on_exit_plan_mode]``
    → auto-approve.
    """
    cb = callback
    if cb is None:
        raw = run_context._metadata.get(PLAN_APPROVAL_META_KEY)
        if callable(raw):
            cb = raw  # type: ignore[assignment]
    if cb is None:
        return PlanApprovalDecision(PlanApprovalAction.APPROVE)

    result = await cb(plan_text, run_context)
    return normalize_plan_decision(result)


def load_plan_text(run_context: Any | None, workspace: str | None = None) -> str:
    """Best-effort plan body for the approval UI."""
    if run_context is not None:
        pending = run_context._metadata.get(PLAN_TEXT_META_KEY)
        if isinstance(pending, str) and pending.strip():
            return pending
    try:
        from clawagents.tools.context_tools import load_plan_preamble

        text = load_plan_preamble(workspace=workspace) if workspace else load_plan_preamble()
        return text or ""
    except Exception:
        return ""


__all__ = [
    "PlanApprovalAction",
    "PlanApprovalDecision",
    "PlanApprovalCallback",
    "PLAN_APPROVAL_META_KEY",
    "PLAN_TEXT_META_KEY",
    "normalize_plan_decision",
    "await_plan_approval",
    "load_plan_text",
]
