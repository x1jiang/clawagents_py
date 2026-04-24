"""Typed user context threaded through an agent run.

``RunContext[T]`` carries user-supplied state (``context``), a live
:class:`~clawagents.usage.Usage` accumulator, and the per-call tool
approval store through the agent loop. It is passed to any tool whose
``execute`` signature declares a ``run_context`` parameter, and to
class-based hooks (:class:`~clawagents.lifecycle.RunHooks`,
:class:`~clawagents.lifecycle.AgentHooks`).

Inspired by openai-agents-python's ``RunContextWrapper`` but kept
backward-compatible: existing tools that accept only ``args`` continue
to work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from clawagents.usage import Usage

TContext = TypeVar("TContext")


@dataclass
class ApprovalRecord:
    """Per-call-ID approval decision for a tool call.

    ``approved`` — True to run, False to reject.
    ``always`` — if True, the decision persists for subsequent calls to
        the same tool (keyed by tool name) in this run.
    ``reason`` — optional explanation echoed back to the model when the
        call is rejected.
    """
    approved: bool
    always: bool = False
    reason: str | None = None


@dataclass
class RunContext(Generic[TContext]):
    """Typed context wrapper passed through a run.

    Tools can declare ``async def execute(self, args, run_context)`` (or
    accept ``run_context`` as a keyword) to receive this object. The
    loop auto-detects that via signature inspection; tools that only
    accept ``args`` keep working.
    """
    context: TContext | None = None
    usage: Usage = field(default_factory=Usage)
    _approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    _always_approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    _metadata: dict[str, Any] = field(default_factory=dict)

    def approve_tool(
        self,
        call_id: str,
        *,
        always: bool = False,
        tool_name: str | None = None,
    ) -> None:
        """Record an approval for a specific tool ``call_id``.

        If ``always`` and ``tool_name`` are provided, future calls to
        the same tool in this run will be auto-approved.
        """
        rec = ApprovalRecord(approved=True, always=always)
        self._approvals[call_id] = rec
        if always and tool_name:
            self._always_approvals[tool_name] = rec

    def reject_tool(
        self,
        call_id: str,
        *,
        always: bool = False,
        tool_name: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record a rejection for a specific tool ``call_id``."""
        rec = ApprovalRecord(approved=False, always=always, reason=reason)
        self._approvals[call_id] = rec
        if always and tool_name:
            self._always_approvals[tool_name] = rec

    def is_tool_approved(
        self,
        call_id: str,
        *,
        tool_name: str | None = None,
    ) -> bool | None:
        """Return True if approved, False if rejected, None if undecided."""
        if call_id in self._approvals:
            return self._approvals[call_id].approved
        if tool_name and tool_name in self._always_approvals:
            return self._always_approvals[tool_name].approved
        return None

    def get_approval(
        self,
        call_id: str,
        *,
        tool_name: str | None = None,
    ) -> ApprovalRecord | None:
        """Return the full :class:`ApprovalRecord`, including reason, if any."""
        if call_id in self._approvals:
            return self._approvals[call_id]
        if tool_name and tool_name in self._always_approvals:
            return self._always_approvals[tool_name]
        return None
