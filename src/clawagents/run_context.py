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

import asyncio
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from clawagents.iteration_budget import IterationBudget
from clawagents.permissions.mode import PermissionMode
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


# Maximum nesting depth for sub-agent delegation. Mirrors Hermes' policy: a
# subagent (depth=1) may not itself spawn another subagent (depth=2 is the
# hard cap; the ``task`` tool refuses any new spawn when depth >= MAX_SUBAGENT_DEPTH).
# This bounds the worst-case token / iteration / time blowup of recursive
# delegation and keeps cost predictable.
MAX_SUBAGENT_DEPTH: int = 2


@dataclass
class RunContext(Generic[TContext]):
    """Typed context wrapper passed through a run.

    Tools can declare ``async def execute(self, args, run_context)`` (or
    accept ``run_context`` as a keyword) to receive this object. The
    loop auto-detects that via signature inspection; tools that only
    accept ``args`` keep working.

    Attributes:
        context: User-supplied state passed in at run start.
        usage: Live token-usage accumulator.
        permission_mode: Active permission mode (``DEFAULT``/``PLAN``/…).
        depth: Nesting depth for sub-agent delegation. ``0`` for the
            top-level / user-facing run, ``1`` for a first-level subagent,
            ``2`` for a sub-subagent (capped at :data:`MAX_SUBAGENT_DEPTH`).
            The ``task`` tool refuses to spawn when ``depth >= MAX_SUBAGENT_DEPTH``.
        skip_memory: When ``True``, the agent loop and any memory loaders
            skip reading the parent's memory directory, lessons, and
            persisted skill state. Sub-agent runs default to ``True`` so
            they remain isolated from parent context.
        iteration_budget: Optional :class:`~clawagents.iteration_budget.IterationBudget`
            attached to this run. When set, the agent loop consumes one
            unit per round and stops when the budget is exhausted, even
            if ``max_iterations`` would still allow more rounds. Each
            subagent gets a *fresh* budget so a runaway delegate cannot
            starve the parent run.
    """
    context: TContext | None = None
    usage: Usage = field(default_factory=Usage)
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    depth: int = 0
    skip_memory: bool = False
    iteration_budget: IterationBudget | None = None
    _approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    _always_approvals: dict[str, ApprovalRecord] = field(default_factory=dict)
    _metadata: dict[str, Any] = field(default_factory=dict)
    _budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    async def ensure_iteration_budget(self, size: int) -> IterationBudget:
        """Lazily attach an :class:`IterationBudget` if none is set yet.

        Safe under concurrent access: the first caller wins, every other
        caller observes the same budget. Returns the (now non-None)
        budget.
        """
        if self.iteration_budget is not None:
            return self.iteration_budget
        async with self._budget_lock:
            if self.iteration_budget is None:
                self.iteration_budget = IterationBudget(max(0, size))
            return self.iteration_budget

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
