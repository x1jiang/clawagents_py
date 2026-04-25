"""Per-agent iteration budget (Hermes-style).

Each agent (parent or subagent) gets its own :class:`IterationBudget`. The
parent's budget is capped at ``max_iterations`` (default ``200`` for
clawagents). Each subagent gets an *independent* budget capped at
``delegation.max_iterations`` (default ``50``) — this means total iterations
across parent + subagents can exceed the parent's cap.

Why a budget rather than a plain counter?
  * Subagents must not silently steal turns from the parent. Giving each
    delegate its own budget ensures that a runaway subagent cannot
    starve the top-level conversation.
  * Some "free" iterations (e.g., ``execute_code`` programmatic batches,
    MCP listing tools that just return schemas) shouldn't eat the user's
    budget. :meth:`refund` lets the loop give an iteration back without
    racing the remaining counter.
  * It's thread-safe so concurrent subagents launched from a parent run
    don't trample each other's accounting.

Mirrors Hermes' ``IterationBudget`` (``run_agent.py``) and the contract
documented in ``hermes-agent-main/website/docs/developer-guide/agent-loop.md``.
"""

from __future__ import annotations

import threading


__all__ = [
    "DEFAULT_DELEGATION_MAX_ITERATIONS",
    "IterationBudget",
]


# Default budget for a delegated subagent. Mirrors Hermes' ``delegation.max_iterations``
# default of 50 so users that import this constant aren't surprised by a
# different ceiling. Override per-call via the ``task`` tool's
# ``max_iterations`` argument or per-agent via ``delegation.max_iterations``.
DEFAULT_DELEGATION_MAX_ITERATIONS = 50


class IterationBudget:
    """Thread-safe iteration counter for a single agent run.

    The budget is consumed once per tool-calling round in the agent loop.
    Long agent runs that delegate to subagents create a *fresh* budget
    for each child; the parent's budget is unaffected by subagent
    iterations.

    Example:
        >>> budget = IterationBudget(max_total=10)
        >>> while budget.remaining > 0:
        ...     ok = budget.consume()
        ...     if not ok: break
        ...     # do one round of tool calls
    """

    __slots__ = ("max_total", "_used", "_lock")

    def __init__(self, max_total: int):
        if max_total < 0:
            raise ValueError(f"max_total must be >= 0, got {max_total}")
        self.max_total = int(max_total)
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration. Returns ``True`` if allowed.

        Returns ``False`` once the budget is exhausted. Callers should treat
        the first ``False`` as a signal to stop the loop and return whatever
        partial output exists, mirroring how Hermes' agent surfaces a
        "max_iterations reached" outcome to the trajectory recorder.
        """
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration.

        Used for "free" turns the user shouldn't pay for: e.g., automatic
        ``execute_code`` continuation rounds, internal MCP-listing tool
        calls, or any synthetic iteration the loop inserted without the
        model actually making a tool-calling decision.
        """
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)

    def __repr__(self) -> str:
        return (
            f"IterationBudget(used={self._used}, max_total={self.max_total}, "
            f"remaining={max(0, self.max_total - self._used)})"
        )
