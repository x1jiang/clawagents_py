"""Goal autopilot product — Grok /goal-style planner → execute → verify → strategize.

Long-horizon completion gate when a goal is active:
  - Planner: fail-closed (must produce plan.md)
  - Worker: update_goal progress reports
  - Verifier: N skeptic LLM votes (majority must accept) — fail-closed
  - Strategist: fail-open advisory strategy.md on stall (never mutates plan)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


class GoalStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    ACTIVE = "active"
    VERIFYING = "verifying"
    STRATEGIZING = "strategizing"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


class GoalPauseReason(str, Enum):
    USER = "user"
    NO_PROGRESS = "no_progress"
    VERIFY_FAILED = "verify_failed"
    PLANNER_FAILED = "planner_failed"
    BLOCKED = "blocked"


@dataclass
class GoalState:
    id: str
    goal: str
    workspace: str
    status: GoalStatus = GoalStatus.IDLE
    pause_reason: GoalPauseReason | None = None
    pause_message: str = ""
    plan_text: str = ""
    strategy_text: str = ""
    messages: list[str] = field(default_factory=list)
    blocked_reason: str = ""
    consecutive_not_achieved: int = 0
    verify_rounds: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "workspace": self.workspace,
            "status": self.status.value,
            "pause_reason": self.pause_reason.value if self.pause_reason else None,
            "pause_message": self.pause_message,
            "plan_text": self.plan_text,
            "strategy_text": self.strategy_text,
            "messages": list(self.messages[-20:]),
            "blocked_reason": self.blocked_reason,
            "consecutive_not_achieved": self.consecutive_not_achieved,
            "verify_rounds": self.verify_rounds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


class GoalTracker:
    """In-memory + disk-backed goal state under ``.clawagents/goal/``."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / ".clawagents" / "goal"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state: GoalState | None = None
        self._load()

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.md"

    @property
    def strategy_path(self) -> Path:
        return self.root / "strategy.md"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            status = GoalStatus(raw.get("status", "idle"))
            pause = raw.get("pause_reason")
            self.state = GoalState(
                id=str(raw.get("id") or uuid.uuid4().hex[:10]),
                goal=str(raw.get("goal") or ""),
                workspace=str(raw.get("workspace") or self.workspace),
                status=status,
                pause_reason=GoalPauseReason(pause) if pause else None,
                pause_message=str(raw.get("pause_message") or ""),
                plan_text=str(raw.get("plan_text") or ""),
                strategy_text=str(raw.get("strategy_text") or ""),
                messages=list(raw.get("messages") or []),
                blocked_reason=str(raw.get("blocked_reason") or ""),
                consecutive_not_achieved=int(raw.get("consecutive_not_achieved") or 0),
                verify_rounds=int(raw.get("verify_rounds") or 0),
                created_at=float(raw.get("created_at") or time.time()),
                updated_at=float(raw.get("updated_at") or time.time()),
                metadata=dict(raw.get("metadata") or {}),
            )
            if self.plan_path.exists() and not self.state.plan_text:
                self.state.plan_text = self.plan_path.read_text(encoding="utf-8")
            if self.strategy_path.exists() and not self.state.strategy_text:
                self.state.strategy_text = self.strategy_path.read_text(encoding="utf-8")
        except Exception:
            self.state = None

    def save(self) -> None:
        if self.state is None:
            return
        self.state.touch()
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2), encoding="utf-8"
        )
        if self.state.plan_text:
            self.plan_path.write_text(self.state.plan_text, encoding="utf-8")
        if self.state.strategy_text:
            self.strategy_path.write_text(self.state.strategy_text, encoding="utf-8")

    def start(self, goal: str) -> GoalState:
        self.state = GoalState(
            id=uuid.uuid4().hex[:10],
            goal=goal.strip(),
            workspace=str(self.workspace),
            status=GoalStatus.PLANNING,
        )
        self.save()
        return self.state

    def pause(self, reason: GoalPauseReason, message: str = "") -> GoalState | None:
        if self.state is None or self.state.status not in (
            GoalStatus.ACTIVE,
            GoalStatus.VERIFYING,
            GoalStatus.STRATEGIZING,
            GoalStatus.PLANNING,
        ):
            return self.state
        self.state.status = GoalStatus.PAUSED
        self.state.pause_reason = reason
        self.state.pause_message = message
        self.save()
        return self.state

    def resume(self) -> GoalState | None:
        if self.state is None or self.state.status != GoalStatus.PAUSED:
            return self.state
        self.state.status = GoalStatus.ACTIVE
        self.state.pause_reason = None
        self.state.pause_message = ""
        self.state.consecutive_not_achieved = 0
        self.save()
        return self.state

    def note(self, message: str) -> None:
        if self.state is None:
            return
        self.state.messages.append(message[:2000])
        self.save()

    def set_plan(self, plan_text: str) -> None:
        if self.state is None:
            return
        self.state.plan_text = plan_text.strip()
        self.state.status = GoalStatus.ACTIVE
        self.save()

    def set_strategy(self, strategy_text: str) -> None:
        if self.state is None:
            return
        self.state.strategy_text = strategy_text.strip()
        self.save()

    def mark_done(self) -> None:
        if self.state is None:
            return
        self.state.status = GoalStatus.DONE
        self.save()

    def mark_failed(self, message: str = "") -> None:
        if self.state is None:
            return
        self.state.status = GoalStatus.FAILED
        if message:
            self.state.messages.append(message[:2000])
        self.save()

    def is_active(self) -> bool:
        return self.state is not None and self.state.status in (
            GoalStatus.ACTIVE,
            GoalStatus.VERIFYING,
            GoalStatus.STRATEGIZING,
            GoalStatus.PLANNING,
        )


LLMComplete = Callable[[str], Awaitable[str]]


async def run_planner(llm: LLMComplete, goal: str, *, workspace: str) -> str:
    """Fail-closed planner: empty/short plan raises."""
    prompt = (
        "You are the GOAL PLANNER. Produce a concrete verifier contract as markdown.\n"
        f"Workspace: {workspace}\n"
        f"Goal:\n{goal}\n\n"
        "Write plan.md content with:\n"
        "1. Success criteria (checklist)\n"
        "2. Ordered steps\n"
        "3. Out of scope\n"
        "4. Verification commands or checks\n"
        "Reply with ONLY the markdown plan body."
    )
    text = (await llm(prompt)).strip()
    lower = text.casefold()
    if len(text) < 80:
        raise RuntimeError("planner produced empty/short plan (fail-closed)")
    if "success" not in lower and "criteria" not in lower and "checklist" not in lower:
        raise RuntimeError("planner plan missing success criteria (fail-closed)")
    return text


async def run_strategist(
    llm: LLMComplete,
    *,
    goal: str,
    plan_text: str,
    recent: str,
) -> str:
    """Fail-open strategist: advisory only; never mutates plan."""
    prompt = (
        "You are the GOAL STRATEGIST (advisory, fail-open).\n"
        "The worker stalled verifying the goal. Suggest a short strategy note.\n"
        "Do NOT rewrite the plan. Do NOT claim the goal is done.\n\n"
        f"Goal:\n{goal}\n\nPlan:\n{plan_text[:4000]}\n\n"
        f"Recent progress:\n{recent[:3000]}\n\n"
        "Reply with strategy.md markdown only."
    )
    try:
        text = (await llm(prompt)).strip()
        return text or "No additional strategy; continue executing the plan carefully."
    except Exception as exc:
        return f"Strategist unavailable ({exc}); continue with existing plan."


_ACHIEVED_RE = re.compile(r"\b(achieved|completed|done|pass|yes)\b", re.I)
_NOT_RE = re.compile(r"\b(not\s+achieved|incomplete|fail|no|blocked)\b", re.I)


def _parse_verdict(text: str) -> bool:
    """Return True if skeptic says achieved."""
    lower = text.casefold()
    # Prefer explicit JSON
    try:
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            obj = json.loads(m.group(0))
            if "achieved" in obj:
                return bool(obj["achieved"])
            if "completed" in obj:
                return bool(obj["completed"])
    except Exception:
        pass
    if _NOT_RE.search(lower) and not _ACHIEVED_RE.search(lower):
        return False
    if "achieved\": true" in lower or '"achieved": true' in lower:
        return True
    if "achieved\": false" in lower or '"achieved": false' in lower:
        return False
    # Last resort: achieved keyword without not
    return bool(_ACHIEVED_RE.search(lower)) and not _NOT_RE.search(lower)


async def run_verifier(
    llm: LLMComplete,
    *,
    goal: str,
    plan_text: str,
    evidence: str,
    skeptics: int = 3,
) -> tuple[bool, list[str]]:
    """Majority skeptic vote. Fail-closed on parse/transport for individual votes
    counts as not-achieved. Overall requires majority True.
    """
    n = max(1, min(7, skeptics))
    votes: list[str] = []
    yes = 0
    for i in range(n):
        prompt = (
            f"You are GOAL VERIFIER skeptic #{i + 1}/{n}.\n"
            "Decide if the goal success criteria are met based on evidence.\n"
            'Reply with JSON only: {"achieved": true|false, "reason": "..."}\n\n'
            f"Goal:\n{goal}\n\nSuccess contract (plan):\n{plan_text[:5000]}\n\n"
            f"Evidence:\n{evidence[:6000]}\n"
        )
        try:
            raw = (await llm(prompt)).strip()
            votes.append(raw[:1500])
            if _parse_verdict(raw):
                yes += 1
        except Exception as exc:
            votes.append(f"skeptic_error: {exc}")
            # fail-closed vote = not achieved
    return yes * 2 > n, votes


@dataclass
class GoalOrchestrator:
    tracker: GoalTracker
    llm: LLMComplete
    skeptics: int = 3
    strategize_after: int = 2

    async def plan(self) -> GoalState:
        st = self.tracker.state
        if st is None:
            raise RuntimeError("no goal started")
        st.status = GoalStatus.PLANNING
        self.tracker.save()
        try:
            plan = await run_planner(self.llm, st.goal, workspace=st.workspace)
        except Exception as exc:
            self.tracker.pause(GoalPauseReason.PLANNER_FAILED, str(exc))
            raise
        self.tracker.set_plan(plan)
        return self.tracker.state  # type: ignore[return-value]

    async def verify(self, evidence: str) -> tuple[bool, GoalState]:
        st = self.tracker.state
        if st is None:
            raise RuntimeError("no goal started")
        st.status = GoalStatus.VERIFYING
        st.verify_rounds += 1
        self.tracker.save()
        ok, votes = await run_verifier(
            self.llm,
            goal=st.goal,
            plan_text=st.plan_text,
            evidence=evidence,
            skeptics=self.skeptics,
        )
        st.metadata["last_verify_votes"] = votes
        if ok:
            st.consecutive_not_achieved = 0
            self.tracker.mark_done()
            return True, st
        st.consecutive_not_achieved += 1
        self.tracker.save()
        if st.consecutive_not_achieved >= self.strategize_after:
            st.status = GoalStatus.STRATEGIZING
            self.tracker.save()
            strategy = await run_strategist(
                self.llm,
                goal=st.goal,
                plan_text=st.plan_text,
                recent=evidence,
            )
            self.tracker.set_strategy(strategy)
            st.status = GoalStatus.ACTIVE
            self.tracker.save()
        else:
            st.status = GoalStatus.ACTIVE
            self.tracker.save()
        return False, st


def goal_system_reminder(state: GoalState | None) -> str:
    if state is None or state.status in (GoalStatus.IDLE, GoalStatus.DONE, GoalStatus.FAILED):
        return ""
    parts = [
        "## Active Goal",
        f"Status: {state.status.value}",
        f"Goal: {state.goal}",
    ]
    if state.plan_text:
        parts.append("Plan (contract):\n" + state.plan_text[:3000])
    if state.strategy_text:
        parts.append("Strategy note:\n" + state.strategy_text[:1500])
    parts.append(
        "Use `update_goal` to report progress. Set completed=true only when "
        "success criteria are met — a verifier panel must confirm."
    )
    return "\n".join(parts)


def attach_goal_to_run_context(run_context: Any, tracker: GoalTracker) -> None:
    if run_context is None:
        return
    meta = getattr(run_context, "_metadata", None)
    if isinstance(meta, dict):
        meta["goal_tracker"] = tracker


def get_goal_tracker(run_context: Any) -> GoalTracker | None:
    if run_context is None:
        return None
    meta = getattr(run_context, "_metadata", None)
    if isinstance(meta, dict):
        t = meta.get("goal_tracker")
        if isinstance(t, GoalTracker):
            return t
    return None


__all__ = [
    "GoalStatus",
    "GoalPauseReason",
    "GoalState",
    "GoalTracker",
    "GoalOrchestrator",
    "run_planner",
    "run_strategist",
    "run_verifier",
    "goal_system_reminder",
    "attach_goal_to_run_context",
    "get_goal_tracker",
    "run_goal",
]


def __getattr__(name: str):
    if name == "run_goal":
        from clawagents.goal.product import run_goal

        return run_goal
    raise AttributeError(name)
