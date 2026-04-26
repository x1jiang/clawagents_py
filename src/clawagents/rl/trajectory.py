"""Normalized trajectory data model for RL fine-tuning.

A :class:`Trajectory` is a serialisable record of one agent run, in a
shape that's friendly to TRL / Atropos / SLIME training pipelines.
Each :class:`TrajectoryStep` corresponds to a single turn (system,
user, assistant, or tool message). The resulting structure round-trips
losslessly through JSON and converts cleanly to ChatML.

This is deliberately separate from :class:`clawagents.trajectory.TurnRecord`
— the latter records observability data (durations, scores, redacted
previews); :class:`Trajectory` records the *training-ready* prompt /
completion / reward triples.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Literal


TrajectoryRole = Literal["system", "user", "assistant", "tool"]
"""Roles supported by ChatML and most RL trainers."""


@dataclass
class ToolCall:
    """A single tool invocation captured during a step.

    Mirrors the OpenAI / ChatML tool_call shape so downstream
    converters don't have to reshape the data.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            arguments=dict(d.get("arguments") or {}),
            result=str(d.get("result", "")),
            success=bool(d.get("success", True)),
            error=d.get("error"),
            duration_ms=float(d.get("duration_ms", 0.0)),
        )


@dataclass
class TrajectoryStep:
    """One message in a trajectory.

    For ``role="assistant"`` the ``tool_calls`` field captures any tool
    invocations the model emitted on this turn. For ``role="tool"`` the
    ``tool_call_id`` field links back to the originating call.
    """

    role: TrajectoryRole
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryStep":
        role = d.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        return cls(
            role=role,
            content=str(d.get("content", "") or ""),
            tool_calls=[ToolCall.from_dict(tc) for tc in (d.get("tool_calls") or [])],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            metadata=dict(d.get("metadata") or {}),
        )


def _is_feedback_step(step: TrajectoryStep) -> bool:
    return (
        step.role == "user"
        or step.metadata.get("feedback") is True
        or step.metadata.get("next_state") is True
    )


@dataclass
class Trajectory:
    """A complete agent run normalised for training pipelines."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str = ""
    model: str = ""
    steps: list[TrajectoryStep] = field(default_factory=list)
    reward: float | None = None
    rewards: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── convenience ─────────────────────────────────────────────────

    def add_system(self, content: str, **meta: Any) -> TrajectoryStep:
        s = TrajectoryStep(role="system", content=content, metadata=dict(meta))
        self.steps.append(s)
        return s

    def add_user(self, content: str, **meta: Any) -> TrajectoryStep:
        s = TrajectoryStep(role="user", content=content, metadata=dict(meta))
        self.steps.append(s)
        return s

    def add_assistant(
        self,
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        **meta: Any,
    ) -> TrajectoryStep:
        s = TrajectoryStep(
            role="assistant",
            content=content,
            tool_calls=list(tool_calls or []),
            metadata=dict(meta),
        )
        self.steps.append(s)
        return s

    def add_tool(
        self,
        result: str,
        tool_call_id: str | None = None,
        name: str | None = None,
        **meta: Any,
    ) -> TrajectoryStep:
        s = TrajectoryStep(
            role="tool",
            content=result,
            tool_call_id=tool_call_id,
            name=name,
            metadata=dict(meta),
        )
        self.steps.append(s)
        return s

    # ── serialisation ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "task": self.task,
            "model": self.model,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
        }
        if self.reward is not None:
            d["reward"] = self.reward
        if self.rewards:
            d["rewards"] = dict(self.rewards)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        return cls(
            run_id=str(d.get("run_id") or uuid.uuid4().hex[:12]),
            task=str(d.get("task", "") or ""),
            model=str(d.get("model", "") or ""),
            steps=[TrajectoryStep.from_dict(s) for s in (d.get("steps") or [])],
            reward=(
                float(d["reward"])
                if d.get("reward") is not None
                else None
            ),
            rewards={k: float(v) for k, v in (d.get("rewards") or {}).items()},
            created_at=float(d.get("created_at") or time.time()),
            metadata=dict(d.get("metadata") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Trajectory":
        return cls.from_dict(json.loads(s))

    # ── views ───────────────────────────────────────────────────────

    @property
    def assistant_text(self) -> str:
        """Concatenated assistant content — used by string-matching scorers."""
        return "\n".join(s.content for s in self.steps if s.role == "assistant" and s.content)

    @property
    def final_assistant(self) -> TrajectoryStep | None:
        for s in reversed(self.steps):
            if s.role == "assistant":
                return s
        return None

    @property
    def prompt_text(self) -> str:
        """All non-assistant content joined for length-based scoring."""
        return "\n".join(
            s.content for s in self.steps if s.role in ("system", "user")
        )

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


def trajectories_to_dicts(trajs: Iterable[Trajectory]) -> list[dict[str, Any]]:
    """Convenience: materialise an iterable of trajectories as plain dicts."""
    return [t.to_dict() for t in trajs]


def to_next_state_transitions(traj: Trajectory) -> list[dict[str, Any]]:
    """Export assistant actions paired with following user/environment feedback."""
    transitions: list[dict[str, Any]] = []
    for idx, action in enumerate(traj.steps):
        if action.role != "assistant":
            continue
        next_idx = next(
            (
                j
                for j, step in enumerate(traj.steps[idx + 1:], start=idx + 1)
                if _is_feedback_step(step)
            ),
            None,
        )
        if next_idx is None:
            continue
        prior = next(
            (step for step in reversed(traj.steps[:idx]) if step.role in ("user", "system")),
            TrajectoryStep(role="user", content=traj.task),
        )
        transitions.append({
            "run_id": traj.run_id,
            "task": traj.task,
            "model": traj.model,
            "step_index": idx,
            "state": prior.to_dict(),
            "action": action.to_dict(),
            "next_state": traj.steps[next_idx].to_dict(),
            "reward": traj.reward,
            "done": next_idx == len(traj.steps) - 1,
            "metadata": {**traj.metadata, **action.metadata},
        })
    return transitions
