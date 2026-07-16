"""Minimal repo autopilot foundation (OpenHarness 0.1.9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class AutopilotPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"


@dataclass
class AutopilotTask:
    id: str
    goal: str
    workspace: str
    phase: AutopilotPhase = AutopilotPhase.IDLE
    plan: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


AutopilotRunner = Callable[[AutopilotTask], Awaitable[dict[str, Any]]]


class AutopilotRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, AutopilotRunner] = {}

    def register(self, name: str, runner: AutopilotRunner) -> None:
        self._runners[name] = runner

    def get(self, name: str) -> AutopilotRunner | None:
        return self._runners.get(name)

    def list_runners(self) -> list[str]:
        return sorted(self._runners.keys())


DEFAULT_AUTOPILOT_REGISTRY = AutopilotRegistry()


def __getattr__(name: str) -> Any:
    if name == "run_autopilot":
        from clawagents.autopilot.loop import run_autopilot

        return run_autopilot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AutopilotPhase",
    "AutopilotTask",
    "AutopilotRunner",
    "AutopilotRegistry",
    "DEFAULT_AUTOPILOT_REGISTRY",
    "run_autopilot",
]
