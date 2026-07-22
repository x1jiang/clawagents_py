"""Act mode must not inherit Goal verifier / tools from a prior Goal run."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from clawagents.goal import GoalPauseReason, GoalStatus, GoalTracker
from clawagents.goal.tools import create_goal_tools
from clawagents.agent import create_claw_agent


def test_goal_tools_only_registered_in_goal_mode(monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_GOAL_AUTOPILOT", "1")
    act = create_claw_agent("gpt-4o-mini", goal_mode=False, streaming=False)
    goal = create_claw_agent("gpt-4o-mini", goal_mode=True, streaming=False)
    assert act.tools.get("start_goal") is None
    assert goal.tools.get("start_goal") is not None
    assert getattr(act, "goal_mode", None) is False
    assert getattr(goal, "goal_mode", None) is True


def test_act_does_not_bind_active_disk_goal(tmp_path, monkeypatch):
    """Regression: Act turns used to load `.clawagents/goal/state.json` and
    keep rejecting completions via the Goal verifier."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAW_FEATURE_GOAL_AUTOPILOT", "1")

    gt = GoalTracker(tmp_path)
    gt.start("finish the review")
    gt.state.status = GoalStatus.ACTIVE
    gt.save()
    assert gt.is_active()

    from clawagents.graph.agent_loop import run_agent_graph
    from clawagents.providers.llm import LLMMessage, LLMResponse
    from clawagents.run_context import RunContext
    from clawagents.tools.registry import ToolRegistry

    class _DoneLLM:
        name = "mock"

        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
            return LLMResponse(content="All done.", model="mock", tokens_used=3)

    ctx = RunContext()
    ctx._metadata["workspace"] = str(tmp_path)
    ctx._metadata["goal_mode"] = False

    events: list[str] = []

    def on_event(kind, data=None):
        if kind == "context" and isinstance(data, dict):
            events.append(str(data.get("message") or ""))

    state = asyncio.run(
        run_agent_graph(
            "say done",
            _DoneLLM(),
            tools=ToolRegistry(),
            max_iterations=3,
            streaming=False,
            run_context=ctx,
            on_event=on_event,
        )
    )
    assert state.status == "done"
    assert state.result == "All done."
    assert not any("goal verifier" in e for e in events)
    assert not any("active goal reminder" in e for e in events)


def test_goal_mode_still_binds_tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAW_FEATURE_GOAL_AUTOPILOT", "1")
    gt = GoalTracker(tmp_path)
    gt.start("keep going")
    gt.state.status = GoalStatus.ACTIVE
    gt.save()

    from clawagents.goal import attach_goal_to_run_context, get_goal_tracker
    from clawagents.run_context import RunContext

    ctx = RunContext()
    ctx._metadata["workspace"] = str(tmp_path)
    ctx._metadata["goal_mode"] = True
    # Mimic agent_loop bind condition
    attach_goal_to_run_context(ctx, GoalTracker(tmp_path))
    bound = get_goal_tracker(ctx)
    assert bound is not None
    assert bound.is_active()


def test_pause_active_goal_helper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gt = GoalTracker(tmp_path)
    gt.start("x")
    gt.state.status = GoalStatus.ACTIVE
    gt.save()
    gt.pause(GoalPauseReason.USER, "Act mode")
    assert gt.state.status == GoalStatus.PAUSED
    assert not gt.is_active()
