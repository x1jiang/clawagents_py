"""Reviewing a plan is a conversation, and it has to survive being slow.

Two things have to hold for "request changes" to be worth having:

* the tool that is blocked on the reviewer must still be alive when they answer
  -- the default tool timeout is sized for commands, and a plan long enough to
  be worth reviewing takes longer than that to read;
* what the reviewer typed has to reach the model, not just the fact that they
  were unhappy.
"""

from __future__ import annotations

import asyncio

import pytest

from clawagents.permissions.mode import PermissionMode
from clawagents.permissions.plan_approval import (
    PlanApprovalAction,
    PlanApprovalDecision,
)
from clawagents.run_context import RunContext
from clawagents.tools.interactive import AskUserTool
from clawagents.tools.plan_mode import ExitPlanModeTool
from clawagents.tools.registry import (
    HUMAN_TOOL_TIMEOUT_S,
    ToolRegistry,
    ToolResult,
)


class _Slow:
    """Blocks long enough to trip a deliberately tiny registry timeout."""

    name = "slow"
    description = "slow"
    keywords: list[str] = []
    parameters: dict = {}

    async def execute(self, args):
        await asyncio.sleep(0.25)
        return ToolResult(True, "finished")


class _SlowHuman(_Slow):
    name = "slow_human"
    waits_for_human = True


# ─── the tool must outlive the reviewer's reading time ────────────────────


@pytest.mark.asyncio
async def test_a_human_gated_tool_is_not_killed_by_the_machine_timeout():
    reg = ToolRegistry(tool_timeout_s=0.05)
    reg.register(_SlowHuman())
    result = await reg.execute_tool("slow_human", {})
    assert result.success, result.error
    assert "timed out" not in (result.error or "")


@pytest.mark.asyncio
async def test_an_ordinary_tool_still_times_out():
    """The exemption has to be opt-in, or every hung command hangs the run."""
    reg = ToolRegistry(tool_timeout_s=0.05)
    reg.register(_Slow())
    result = await reg.execute_tool("slow", {})
    assert result.success is False
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_a_human_timeout_does_not_advise_tuning_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    """Nothing the model passes makes a person answer faster; saying otherwise
    invites it to retry with different arguments instead of reporting."""
    # Shrink the human ceiling so the test does not wait an hour for it.
    monkeypatch.setattr(
        "clawagents.tools.registry.HUMAN_TOOL_TIMEOUT_S", 0.05, raising=True
    )

    class _Hang(_SlowHuman):
        name = "hang"

        async def execute(self, args):
            await asyncio.sleep(30)
            return ToolResult(True, "unreachable")

    reg = ToolRegistry(tool_timeout_s=0.05)
    reg.register(_Hang())
    result = await reg.execute_tool("hang", {})

    assert result.success is False
    assert "timeout parameter" not in (result.error or "")
    assert "Nobody responded" in (result.error or "")


def test_the_tools_that_block_on_a_person_say_so():
    """The flag is what buys the longer ceiling, so it must actually be set."""
    assert ExitPlanModeTool.waits_for_human is True
    assert AskUserTool.waits_for_human is True


def test_a_callers_larger_timeout_is_not_shortened():
    reg = ToolRegistry(tool_timeout_s=HUMAN_TOOL_TIMEOUT_S * 2)
    assert reg._timeout_for(_SlowHuman()) == HUMAN_TOOL_TIMEOUT_S * 2  # noqa: SLF001


def test_the_human_ceiling_leaves_room_to_actually_read_a_plan():
    assert HUMAN_TOOL_TIMEOUT_S >= 600


# ─── the feedback itself has to arrive ────────────────────────────────────


def _plan_ctx() -> RunContext:
    ctx = RunContext()
    ctx.permission_mode = PermissionMode.PLAN
    ctx._metadata["pending_plan_text"] = "# Plan\n1. Extract everything"
    return ctx


@pytest.mark.asyncio
async def test_feedback_text_reaches_the_model():
    """Without this the model learns only that someone objected, which is not
    enough to produce a different plan."""
    feedback = "Use the existing .venv, and confirm the cohort count first."

    async def on_exit(_plan_text, _ctx):
        return PlanApprovalDecision(
            PlanApprovalAction.REQUEST_CHANGES, comment=feedback
        )

    result = await ExitPlanModeTool(on_exit_plan_mode=on_exit).execute(
        {}, run_context=_plan_ctx()
    )
    assert result.success is False
    assert feedback in result.output


@pytest.mark.asyncio
async def test_requesting_changes_keeps_the_plan_gate_closed():
    """The point of feedback is a revision, so writes must stay locked."""

    async def on_exit(_plan_text, _ctx):
        return PlanApprovalDecision(
            PlanApprovalAction.REQUEST_CHANGES, comment="narrow the scope"
        )

    ctx = _plan_ctx()
    await ExitPlanModeTool(on_exit_plan_mode=on_exit).execute({}, run_context=ctx)
    assert ctx.permission_mode == PermissionMode.PLAN


@pytest.mark.asyncio
async def test_the_reviewer_sees_the_plan_they_are_commenting_on():
    """Feedback is about a specific plan; the callback has to be handed that
    text rather than the reviewer being asked to trust the transcript."""
    seen: list[str] = []

    async def on_exit(plan_text, _ctx):
        seen.append(plan_text)
        return PlanApprovalDecision(PlanApprovalAction.APPROVE)

    await ExitPlanModeTool(on_exit_plan_mode=on_exit).execute(
        {}, run_context=_plan_ctx()
    )
    assert seen and "Extract everything" in seen[0]


@pytest.mark.asyncio
async def test_a_revised_plan_can_be_approved_on_the_next_pass():
    """The loop has to terminate: feedback, revise, approve."""
    calls = {"n": 0}

    async def on_exit(_plan_text, _ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return PlanApprovalDecision(
                PlanApprovalAction.REQUEST_CHANGES, comment="use uv"
            )
        return PlanApprovalDecision(PlanApprovalAction.APPROVE)

    tool = ExitPlanModeTool(on_exit_plan_mode=on_exit)
    ctx = _plan_ctx()

    first = await tool.execute({}, run_context=ctx)
    assert first.success is False
    assert ctx.permission_mode == PermissionMode.PLAN

    second = await tool.execute({}, run_context=ctx)
    assert second.success is True
    assert ctx.permission_mode == PermissionMode.DEFAULT


@pytest.mark.asyncio
async def test_empty_feedback_does_not_masquerade_as_a_reason():
    """The UI blocks an empty send, but a host or an older client can still
    produce one -- it must not render as `Feedback: `."""

    async def on_exit(_plan_text, _ctx):
        return PlanApprovalDecision(PlanApprovalAction.REQUEST_CHANGES, comment="")

    result = await ExitPlanModeTool(on_exit_plan_mode=on_exit).execute(
        {}, run_context=_plan_ctx()
    )
    assert result.success is False
    assert "Feedback:" not in result.output
