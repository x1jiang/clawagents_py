"""Tests for subagent depth-cap and memory isolation (Hermes parity).

The :class:`TaskTool` must refuse to spawn a child sub-agent when the parent
``RunContext.depth`` is already at :data:`MAX_SUBAGENT_DEPTH`. This bounds
recursive delegation and keeps cost predictable. Children must also run with
``skip_memory=True`` so they cannot read or write the parent's memory state.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from clawagents.run_context import MAX_SUBAGENT_DEPTH, RunContext
from clawagents.tools.subagent import TaskTool


def _make_task_tool() -> TaskTool:
    """A TaskTool whose dependencies aren't actually exercised by the cap check."""
    # ``llm`` and ``tools`` aren't dereferenced before the depth check fires,
    # so passing ``None`` is fine for these tests.
    return TaskTool(llm=None, tools=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_top_level_call_passes_depth_check_and_invokes_run_agent_graph():
    tool = _make_task_tool()

    captured: dict[str, Any] = {}

    async def fake_run_agent_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)

        class _State:
            status = "done"
            result = "ok"
            tool_calls = 0
            iterations = 1

        return _State()

    with patch(
        "clawagents.graph.agent_loop.run_agent_graph",
        new=fake_run_agent_graph,
    ):
        ctx = RunContext()  # depth=0
        ctx.activate_skill("restricted", ["task", "read_file"], "abc123")
        result = await tool.execute(
            {"description": "do a thing"},
            run_context=ctx,
        )

    assert result.success is True
    child_ctx = captured["run_context"]
    assert child_ctx.depth == 1
    assert child_ctx.skip_memory is True
    assert child_ctx.active_skill_name == "restricted"
    assert child_ctx.active_skill_content_hash == "abc123"
    assert child_ctx.active_skills == {"restricted": "abc123"}
    assert child_ctx.active_skill_allowed_tools == frozenset({"task", "read_file"})


@pytest.mark.asyncio
async def test_subagent_at_depth_one_can_still_spawn_one_more():
    """depth=1 → MAX=2 → still strictly less, so a single grandchild is allowed."""
    tool = _make_task_tool()

    async def fake_run_agent_graph(**kwargs: Any) -> Any:
        class _State:
            status = "done"
            result = "ok"
            tool_calls = 0
            iterations = 1

        return _State()

    captured: dict[str, Any] = {}

    async def capturing(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return await fake_run_agent_graph(**kwargs)

    with patch(
        "clawagents.graph.agent_loop.run_agent_graph",
        new=capturing,
    ):
        ctx = RunContext(depth=1)
        result = await tool.execute(
            {"description": "do a thing"},
            run_context=ctx,
        )

    assert result.success is True
    child_ctx = captured["run_context"]
    assert child_ctx.depth == 2  # bumped to the cap, but spawn was allowed.
    assert child_ctx.skip_memory is True


@pytest.mark.asyncio
async def test_subagent_at_cap_is_refused_without_running_child():
    """depth >= MAX_SUBAGENT_DEPTH → refuse, never call run_agent_graph."""
    tool = _make_task_tool()

    called = False

    async def fake_run_agent_graph(**_: Any) -> Any:
        nonlocal called
        called = True

        class _State:
            status = "done"
            result = "should not run"
            tool_calls = 0
            iterations = 1

        return _State()

    with patch(
        "clawagents.graph.agent_loop.run_agent_graph",
        new=fake_run_agent_graph,
    ):
        ctx = RunContext(depth=MAX_SUBAGENT_DEPTH)
        result = await tool.execute(
            {"description": "another nested call"},
            run_context=ctx,
        )

    assert called is False
    assert result.success is False
    # Refusal message should be informative and reference the cap.
    msg = (result.error or "") + (result.output or "")
    assert "depth cap" in msg.lower()
    assert str(MAX_SUBAGENT_DEPTH) in msg


@pytest.mark.asyncio
async def test_missing_run_context_treated_as_top_level():
    """Backward compat: callers that don't pass run_context are treated as depth=0."""
    tool = _make_task_tool()

    captured: dict[str, Any] = {}

    async def fake_run_agent_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)

        class _State:
            status = "done"
            result = "ok"
            tool_calls = 0
            iterations = 1

        return _State()

    with patch(
        "clawagents.graph.agent_loop.run_agent_graph",
        new=fake_run_agent_graph,
    ):
        result = await tool.execute({"description": "do a thing"})

    assert result.success is True
    child_ctx = captured["run_context"]
    assert child_ctx.depth == 1
    assert child_ctx.skip_memory is True


def test_run_context_depth_default_is_zero_and_skip_memory_is_false():
    ctx = RunContext()
    assert ctx.depth == 0
    assert ctx.skip_memory is False


def test_max_subagent_depth_is_two():
    """The cap is documented in AGENTS.md; lock it to 2."""
    assert MAX_SUBAGENT_DEPTH == 2
