"""Background jobs announce their own completion.

Otherwise the agent only learns a job ended if it remembers to poll
``task_status`` — a job finishing while it works elsewhere goes unnoticed.
"""

from __future__ import annotations

import json

import pytest

from clawagents.background import BackgroundJobManager
from clawagents.tools.background_task import _TaskCreateTool
from clawagents.tools.registry import ToolRegistry, ToolResult


class _Echo:
    name = "echo"
    description = "echo"
    keywords: list[str] = []
    parameters: dict = {}

    async def execute(self, args):
        return ToolResult(True, "echo output")


def _registry(manager: BackgroundJobManager) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_Echo())
    reg.register(_TaskCreateTool(manager))
    return reg


async def _start_and_finish(reg: ToolRegistry, manager, command):
    created = await reg.execute_tool("task_create", {"command": command})
    job_id = json.loads(created.raw_output)["job_id"]
    await manager.await_complete(job_id, timeout=10)
    return job_id


@pytest.mark.asyncio
async def test_completion_is_announced_on_the_next_tool_result():
    manager = BackgroundJobManager()
    reg = _registry(manager)
    await _start_and_finish(reg, manager, ["/bin/sh", "-c", "exit 0"])

    result = await reg.execute_tool("echo", {})
    assert "<system-reminder>" in result.output
    assert "Background job(s) finished" in result.output


@pytest.mark.asyncio
async def test_a_failure_carries_evidence():
    manager = BackgroundJobManager()
    reg = _registry(manager)
    await _start_and_finish(
        reg, manager, ["/bin/sh", "-c", "echo boom >&2; exit 3"]
    )

    output = (await reg.execute_tool("echo", {})).output
    assert "exit 3" in output
    assert "boom" in output  # the actionable part


@pytest.mark.asyncio
async def test_each_completion_is_announced_exactly_once():
    manager = BackgroundJobManager()
    reg = _registry(manager)
    await _start_and_finish(reg, manager, ["/bin/sh", "-c", "exit 0"])

    assert "<system-reminder>" in (await reg.execute_tool("echo", {})).output
    assert "<system-reminder>" not in (await reg.execute_tool("echo", {})).output


@pytest.mark.asyncio
async def test_agents_never_see_each_others_jobs():
    """One process can host concurrent agents over a shared job manager."""
    manager = BackgroundJobManager()
    owner, bystander = _registry(manager), _registry(manager)
    await _start_and_finish(owner, manager, ["/bin/sh", "-c", "exit 0"])

    assert "<system-reminder>" not in (await bystander.execute_tool("echo", {})).output
    assert "<system-reminder>" in (await owner.execute_tool("echo", {})).output


@pytest.mark.asyncio
async def test_running_jobs_are_not_announced():
    manager = BackgroundJobManager()
    reg = _registry(manager)
    created = await reg.execute_tool("task_create", {"command": ["/bin/sh", "-c", "sleep 30"]})
    job_id = json.loads(created.raw_output)["job_id"]
    try:
        assert "<system-reminder>" not in (await reg.execute_tool("echo", {})).output
    finally:
        await manager.cancel(job_id)


@pytest.mark.asyncio
async def test_task_tools_do_not_carry_the_notice():
    """task_status already reports job state; a notice there is redundant."""
    manager = BackgroundJobManager()
    reg = _registry(manager)
    await _start_and_finish(reg, manager, ["/bin/sh", "-c", "exit 0"])

    from clawagents.tools.background_task import _TaskStatusTool

    reg.register(_TaskStatusTool(manager))
    status = await reg.execute_tool("task_status", {"job_id": list(reg._owned_jobs)[0]})
    assert "<system-reminder>" not in str(status.output)
