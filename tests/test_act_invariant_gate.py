from __future__ import annotations

import asyncio

from clawagents.permissions.act_invariants import (
    contract_preamble,
    is_high_impact_command,
    load_contract,
)
from clawagents.permissions.mode import PermissionMode
from clawagents.run_context import RunContext
from clawagents.tools.plan_mode import ExitPlanModeTool
from clawagents.tools.plan_mode import EnterPlanModeTool
from clawagents.tools.registry import ToolRegistry, ToolResult


PLAN = """# Safe publish plan

## Invariants
- Publish only after both the focused tests and the full replay pass.
- Never reuse evidence collected before the latest source edit.

## Verification gates
- `pytest tests/test_publish.py`
- `python scripts/replay.py --all`

## Execution
- Publish with `python publish.py --confirm`.
"""


class _FakeExecute:
    name = "execute"
    keywords = ["shell"]
    description = "fake shell"
    parameters = {"command": {"type": "string", "required": True}}

    async def execute(self, args, run_context=None):
        command = str(args.get("command") or "")
        if command == "pytest tests/test_publish.py --fail":
            return ToolResult(success=False, output="failed", error="exit 1")
        return ToolResult(success=True, output=f"ran: {command}")


class _FakeEdit:
    name = "edit_file"
    keywords = ["edit"]
    description = "fake edit"
    parameters = {"path": {"type": "string", "required": True}}

    async def execute(self, args, run_context=None):
        return ToolResult(success=True, output="edited")


async def _approve_plan(tmp_path):
    ctx = RunContext()
    ctx.permission_mode = PermissionMode.PLAN
    ctx._metadata["workspace"] = str(tmp_path)
    ctx._metadata["pending_plan_text"] = PLAN
    result = await ExitPlanModeTool().execute({}, run_context=ctx)
    assert result.success is True
    assert ctx.permission_mode == PermissionMode.DEFAULT
    return ctx


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_FakeExecute())
    registry.register(_FakeEdit())
    return registry


def test_approved_plan_survives_mode_transition_and_persists(tmp_path):
    async def run():
        ctx = await _approve_plan(tmp_path)
        contract = load_contract(ctx)
        assert contract is not None
        assert contract["plan_sha256"]
        assert contract["verification_commands"] == [
            "pytest tests/test_publish.py",
            "python scripts/replay.py --all",
        ]
        assert "both the focused tests" in contract["invariants"][0]

        resumed = RunContext()
        resumed._metadata["workspace"] = str(tmp_path)
        resumed_contract = load_contract(resumed)
        assert resumed_contract is not None
        assert resumed_contract["plan_sha256"] == contract["plan_sha256"]
        assert "2 verification gate(s) remaining" in contract_preamble(resumed)

    asyncio.run(run())


def test_entering_plan_mode_persists_fail_closed_pending_state(tmp_path):
    async def run():
        planning = RunContext()
        planning._metadata["workspace"] = str(tmp_path)
        entered = await EnterPlanModeTool().execute({}, run_context=planning)
        assert entered.success is True

        resumed = RunContext()
        resumed._metadata["workspace"] = str(tmp_path)
        blocked = await _registry().execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=resumed
        )
        assert blocked.success is False
        assert "pending approval" in (blocked.error or "").lower()

    asyncio.run(run())


def test_high_impact_action_requires_every_planned_verification(tmp_path):
    async def run():
        ctx = await _approve_plan(tmp_path)
        registry = _registry()

        blocked = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert blocked.success is False
        assert "plan invariant gate" in (blocked.error or "").lower()
        assert "pytest tests/test_publish.py" in (blocked.error or "")
        assert "python scripts/replay.py --all" in (blocked.error or "")

        first = await registry.execute_tool(
            "execute", {"command": "pytest tests/test_publish.py"}, run_context=ctx
        )
        assert first.success is True
        assert "verification gate satisfied" in str(first.output).lower()

        still_blocked = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert still_blocked.success is False
        assert "python scripts/replay.py --all" in (still_blocked.error or "")

        second = await registry.execute_tool(
            "execute", {"command": "python scripts/replay.py --all"}, run_context=ctx
        )
        assert second.success is True

        allowed = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert allowed.success is True

        consumed = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert consumed.success is False
        assert "consumed" in (consumed.error or "").lower()

    asyncio.run(run())


def test_failed_check_and_later_edit_cannot_authorize_publish(tmp_path):
    async def run():
        ctx = await _approve_plan(tmp_path)
        registry = _registry()

        failed = await registry.execute_tool(
            "execute",
            {"command": "pytest tests/test_publish.py --fail"},
            run_context=ctx,
        )
        assert failed.success is False

        await registry.execute_tool(
            "execute", {"command": "pytest tests/test_publish.py"}, run_context=ctx
        )
        await registry.execute_tool(
            "execute", {"command": "python scripts/replay.py --all"}, run_context=ctx
        )

        edited = await registry.execute_tool(
            "edit_file", {"path": "publish.py"}, run_context=ctx
        )
        assert edited.success is True

        blocked = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert blocked.success is False
        assert "latest mutation" in (blocked.error or "").lower()
        assert "pytest tests/test_publish.py" in (blocked.error or "")

    asyncio.run(run())


def test_unplanned_ordinary_commands_remain_unaffected(tmp_path):
    async def run():
        ctx = RunContext()
        ctx._metadata["workspace"] = str(tmp_path)
        result = await _registry().execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert result.success is True

    asyncio.run(run())


def test_invariant_only_plan_uses_fresh_generic_verification(tmp_path):
    async def run():
        ctx = RunContext()
        ctx.permission_mode = PermissionMode.PLAN
        ctx._metadata["workspace"] = str(tmp_path)
        ctx._metadata["pending_plan_text"] = (
            "# Plan\n\n## Invariants\n- Never publish before a fresh validation passes."
        )
        assert (await ExitPlanModeTool().execute({}, run_context=ctx)).success
        registry = _registry()

        blocked = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert blocked.success is False
        assert "test, validation, or dry-run" in (blocked.error or "")

        verified = await registry.execute_tool(
            "execute", {"command": "python -m py_compile publish.py"}, run_context=ctx
        )
        assert verified.success is True
        assert "fresh verification evidence recorded" in str(verified.output)
        assert (
            await registry.execute_tool(
                "execute", {"command": "python publish.py --confirm"}, run_context=ctx
            )
        ).success

    asyncio.run(run())


def test_corrupt_contract_fails_closed_for_external_side_effect(tmp_path):
    async def run():
        state = tmp_path / ".clawagents" / "act-invariants.json"
        state.parent.mkdir()
        state.write_text("{not-json", encoding="utf-8")
        ctx = RunContext()
        ctx._metadata["workspace"] = str(tmp_path)
        blocked = await _registry().execute_tool(
            "execute", {"command": "git push origin main"}, run_context=ctx
        )
        assert blocked.success is False
        assert "unreadable" in (blocked.error or "").lower()

    asyncio.run(run())


def test_high_impact_classifier_covers_real_actions_without_blocking_checks():
    assert is_high_impact_command(
        "python3 publish_sandbox.py --run-dir ready/1 --confirm"
    )
    assert is_high_impact_command("PUBLISH_ENABLED=true docker compose up -d")
    assert is_high_impact_command("git push origin main")
    assert not is_high_impact_command("pytest tests/test_publish.py")
    assert not is_high_impact_command("python publish.py --dry-run")
    assert not is_high_impact_command("python split_all.py --profile billing_img")
