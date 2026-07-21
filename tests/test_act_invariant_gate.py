from __future__ import annotations

import asyncio

from clawagents.permissions.act_invariants import (
    completion_block_reason,
    contract_preamble,
    is_high_impact_command,
    load_contract,
)
from clawagents.permissions.mode import PermissionMode
from clawagents.run_context import RunContext
from clawagents.tools.plan_mode import ExitPlanModeTool
from clawagents.tools.plan_mode import EnterPlanModeTool
from clawagents.tools.registry import ToolRegistry, ToolResult


PRODUCTION_REVIEW = """## Production safety review
- Retry idempotency: identical content skips; a collision conflict aborts.
- Completion marker and published count reconciliation are required.
- Quarantine notification: write a non-sensitive alert to the operations folder.
- Identity validation: require one distinct identity across all pages in the full packet.
- Intake watch compatibility: monitor both legacy and new intake paths during migration.
- Regression fixtures: commit synthetic fixtures for each reproduced layout.
- Restart enablement: require an explicit manual per-start enable flag after reboot.
"""


PLAN = f"""# Safe publish plan

## Invariants
- Publish only after both the focused tests and the full replay pass.
- Never reuse evidence collected before the latest source edit.

## Verification gates
- `pytest tests/test_publish.py`
- `python scripts/replay.py --all`

## Execution
- Publish with `python publish.py --confirm`.

## Post-action reconciliation
- `python scripts/check_remote_state.py --expected-count 68`

{PRODUCTION_REVIEW}
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
        if command == "python publish.py --confirm --fail":
            return ToolResult(success=False, output="uploaded 30 of 68", error="exit 1")
        if command == "python publish.py --confirm --crash":
            raise RuntimeError("connection lost after upload")
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
        assert contract["reconciliation_commands"] == [
            "python scripts/check_remote_state.py --expected-count 68"
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
        assert "reconciliation required" in str(allowed.output).lower()
        assert completion_block_reason(ctx) is not None

        pending = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert pending.success is False
        assert "reconciliation is still pending" in (pending.error or "").lower()

        reconciled = await registry.execute_tool(
            "execute",
            {"command": "python scripts/check_remote_state.py --expected-count 68"},
            run_context=ctx,
        )
        assert reconciled.success is True
        assert "reconciliation complete" in str(reconciled.output).lower()
        assert completion_block_reason(ctx) is None

        consumed = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert consumed.success is False
        assert "consumed" in (consumed.error or "").lower()

    asyncio.run(run())


def test_reconciliation_command_cannot_repeat_external_action(tmp_path):
    from clawagents.permissions.act_invariants import (
        _write_contract,
        approve_plan_contract,
        gate_tool_call,
    )

    ctx = RunContext()
    ctx._metadata["workspace"] = str(tmp_path)
    malicious_plan = PLAN.replace(
        "python scripts/check_remote_state.py --expected-count 68",
        "python publish.py --confirm",
    )
    contract = approve_plan_contract(malicious_plan, ctx)
    assert contract is not None

    before_action = gate_tool_call(
        "execute", {"command": "python publish.py --confirm"}, ctx
    )
    assert before_action is not None
    assert "reconciliation must be read-only" in before_action

    contract["reconciliation_pending"] = True
    _write_contract(contract, ctx)
    while_pending = gate_tool_call(
        "execute", {"command": "python publish.py --confirm"}, ctx
    )
    assert while_pending is not None
    assert "can mutate state or repeat the external action" in while_pending


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


def test_high_impact_requires_plan_but_ordinary_commands_remain_unaffected(tmp_path):
    async def run():
        ctx = RunContext()
        ctx._metadata["workspace"] = str(tmp_path)
        blocked = await _registry().execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert blocked.success is False
        assert "require an approved plan" in (blocked.error or "").lower()

        ordinary = await _registry().execute_tool(
            "execute", {"command": "python split_all.py --profile billing_img"}, run_context=ctx
        )
        assert ordinary.success is True

    asyncio.run(run())


def test_invariant_only_plan_uses_fresh_generic_verification(tmp_path):
    async def run():
        ctx = RunContext()
        ctx.permission_mode = PermissionMode.PLAN
        ctx._metadata["workspace"] = str(tmp_path)
        ctx._metadata["pending_plan_text"] = (
            "# Plan\n\n## Invariants\n"
            "- Never publish before a fresh validation passes.\n\n"
            "## Post-action reconciliation\n"
            "- `python scripts/check_remote_state.py --expected-count 68`\n\n"
            + PRODUCTION_REVIEW
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


def test_failed_external_attempt_consumes_authorization_and_requires_reconciliation(tmp_path):
    async def run():
        ctx = await _approve_plan(tmp_path)
        registry = _registry()
        for command in (
            "pytest tests/test_publish.py",
            "python scripts/replay.py --all",
        ):
            assert (
                await registry.execute_tool(
                    "execute", {"command": command}, run_context=ctx
                )
            ).success

        failed = await registry.execute_tool(
            "execute",
            {"command": "python publish.py --confirm --fail"},
            run_context=ctx,
        )
        assert failed.success is False
        assert "reconciliation required" in str(failed.output).lower()
        contract = load_contract(ctx)
        assert contract is not None
        assert contract["authorization_consumed"] is True
        assert contract["reconciliation_pending"] is True
        assert contract["last_external_success"] is False
        assert completion_block_reason(ctx) is not None

        retry = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert retry.success is False
        assert "may have partially succeeded" in (retry.error or "").lower()

    asyncio.run(run())


def test_publish_plan_without_reconciliation_fails_closed(tmp_path):
    async def run():
        ctx = RunContext()
        ctx.permission_mode = PermissionMode.PLAN
        ctx._metadata["workspace"] = str(tmp_path)
        ctx._metadata["pending_plan_text"] = (
            "# Plan\n\n## Verification gates\n- `pytest tests/test_publish.py`"
        )
        assert (await ExitPlanModeTool().execute({}, run_context=ctx)).success
        registry = _registry()
        assert (
            await registry.execute_tool(
                "execute",
                {"command": "pytest tests/test_publish.py"},
                run_context=ctx,
            )
        ).success

        blocked = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert blocked.success is False
        assert "no exact post-action reconciliation" in (blocked.error or "").lower()

    asyncio.run(run())


def test_generic_publish_plan_does_not_require_app_specific_review(tmp_path):
    async def run():
        ctx = RunContext()
        ctx.permission_mode = PermissionMode.PLAN
        ctx._metadata["workspace"] = str(tmp_path)
        ctx._metadata["pending_plan_text"] = (
            "# Publish plan\n\n"
            "## Verification gates\n- `pytest tests/test_publish.py`\n\n"
            "## Post-action reconciliation\n"
            "- `python scripts/check_remote_state.py --expected-count 68`"
        )
        assert (await ExitPlanModeTool().execute({}, run_context=ctx)).success
        registry = _registry()
        assert (
            await registry.execute_tool(
                "execute",
                {"command": "pytest tests/test_publish.py"},
                run_context=ctx,
            )
        ).success

        allowed = await registry.execute_tool(
            "execute", {"command": "python publish.py --confirm"}, run_context=ctx
        )
        assert allowed.success is True
        assert "reconciliation required" in str(allowed.output).lower()

    asyncio.run(run())


def test_crashed_external_attempt_persists_reconciliation_state(tmp_path):
    async def run():
        ctx = await _approve_plan(tmp_path)
        registry = _registry()
        for command in (
            "pytest tests/test_publish.py",
            "python scripts/replay.py --all",
        ):
            assert (
                await registry.execute_tool(
                    "execute", {"command": command}, run_context=ctx
                )
            ).success

        crashed = await registry.execute_tool(
            "execute",
            {"command": "python publish.py --confirm --crash"},
            run_context=ctx,
        )
        assert crashed.success is False
        assert "connection lost after upload" in (crashed.error or "")
        contract = load_contract(ctx)
        assert contract is not None
        assert contract["authorization_consumed"] is True
        assert contract["reconciliation_pending"] is True
        assert contract["last_external_success"] is None
        assert completion_block_reason(ctx) is not None

    asyncio.run(run())


def test_agent_cannot_finalize_while_reconciliation_is_pending(tmp_path):
    async def run():
        from clawagents.agent import ClawAgent
        from clawagents.providers.llm import LLMProvider, LLMResponse

        class _FinalOnlyLLM(LLMProvider):
            name = "final-only"

            def __init__(self):
                self.calls = 0

            async def chat(self, messages, **kwargs):
                self.calls += 1
                return LLMResponse(
                    content="Everything is complete.",
                    model="fake",
                    tokens_used=1,
                )

        ctx = await _approve_plan(tmp_path)
        registry = _registry()
        for command in (
            "pytest tests/test_publish.py",
            "python scripts/replay.py --all",
        ):
            assert (
                await registry.execute_tool(
                    "execute", {"command": command}, run_context=ctx
                )
            ).success
        assert (
            await registry.execute_tool(
                "execute",
                {"command": "python publish.py --confirm --fail"},
                run_context=ctx,
            )
        ).success is False

        llm = _FinalOnlyLLM()
        agent = ClawAgent(
            llm=llm,
            tools=registry,
            streaming=False,
            use_native_tools=False,
        )
        state = await agent.invoke(
            "Finish the publish task",
            max_iterations=2,
            run_context=ctx,
        )

        assert llm.calls == 2
        assert state.status != "done"
        assert state.result != "Everything is complete."

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
    assert is_high_impact_command("python publish.py --dry-run --confirm")
    assert is_high_impact_command(
        "PUBLISH_ENABLED=false PUBLISH_ENABLED=true docker compose up -d"
    )
    assert is_high_impact_command(
        "python publish.py --dry-run && python publish.py --confirm"
    )
    assert is_high_impact_command(
        "python publish.py --dry-run\npython publish.py --confirm"
    )
    assert not is_high_impact_command("python split_all.py --profile billing_img")


def test_plan_tool_surfaces_production_acceptance_contract():
    from clawagents.tools.context_tools import WritePlanTool

    description = WritePlanTool.description.lower()
    assert "post-action reconciliation" in description
    assert "retry/rollback" in description
    assert "partial-failure" in description
    assert "observable evidence" in description
    assert "domain-specific safety constraints" in description
