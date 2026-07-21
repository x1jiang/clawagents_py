"""Tests for v6.15 goal / sandbox / permissions / best-of-n / prefire."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PRODUCTION_REVIEW = """## Production safety review
- Retry idempotency: identical content skips; a collision conflict aborts.
- Completion marker and published count reconciliation are required.
- Quarantine notification: write a non-sensitive alert to the operations folder.
- Identity validation: require one distinct identity across all pages in the full packet.
- Intake watch compatibility: monitor both legacy and new intake paths during migration.
- Regression fixtures: commit synthetic fixtures for every reproduced layout.
- Restart enablement: require an explicit manual per-start enable flag after reboot.
"""


def test_permission_deny_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_PERMISSION_RULES", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.tools.permissions import PermissionEngine, PermissionRule, load_permission_engine

    engine = PermissionEngine()
    engine.add_rule(PermissionRule(tool="execute", decision="allow", priority=1))
    engine.add_rule(
        PermissionRule(
            tool="execute",
            arg_pattern="*rm -rf *",
            decision="deny",
            priority=10,
            message="no",
        )
    )
    d, msg = engine.evaluate("execute", {"command": "rm -rf /tmp/x"})
    assert d == "deny"
    assert msg == "no"

    cfg = tmp_path / ".clawagents" / "permissions.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            [{"tool": "write_file", "path_pattern": "**/.env", "decision": "deny"}]
        ),
        encoding="utf-8",
    )
    loaded = load_permission_engine(tmp_path)
    ok, _ = loaded.gate("write_file", {"path": "secrets/.env"})
    assert ok is False


def test_sandbox_allow_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_OS_SANDBOX_PROFILES", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.sandbox.profiles import resolve_sandbox

    sb = resolve_sandbox("workspace", workspace=str(tmp_path), default="workspace")
    # Inside workspace ok
    p = sb.safe_path("ok.txt")
    assert str(tmp_path) in p or p.endswith("ok.txt")
    with pytest.raises(ValueError):
        sb.safe_path("/etc/passwd")


def test_seatbelt_denies_writes_outside_first():
    from clawagents.sandbox.profiles import _seatbelt_profile_text

    text = _seatbelt_profile_text(cwd="/ws", network=True, read_only=False)
    assert "(deny file-write*)" in text
    assert text.index("(deny file-write*)") < text.index('(allow file-write* (subpath "/ws"))')


def test_seatbelt_allows_dev_null_in_writable_profile():
    from clawagents.sandbox.profiles import _seatbelt_profile_text

    text = _seatbelt_profile_text(cwd="/ws", network=True, read_only=False)
    assert '(allow file-write-data (literal "/dev/null"))' in text
    ro = _seatbelt_profile_text(cwd="/ws", network=False, read_only=True)
    assert '(allow file-write-data (literal "/dev/null"))' in ro


@pytest.mark.asyncio
async def test_goal_verifier_majority():
    from clawagents.goal import run_verifier

    async def llm(prompt: str) -> str:
        if "skeptic #1" in prompt or "skeptic #2" in prompt:
            return '{"achieved": true, "reason": "ok"}'
        return '{"achieved": false, "reason": "no"}'

    ok, votes = await run_verifier(
        llm,
        goal="ship",
        plan_text="Success criteria: done",
        evidence="done",
        skeptics=3,
    )
    assert ok is True
    assert len(votes) == 3


@pytest.mark.asyncio
async def test_goal_planner_fail_closed():
    from clawagents.goal import run_planner

    async def short(_prompt: str) -> str:
        return "too short"

    with pytest.raises(RuntimeError):
        await run_planner(short, "goal", workspace="/tmp")


@pytest.mark.asyncio
async def test_goal_planner_uses_generic_external_action_contract():
    from clawagents.goal import run_planner

    captured = ""

    async def llm(prompt: str) -> str:
        nonlocal captured
        captured = prompt
        return (
            "# Plan\n\n## Success criteria\n- [ ] Safe publish\n\n"
            "## Ordered steps\n1. Verify\n2. Publish\n3. Reconcile\n\n"
            "## Out of scope\n- None\n\n## Verification commands\n- test\n\n"
            "## Post-action reconciliation\n- count\n\n"
            "## Failure policy\n- Retry safely after reconciling remote state.\n"
        )

    await run_planner(llm, "Deploy the report watcher to production", workspace="/tmp")

    assert "retry, rollback, and partial-failure behavior" in captured
    assert "domain-specific safety constraints discovered from the task" in captured
    assert "quarantine notification" not in captured
    assert "identity values" not in captured
    assert "intake locations" not in captured


@pytest.mark.asyncio
async def test_goal_verifier_does_not_treat_exit_zero_as_production_proof():
    from clawagents.goal import run_verifier

    prompts: list[str] = []

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return '{"achieved": false, "reason": "reconciliation absent"}'

    ok, _ = await run_verifier(
        llm,
        goal="Publish packets to the remote share",
        plan_text="Success criteria: idempotent and reconciled",
        evidence="publisher exited 0",
        skeptics=1,
    )

    assert ok is False
    assert "exit code alone does not prove remote reconciliation" in prompts[0]


@pytest.mark.asyncio
async def test_production_goal_installs_deterministic_action_contract(tmp_path):
    from clawagents.goal.tools import StartGoalTool
    from clawagents.permissions.act_invariants import load_contract
    from clawagents.run_context import RunContext

    async def llm(_prompt: str) -> str:
        return (
            "# Production publish plan\n\n"
            "## Success criteria\n- [ ] Remote packet count reconciles\n\n"
            "## Ordered steps\n1. Verify\n2. Publish\n3. Reconcile\n\n"
            "## Out of scope\n- Changing source data\n\n"
            "## Verification gates\n- `pytest tests/test_publish.py`\n\n"
            "## Post-action reconciliation\n"
            "- `python scripts/check_remote_state.py --expected-count 68`\n\n"
            + PRODUCTION_REVIEW
        )

    ctx = RunContext()
    ctx._metadata["workspace"] = str(tmp_path)
    ctx._metadata["goal_llm_complete"] = llm
    result = await StartGoalTool().execute(
        {"goal": "Publish packets to the production share"},
        run_context=ctx,
    )

    assert result.success is True
    contract = load_contract(ctx)
    assert contract is not None
    assert contract["verification_commands"] == ["pytest tests/test_publish.py"]
    assert contract["reconciliation_commands"] == [
        "python scripts/check_remote_state.py --expected-count 68"
    ]


@pytest.mark.asyncio
async def test_production_goal_fails_closed_without_exact_reconciliation(tmp_path):
    from clawagents.goal.tools import StartGoalTool
    from clawagents.run_context import RunContext

    async def llm(_prompt: str) -> str:
        return (
            "# Production publish plan\n\n"
            "## Success criteria\n- [ ] Safe publish with verification\n\n"
            "## Ordered steps\n1. Test\n2. Publish\n\n"
            "## Out of scope\n- None\n\n"
            "## Verification gates\n- `pytest tests/test_publish.py`\n\n"
            + PRODUCTION_REVIEW
        )

    ctx = RunContext()
    ctx._metadata["workspace"] = str(tmp_path)
    ctx._metadata["goal_llm_complete"] = llm
    result = await StartGoalTool().execute(
        {"goal": "Deploy the production watcher"},
        run_context=ctx,
    )

    assert result.success is False
    assert "post-action reconciliation" in (result.error or "").lower()


def test_best_of_n_skill_bundled():
    from clawagents.skills.best_of_n import ensure_best_of_n_skill
    from clawagents.agent import _get_bundled_skills_dir

    path = ensure_best_of_n_skill(Path(_get_bundled_skills_dir()))
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "isolation" in body
    assert "worktree" in body


def test_prefire_forces_over_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_PREFIRE_COMPACTION", "1")
    from clawagents.config.features import is_enabled, reset

    reset()
    assert is_enabled("prefire_compaction")
    assert is_enabled("goal_autopilot")
    assert is_enabled("mid_turn_interject")
    assert is_enabled("permission_rules")
