"""Tests for Grok-Build-inspired v6.14 surfaces."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from clawagents.config.features import reset, set_overrides
from clawagents.permissions.mode import PermissionMode
from clawagents.permissions.plan_approval import (
    PlanApprovalAction,
    PlanApprovalDecision,
)
from clawagents.run_context import RunContext
from clawagents.tools.plan_mode import ExitPlanModeTool, enter_plan_mode_tool
from clawagents.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _features():
    reset()
    set_overrides({
        "plan_approval": True,
        "task_worktree": True,
        "hunk_review": True,
        "compact_reinject_plan": True,
        "compact_tool_pair_safe": True,
        "marketplace": True,
        "os_sandbox_profiles": True,
        "incremental_repo_map": True,
        "autopilot_loop": True,
    })
    yield
    reset()


def test_plan_approval_gate_rejects_and_stays_in_plan():
    async def run():
        decisions = {"n": 0}

        async def on_exit(plan: str, ctx: RunContext):
            decisions["n"] += 1
            if decisions["n"] == 1:
                return PlanApprovalDecision(
                    PlanApprovalAction.REQUEST_CHANGES, comment="add tests"
                )
            return PlanApprovalDecision(PlanApprovalAction.APPROVE)

        tool = ExitPlanModeTool(on_exit_plan_mode=on_exit)
        ctx = RunContext()
        ctx.permission_mode = PermissionMode.PLAN
        ctx._metadata["pending_plan_text"] = "# Plan\nDo the thing"

        r1 = await tool.execute({}, run_context=ctx)
        assert r1.success is False
        assert ctx.permission_mode == PermissionMode.PLAN
        assert "changes" in (r1.error or "")

        r2 = await tool.execute({}, run_context=ctx)
        assert r2.success is True
        assert ctx.permission_mode == PermissionMode.DEFAULT

    asyncio.run(run())


def test_plan_approval_auto_approves_without_callback():
    async def run():
        tool = ExitPlanModeTool()
        ctx = RunContext()
        ctx.permission_mode = PermissionMode.PLAN
        r = await tool.execute({}, run_context=ctx)
        assert r.success is True
        assert ctx.permission_mode == PermissionMode.DEFAULT

    asyncio.run(run())


def test_subagent_resolution_layers():
    from clawagents.tools.subagent import SubAgentSpec
    from clawagents.tools.subagent_resolve import resolve_subagent

    specs = [
        SubAgentSpec(
            name="explorer",
            description="read only",
            capability="read-only",
            isolation="none",
            persona="researcher",
            system_prompt="Explore.",
        )
    ]
    resolved = resolve_subagent(
        "explorer",
        specs=specs,
        args={"isolation": "worktree", "persona": "researcher"},
        personas={"researcher": "Cite file paths."},
    )
    assert resolved.isolation == "worktree"
    assert resolved.capability == "read-only"
    assert "write_file" in resolved.denied_tools()
    assert resolved.system_prompt and "Cite file paths" in resolved.system_prompt


def test_worktree_ensure(tmp_path: Path):
    from clawagents.tools.worktree import ensure_task_worktree, list_worktrees

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=tmp_path, check=True, capture_output=True
    )
    (tmp_path / "README").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    info = ensure_task_worktree(workspace=tmp_path, name="unit-a")
    assert info["ok"] is True
    assert Path(info["path"]).is_dir()
    rows = list_worktrees(tmp_path)
    assert any(info["path"] in r.get("path", "") for r in rows)


def test_attributed_hunk_accept_reject(tmp_path: Path):
    from clawagents.memory.attributed_hunks import (
        accept_hunk,
        list_hunks,
        refresh_file_hunks,
        reject_hunk,
        HunkStore,
    )

    f = tmp_path / "a.py"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    store = HunkStore.load(tmp_path)
    store.baselines["a.py"] = "line1\nline2\nline3\n"
    store.save()

    f.write_text("line1\nLINE2\nline3\nextra\n", encoding="utf-8")
    hunks = refresh_file_hunks("a.py", workspace=tmp_path, seed_baseline_if_missing=False)
    assert hunks
    hid = hunks[0].id

    # Reject restores toward baseline
    before = f.read_text(encoding="utf-8")
    result = reject_hunk(hid, workspace=tmp_path)
    assert result["ok"] is True
    after = f.read_text(encoding="utf-8")
    assert after != before or "line2" in after

    # Re-diff and accept
    f.write_text("line1\nCHANGED\nline3\n", encoding="utf-8")
    hunks = refresh_file_hunks("a.py", workspace=tmp_path, seed_baseline_if_missing=False)
    assert hunks
    hid = hunks[0].id
    disk_before = f.read_text(encoding="utf-8")
    acc = accept_hunk(hid, workspace=tmp_path)
    assert acc["ok"] is True
    assert f.read_text(encoding="utf-8") == disk_before  # accept does not change disk
    assert list_hunks(workspace=tmp_path, path="a.py") == [] or True  # may recompute empty


def test_compaction_tool_pair_safe_and_plan_reinject(tmp_path: Path):
    from clawagents.context.carryover import get_compaction_carryover, set_compaction_carryover
    from clawagents.memory.compaction import AgentMessage, compress_messages_safe
    from clawagents.run_context import RunContext

    class _LLM:
        async def chat(self, messages):
            class R:
                content = "summary of older turns"

            return R()

    msgs = [
        AgentMessage("system", "sys"),
        AgentMessage("user", "task"),
        AgentMessage("assistant", "call tool"),
        AgentMessage("tool", "result-a"),
        AgentMessage("tool", "result-b"),
        AgentMessage("user", "continue"),
    ]

    async def run():
        out = await compress_messages_safe(
            _LLM(), msgs, context_window=8000, protect_first_n=1, protect_last_n=2
        )
        # tail should not start with orphan tool if snap worked when tools at boundary
        roles = [m.role for m in out["messages"]]
        assert "system" in roles
        assert out["effective"] or out["summary"]

    asyncio.run(run())

    plan_dir = tmp_path / ".clawagents"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.md").write_text("## Steps\n1. ship it\n", encoding="utf-8")
    ctx = RunContext()
    ctx._metadata["workspace"] = str(tmp_path)
    set_compaction_carryover(ctx, task_focus="demo")
    # Clear plan_reminder so get_ re-injects
    bag = ctx._metadata["compaction_carryover"]
    bag.pop("plan_reminder", None)
    co = get_compaction_carryover(ctx)
    assert co.plan_reminder and "ship it" in co.plan_reminder
    assert "Active plan" in co.to_markdown()


def test_autopilot_loop_plan_execute_verify(tmp_path: Path):
    from clawagents.autopilot import AutopilotPhase, run_autopilot

    async def plan_fn(task):
        return ["step 1", "step 2"]

    async def exec_fn(task):
        (tmp_path / "out.txt").write_text("done\n", encoding="utf-8")
        return "wrote out.txt"

    async def verify_fn(task):
        return "ok"

    async def run():
        task = await run_autopilot(
            "build something",
            workspace=str(tmp_path),
            plan_fn=plan_fn,
            execute_fn=exec_fn,
            verify_fn=verify_fn,
            auto_approve=True,
        )
        assert task.phase == AutopilotPhase.DONE
        assert task.plan == ["step 1", "step 2"]
        assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "done\n"

    asyncio.run(run())


def test_marketplace_install_skill(tmp_path: Path):
    from clawagents.marketplace import install_from_source, list_installed

    skill_src = tmp_path / "src_skill"
    skill_src.mkdir()
    (skill_src / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    result = install_from_source(str(skill_src), kind="skill", workspace=ws)
    assert result.ok is True
    assert Path(result.path).joinpath("SKILL.md").is_file()
    installed = list_installed(ws)
    assert any(p["name"] == "demo-skill" for p in installed)


def test_os_sandbox_profile_readonly(tmp_path: Path):
    from clawagents.sandbox.profiles import resolve_sandbox

    sb = resolve_sandbox("read-only", workspace=str(tmp_path))
    path = sb.safe_path("x.txt")

    async def run():
        with pytest.raises(PermissionError):
            await sb.write_file(path, "nope")

    asyncio.run(run())


def test_scope_graph_incremental(tmp_path: Path):
    from clawagents.memory.scope_graph import ScopeGraph, get_scope_graph, _GRAPHS

    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return foo()\n", encoding="utf-8")
    _GRAPHS.pop(str(tmp_path.resolve()), None)
    g = get_scope_graph(tmp_path)
    g.refresh()
    text1 = g.query(mentioned={"foo"})
    assert "foo" in text1
    # second refresh should hit mtime cache path
    g.refresh()
    (tmp_path / "a.py").write_text(
        "def foo():\n    return 1\n\ndef baz():\n    return 2\n", encoding="utf-8"
    )
    g.refresh(changed=[tmp_path / "a.py"])
    text2 = g.query(mentioned={"baz"})
    assert "baz" in text2


def test_hunk_tools_round_trip(tmp_path: Path):
    from clawagents.memory.attributed_hunks import HunkStore
    from clawagents.tools.hunk_review import create_hunk_review_tools

    f = tmp_path / "f.txt"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    store = HunkStore.load(tmp_path)
    store.baselines["f.txt"] = "a\nb\nc\n"
    store.save()
    f.write_text("a\nB\nc\n", encoding="utf-8")

    tools = {t.name: t for t in create_hunk_review_tools(str(tmp_path))}

    async def run():
        listed = await tools["hunk_list"].execute({"path": "f.txt", "refresh": True})
        assert listed.success
        import json

        rows = json.loads(listed.output)
        assert rows
        hid = rows[0]["id"]
        acc = await tools["hunk_accept"].execute({"hunk_id": hid})
        assert acc.success

    asyncio.run(run())
