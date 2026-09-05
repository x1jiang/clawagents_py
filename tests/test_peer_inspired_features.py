"""Tests for peer-inspired features (OpenClaw, OpenHarness, DeepAgents, Hermes)."""

from __future__ import annotations

import asyncio
import json


def test_skill_workshop_create_scan_apply_rollback(tmp_path):
    from clawagents.skills.workshop.service import SkillWorkshopService

    skills = tmp_path / "skills"
    skills.mkdir()
    svc = SkillWorkshopService(tmp_path, skills)
    created = svc.create(
        name="demo-skill",
        description="Demo",
        body="# Demo\nDo the thing.",
        goal="test",
    )
    assert created["status"] == "pending"
    proposal_id = created["id"]
    applied = svc.apply(proposal_id)
    assert applied["ok"] is True
    assert (skills / "demo-skill" / "SKILL.md").is_file()
    rollback_id = applied["rollback_id"]
    assert rollback_id
    (skills / "demo-skill" / "SKILL.md").write_text("# mutated", encoding="utf-8")
    rolled = svc.rollback(rollback_id)
    assert rolled["ok"] is True
    # Create rollback removes a skill that did not exist before apply.
    assert not (skills / "demo-skill" / "SKILL.md").exists()


def test_known_poll_no_progress_detects_streak():
    from clawagents.loop_detection import LoopDetectionConfig, detect_known_poll_no_progress, hash_tool_call

    cfg = LoopDetectionConfig(warning_threshold=2, critical_threshold=3)
    args = {"command": "sleep 1"}
    h = hash_tool_call("execute", args)
    history = [("execute", h, "same"), ("execute", h, "same")]
    warn = detect_known_poll_no_progress(tool_name="execute", params=args, history=history, config=cfg)
    assert warn and warn.level == "warning"
    history.append(("execute", h, "same"))
    crit = detect_known_poll_no_progress(tool_name="execute", params=args, history=history, config=cfg)
    assert crit and crit.level == "critical"


def test_tool_output_offload_writes_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.tool_output_artifacts import offload_tool_output_if_needed

    big = "x" * 20_000
    inline, path = offload_tool_output_if_needed(
        tool_name="grep",
        tool_use_id="tc1",
        output=big,
        inline_limit=1000,
    )
    assert path is not None
    assert path.is_file()
    assert "truncated" in inline.lower() or "preview" in inline.lower()


def test_compact_tool_results_truncates_tool_messages():
    from clawagents.memory.compact_tool_results import compact_tool_results
    from clawagents.providers.llm import LLMMessage

    messages = [
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="tool", content="a" * 50_000, tool_call_id="1"),
        LLMMessage(role="tool", content="b" * 50_000, tool_call_id="2"),
    ]
    out, modified = compact_tool_results(messages, max_input_tokens=4000)
    assert modified
    assert all(len(m.content) < 50_000 for m in out if m.role == "tool")


def test_sqlite_session_search_and_undo(tmp_path):
    from clawagents.providers.llm import LLMMessage
    from clawagents.session.backends import SQLiteSession

    async def _run():
        session = SQLiteSession("s1", db_path=tmp_path / "s.db")
        await session.add_items([
            LLMMessage(role="user", content="find the failing pytest case"),
            LLMMessage(role="assistant", content="I'll grep the logs"),
        ])
        hits = await session.search("pytest")
        assert hits
        removed = await session.undo_last(1)
        assert len(removed) == 1
        remaining = await session.get_items()
        assert len(remaining) == 1

    asyncio.run(_run())


def test_harness_profile_resolves_for_codex():
    from clawagents.harness_profiles import resolve_harness_profile

    profile = resolve_harness_profile("gpt-5.3-codex-high")
    assert profile is not None
    assert profile.name == "openai-codex"


def test_dry_run_includes_skills_hooks_mcp_preview(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    skills = tmp_path / "skills" / "lint"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# lint", encoding="utf-8")
    hooks = tmp_path / ".clawagents" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "pre_run.py").write_text("pass", encoding="utf-8")
    mcp = tmp_path / ".clawagents" / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"demo": {"command": "echo"}}}), encoding="utf-8")

    from clawagents.dry_run import build_dry_run_preview

    preview = build_dry_run_preview(task="lint files", profile="ollama")
    assert "lint" in preview["skills_preview"]
    assert "pre_run.py" in preview["hooks_preview"]
    assert "demo" in preview["mcp_preview"]


async def _noop_runner(task):
    return {"ok": True}


def test_autopilot_registry_lists_runners():
    from clawagents.autopilot import AutopilotRegistry

    reg = AutopilotRegistry()
    reg.register("demo", _noop_runner)
    assert "demo" in reg.list_runners()
