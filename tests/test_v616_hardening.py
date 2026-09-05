"""Regression tests for 6.16 security / Goal hardening."""

from __future__ import annotations

from pathlib import Path



def test_subagent_registry_inherits_permission_engine(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_PERMISSION_RULES", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.tools.permissions import PermissionRule, load_permission_engine
    from clawagents.tools.registry import ToolRegistry
    from clawagents.tools.subagent import _registry_for_workspace

    parent = ToolRegistry()
    engine = load_permission_engine(tmp_path)
    engine.add_rule(
        PermissionRule(
            tool="execute",
            arg_pattern="*rm -rf *",
            decision="deny",
            priority=100,
            message="no",
        )
    )
    parent._permission_engine = engine  # type: ignore[attr-defined]

    child = _registry_for_workspace(parent, str(tmp_path))
    assert getattr(child, "_permission_engine", None) is engine
    ok, msg = child._permission_engine.gate("execute", {"command": "rm -rf /tmp/x"})
    assert ok is False
    assert msg  # deny message from inherited engine / defaults


def test_hunk_accept_all_rejects_path_traversal(tmp_path: Path):
    from clawagents.memory.attributed_hunks import accept_all, HunkStore

    store = HunkStore.load(tmp_path)
    store.baselines["safe.txt"] = "a\n"
    store.save()

    bad = accept_all("../outside.txt", workspace=tmp_path)
    assert bad.get("ok") is False
    assert "escape" in str(bad.get("error", "")).lower() or ".." in str(bad.get("error", ""))

    abs_bad = accept_all("/etc/passwd", workspace=tmp_path)
    assert abs_bad.get("ok") is False


def test_permission_ask_handler_allows(monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_PERMISSION_RULES", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.tools.permissions import PermissionEngine, PermissionRule

    engine = PermissionEngine()
    engine.add_rule(
        PermissionRule(tool="execute", arg_pattern="*sudo *", decision="ask", priority=50)
    )
    ok, _ = engine.gate("execute", {"command": "sudo ls"})
    assert ok is False

    engine.ask_handler = lambda tool, args, msg: True
    ok2, _ = engine.gate("execute", {"command": "sudo ls"})
    assert ok2 is True


def test_goal_mode_injects_nudge(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("CLAW_ATLAS", raising=False)
    from clawagents.agent import create_claw_agent

    agent = create_claw_agent("gpt-4o-mini", goal_mode=True, streaming=False)
    assert agent.system_prompt
    assert "start_goal" in agent.system_prompt
