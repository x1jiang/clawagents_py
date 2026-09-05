"""Tests for peer-inspired improvements (ledger, repo map, patch, memory, checkpoints)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clawagents.memory.context_ledger import (
    load_ledger_preamble,
    record_commit_ledger,
    rehydrate_from_git,
)
from clawagents.memory.core_memory import (
    core_memory_append,
    core_memory_replace,
    load_core_memory,
)
from clawagents.memory.facts import add_fact, list_facts, live_facts_preamble
from clawagents.memory.repo_map import build_repo_map
from clawagents.memory.shadow_checkpoint import (
    create_checkpoint,
    list_checkpoints,
    restore_checkpoint,
)
from clawagents.tools.apply_patch import _apply_search_replace, _apply_unified_diff
from clawagents.events.envelope import map_legacy_event, wrap_event
from clawagents.tools.auto_verify import detect_verify_commands
from clawagents.tools.context_tools import load_plan_preamble, WritePlanTool


def _git_init(tmp: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp, check=True, capture_output=True)


def test_apply_search_replace():
    ok, out, msg = _apply_search_replace("hello world\n", "world", "there")
    assert ok and "hello there" in out


def test_apply_unified_diff_simple():
    src = "a\nb\nc\n"
    patch = "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    ok, out, msg = _apply_unified_diff(src, patch)
    assert ok, msg
    assert out.splitlines() == ["a", "B", "c"]


def test_repo_map_ranks_symbols(tmp_path: Path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n\nclass Bar:\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from a import foo\n\ndef baz():\n    foo()\n", encoding="utf-8")
    text = build_repo_map(tmp_path, max_chars=2000, mentioned={"foo"})
    assert "Repo Map" in text
    assert "foo" in text


def test_core_memory_edit(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "Core Memory" in load_core_memory(workspace=tmp_path)
    ok, _ = core_memory_replace("project", "(fill with durable project facts)", "Uses SQLite", workspace=tmp_path)
    assert ok
    ok, _ = core_memory_append("human", "Prefers concise answers", workspace=tmp_path)
    assert ok
    body = load_core_memory(workspace=tmp_path)
    assert "Uses SQLite" in body
    assert "concise" in body


def test_facts_supersede(tmp_path: Path):
    a = add_fact("Use Postgres", workspace=tmp_path)
    b = add_fact("Use SQLite instead", workspace=tmp_path, supersedes=a.id)
    live = list_facts(workspace=tmp_path, live_only=True)
    assert all(f.id != a.id for f in live)
    assert any(f.id == b.id for f in live)
    assert "SQLite" in live_facts_preamble(workspace=tmp_path)


def test_context_ledger_roundtrip(tmp_path: Path):
    _git_init(tmp_path)
    f = tmp_path / "mod.py"
    f.write_text("def hello():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "mod.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: add hello helper"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    entry = record_commit_ledger(workspace=tmp_path)
    assert entry is not None
    assert "hello" in entry.subject.lower() or entry.signatures
    preamble = load_ledger_preamble(workspace=tmp_path)
    assert "Context Ledger" in preamble
    ok, text = rehydrate_from_git(entry.sha, workspace=tmp_path, path="mod.py")
    assert ok
    assert "def hello" in text


def test_shadow_checkpoint_restore(tmp_path: Path):
    (tmp_path / "x.txt").write_text("v1", encoding="utf-8")
    cp1 = create_checkpoint("one", workspace=tmp_path)
    assert cp1["ok"] and cp1["sha"]
    (tmp_path / "x.txt").write_text("v2", encoding="utf-8")
    cp2 = create_checkpoint("two", workspace=tmp_path)
    assert cp2["ok"]
    rows = list_checkpoints(workspace=tmp_path, limit=5)
    assert len(rows) >= 2
    restored = restore_checkpoint(cp1["sha"], workspace=tmp_path)
    assert restored["ok"]
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "v1"


@pytest.mark.asyncio
async def test_write_plan_tool(tmp_path: Path):
    tool = WritePlanTool(workspace=str(tmp_path))
    r = await tool.execute({"content": "## Steps\n1. Do thing\n"})
    assert r.success
    assert "Active Plan" in load_plan_preamble(tmp_path)
    assert "Do thing" in load_plan_preamble(tmp_path)


def test_event_envelope():
    ev = wrap_event("compaction", "compact_progress", {"phase": "start"})
    assert ev["schema_version"] == "1"
    assert ev["kind"] == "compaction"
    mapped = map_legacy_event("tool_result", {"ok": True})
    assert mapped["kind"] == "observation"


def test_detect_verify_commands(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    cmds = detect_verify_commands(tmp_path)
    assert any("pytest" in c for c in cmds)
