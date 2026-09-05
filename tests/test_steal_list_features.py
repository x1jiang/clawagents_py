"""Tests for steal-list features: shadow restore modes, rules, modes, codeact."""

from __future__ import annotations

from pathlib import Path



def test_shadow_checkpoint_modes_and_index(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    (tmp_path / "a.txt").write_text("v1", encoding="utf-8")

    from clawagents.memory.shadow_checkpoint import (
        checkpoint_diff,
        create_checkpoint,
        get_checkpoint_meta,
        list_checkpoints,
        restore_checkpoint,
    )

    sess = tmp_path / "session.jsonl"
    sess.write_text(
        '{"role":"user","content":"hi"}\n{"role":"assistant","content":"yo"}\n{"role":"user","content":"more"}\n',
        encoding="utf-8",
    )

    cp1 = create_checkpoint(
        "first",
        workspace=tmp_path,
        tool="write_file",
        turn_index=0,
        message_count=2,
        session_path=sess,
        phase="pre",
    )
    assert cp1["ok"] and cp1["sha"]
    meta = get_checkpoint_meta(cp1["sha"], workspace=tmp_path)
    assert meta and meta.get("message_count") == 2

    (tmp_path / "a.txt").write_text("v2", encoding="utf-8")
    cp2 = create_checkpoint("second", workspace=tmp_path, phase="post")
    assert cp2["ok"]

    diff = checkpoint_diff(cp1["sha"], cp2["sha"], workspace=tmp_path)
    assert diff["ok"]
    assert any(f.get("path") == "a.txt" for f in diff.get("files") or [])

    restored = restore_checkpoint(cp1["sha"], workspace=tmp_path, mode="both")
    assert restored["ok"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "v1"
    lines = [ln for ln in sess.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2

    rows = list_checkpoints(workspace=tmp_path, limit=5)
    assert rows and rows[0].get("sha")


def test_file_snapshot_preserves_relpath(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "src" / "foo.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("x=1\n", encoding="utf-8")

    from clawagents.config.features import reset, set_overrides
    from clawagents.tools.registry import _snapshot_before_write

    reset()
    set_overrides({"file_snapshots": True, "shadow_checkpoints": False})
    _snapshot_before_write("write_file", {"path": str(nested)})

    snaps = list((tmp_path / ".clawagents" / "snapshots").rglob("foo.py"))
    assert snaps
    assert "src" in snaps[0].parts


def test_rules_discovery_and_budget(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("claude rules", encoding="utf-8")
    rules = tmp_path / ".clawagents" / "rules"
    rules.mkdir(parents=True)
    (rules / "a.md").write_text("rule a", encoding="utf-8")
    (rules / "b.md").write_text("rule b", encoding="utf-8")

    from clawagents.memory.rules import discover_rule_paths, load_rules_text

    paths = discover_rule_paths(tmp_path)
    assert any(p.name == "CLAUDE.md" for p in paths)
    assert any(p.name == "a.md" for p in paths)
    text = load_rules_text(tmp_path, max_chars=40)
    assert text is not None
    assert "truncated" in text.lower() or len(text) <= 60


def test_modes_allowlist_blocks():
    from clawagents.modes import get_mode, make_mode_before_tool

    mode = get_mode("architect")
    assert mode is not None
    hook = make_mode_before_tool(mode)
    denied = hook("write_file", {"path": "x"})
    assert getattr(denied, "allowed", True) is False
    allowed = hook("read_file", {"path": "x"})
    assert getattr(allowed, "allowed", False) is True


def test_codeact_extract_and_run():
    import asyncio
    from clawagents.graph.codeact import extract_code_action, run_code_action
    from clawagents.tools.registry import ToolRegistry, ToolResult

    class _Echo:
        name = "echo"
        description = "echo"
        parameters = {"msg": {"type": "string"}}
        keywords: list[str] = []

        async def execute(self, args):
            return ToolResult(success=True, output=str(args.get("msg")))

    code = extract_code_action('```python\nprint(tools.echo(msg="hi"))\ndone = True\n```')
    assert code and "tools.echo" in code

    reg = ToolRegistry()
    reg.register(_Echo())

    def _run_async(coro):
        return asyncio.run(coro)

    out = run_code_action(code, reg, run_async=_run_async)
    assert "hi" in out["observation"]
    assert out["done"] is True


def test_evals_suite_loader():
    from clawagents.evals_cli import _load_suite, _score_case

    suite = Path(__file__).parent / "evals" / "smoke_suite.json"
    cases = _load_suite(suite)
    assert len(cases) == 1
    scored = _score_case("say hi", "hello-evals", 0, {"contains": "hello-evals"})
    assert scored["passed"] is True


def test_require_approval_blocks_until_handler():
    import asyncio
    from clawagents.graph.agent_loop import _wait_for_tool_approval
    from clawagents.run_context import RunContext

    ctx = RunContext()

    async def _go():
        ok = await _wait_for_tool_approval(
            ctx,
            "c1",
            "write_file",
            {"path": "x"},
            approval_handler=lambda name, args, cid: name == "write_file",
            emit=lambda *a, **k: None,
        )
        assert ok is True
        denied = await _wait_for_tool_approval(
            ctx,
            "c2",
            "write_file",
            {},
            approval_handler=lambda *a, **k: False,
            emit=lambda *a, **k: None,
        )
        assert denied is False

    asyncio.run(_go())
