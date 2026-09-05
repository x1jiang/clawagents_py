"""Control-plane crush exemption + post-edit syntax gate."""

from __future__ import annotations



def test_use_skill_output_never_crushed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

    # Larger than aggressive crush floor / target — must stay verbatim.
    body = "# Security review checklist\n" + ("- check item detail\n" * 400)
    assert len(body) > 5000
    out, aid = prepare_tool_output_for_context(
        tool_name="use_skill",
        tool_use_id="skill1",
        output=body,
        workspace=str(tmp_path),
        success=True,
    )
    assert out == body
    assert aid is None
    assert "[Crushed tool output" not in out


def test_list_skills_and_retrieve_also_exempt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

    blob = "skill-a\n" * 2000
    for name in ("list_skills", "retrieve_tool_result"):
        out, aid = prepare_tool_output_for_context(
            tool_name=name,
            tool_use_id="x",
            output=blob,
            workspace=str(tmp_path),
        )
        assert out == blob
        assert aid is None


def test_syntax_gate_catches_duplicate_js_syntax(tmp_path):
    from clawagents.tools.syntax_gate import append_syntax_gate, run_syntax_gate

    bad = tmp_path / "server.js"
    # Duplicate import — plain node --check on .js can miss ESM; gate must catch it.
    bad.write_text(
        "import crypto from 'node:crypto';\nimport crypto from 'node:crypto';\n",
        encoding="utf-8",
    )
    note = run_syntax_gate(bad)
    assert note is not None
    assert "FAILED" in note
    assert "already been declared" in note or "Identifier" in note

    good = tmp_path / "ok.py"
    good.write_text("x = 1\n", encoding="utf-8")
    assert "ok" in (run_syntax_gate(good) or "")

    out = append_syntax_gate(
        "apply_patch",
        {"path": str(bad)},
        "Applied patch to server.js",
        workspace=tmp_path,
    )
    assert "syntax_gate" in out
    assert "FAILED" in out


def test_syntax_gate_skips_non_code(tmp_path):
    from clawagents.tools.syntax_gate import run_syntax_gate

    md = tmp_path / "README.md"
    md.write_text("# hi\n", encoding="utf-8")
    assert run_syntax_gate(md) is None
