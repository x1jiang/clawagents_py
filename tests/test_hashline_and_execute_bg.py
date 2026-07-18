"""Grok-inspired hashline + execute is_background (additive, feature-gated)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from clawagents.tools.hashline import (
    ChunkFingerprint,
    ParsedAnchor,
    apply_edits,
    encode_hash,
    format_hashline_content,
    line_hash,
    split_lines,
)
from clawagents.tools.filesystem import EditFileTool, _nearest_edit_hint


def test_line_hash_whitespace_normalization():
    assert line_hash("    let x = 1;") == line_hash("  let x = 1;")
    assert line_hash("let x = 1;") == line_hash("let  x  =  1;")
    assert line_hash("return x") != line_hash("returnx")
    assert line_hash("") == line_hash("   ")


def test_encode_hash_letters():
    h = encode_hash(line_hash("hello"), 3)
    assert len(h) == 3
    assert h.isalpha() and h.islower()


def test_parsed_anchor():
    a = ParsedAnchor.parse("22:abc:rst")
    assert a is not None
    assert a.line == 22 and a.local == "abc" and a.context == "rst"
    assert ParsedAnchor.parse("0:abc") is None
    assert ParsedAnchor.parse("bad") is None


def test_format_and_replace_roundtrip():
    content = "fn main() {\n    let x = 1;\n    let y = 2;\n}\n"
    scheme = ChunkFingerprint()
    anchors = scheme.generate_anchors(split_lines(content))
    # Replace the `let x` line
    x_anchor = next(a for a in anchors if "let x" in split_lines(content)[a.line - 1])
    new_content, result = apply_edits(
        content,
        [{"op": "replace", "anchor": x_anchor.render(), "content": "    let x = 42;"}],
        path="main.rs",
        scheme=scheme,
    )
    assert new_content is not None
    assert result["status"] == "ok"
    assert "let x = 42;" in new_content
    assert "snippet" in result and "→" in result["snippet"]


def test_stale_anchor_returns_recovery():
    content = "a\nb\nc\n"
    scheme = ChunkFingerprint()
    anchors = scheme.generate_anchors(split_lines(content))
    # Mutate file under the model
    mutated = "a\nB\nc\n"
    new_content, result = apply_edits(
        mutated,
        [{"op": "replace", "anchor": anchors[1].render(), "content": "bb"}],
        path="t.txt",
        scheme=scheme,
    )
    assert new_content is None
    assert result["status"] == "error"
    assert result["error"] in {"anchor_stale", "ambiguous_anchor"}
    assert "context" in result


def test_atomic_batch_none_applied_on_bad_second():
    content = "one\ntwo\nthree\n"
    scheme = ChunkFingerprint()
    anchors = scheme.generate_anchors(split_lines(content))
    new_content, result = apply_edits(
        content,
        [
            {"op": "replace", "anchor": anchors[0].render(), "content": "ONE"},
            {"op": "replace", "anchor": "99:zzz:zzz", "content": "X"},
        ],
        path="t.txt",
        scheme=scheme,
    )
    assert new_content is None
    assert "none of the edits were applied" in result["message"]


def test_insert_after_eof_and_bof():
    content = "hello\n"
    _, result = apply_edits(
        content,
        [{"op": "insert_after", "anchor": "0:", "content": "first"}],
        path="t.txt",
    )
    assert result["status"] == "ok"
    new_content, result2 = apply_edits(
        "hello\n",
        [{"op": "insert_after", "anchor": "EOF", "content": "tail"}],
        path="t.txt",
    )
    assert result2["status"] == "ok"
    assert new_content is not None
    assert "tail" in new_content


@pytest.mark.asyncio
async def test_hashline_tools_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_HASHLINE_TOOLS", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.hashline import create_hashline_tools

    f = tmp_path / "demo.py"
    f.write_text("def f():\n    return 1\n", encoding="utf-8")
    sb = LocalBackend(root=str(tmp_path))
    tools = create_hashline_tools(sb)
    assert [t.name for t in tools] == ["hashline_read", "hashline_grep", "hashline_edit"]
    read_t, grep_t, edit_t = tools
    r = await read_t.execute({"path": "demo.py", "limit": 20})
    assert r.success
    assert "→" in r.output
    # Grab first content line anchor
    body = r.output.split("\n", 1)[1]
    first = body.splitlines()[0]
    anchor = first.split("→", 1)[0]
    er = await edit_t.execute(
        {
            "path": "demo.py",
            "edits": [{"op": "replace", "anchor": anchor, "content": "def f():"}],
        }
    )
    assert er.success, er.error
    assert "def f():" in f.read_text(encoding="utf-8")

    gr = await grep_t.execute({"pattern": r"def f", "path": "."})
    assert gr.success, gr.error
    assert "→" in gr.output
    assert "hashline_edit" in gr.output


def test_nearest_edit_hint():
    content = "alpha\nhello world\nomega\n"
    hint = _nearest_edit_hint(content, "hello wrld")
    assert "Nearest similar line" in hint


@pytest.mark.asyncio
async def test_edit_file_miss_has_hint(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello world\n", encoding="utf-8")
    from clawagents.sandbox.local import LocalBackend

    tool = EditFileTool(LocalBackend(root=str(tmp_path)))
    r = await tool.execute({"path": "a.txt", "target": "hello wrld", "replacement": "x"})
    assert not r.success
    assert "Nearest similar" in (r.error or "")
    assert "user may have changed" in (r.error or "").lower()


@pytest.mark.asyncio
async def test_edit_file_create_if_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_EDIT_FILE_CREATE_EMPTY", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend

    tool = EditFileTool(LocalBackend(root=str(tmp_path)))
    assert "create_if_missing" in tool.parameters
    r = await tool.execute(
        {
            "path": "new.txt",
            "target": "",
            "replacement": "created\n",
            "create_if_missing": True,
        }
    )
    assert r.success, r.error
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "created\n"

    # Empty target on existing path still refused (even if empty file)
    bad = await tool.execute(
        {
            "path": "new.txt",
            "target": "",
            "replacement": "overwrite",
            "create_if_missing": True,
        }
    )
    assert not bad.success
    assert "existing" in (bad.error or "").lower()

    # String "false" must not enable create_if_missing
    (tmp_path / "emptyish.txt").write_text("", encoding="utf-8")
    no = await tool.execute(
        {
            "path": "other.txt",
            "target": "",
            "replacement": "x",
            "create_if_missing": "false",
        }
    )
    assert not no.success
    assert "non-empty" in (no.error or "").lower()


@pytest.mark.asyncio
async def test_execute_is_background(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool
    from clawagents.tools.background_task import create_background_task_tools

    tool = ExecTool(LocalBackend())
    r = await tool.execute(
        {
            "command": "echo claw-bg-ok",
            "description": "smoke bg",
            "is_background": True,
        }
    )
    assert r.success, r.error
    raw = r.output
    start = raw.find("{")
    payload = json.loads(raw[start:] if start >= 0 else raw)
    assert payload["backgrounded"] is True
    job_id = payload["job_id"]

    status_t = next(t for t in create_background_task_tools() if t.name == "task_status")
    # Wait briefly for completion
    for _ in range(50):
        st = await status_t.execute({"job_id": job_id})
        data = json.loads(st.output)
        if not data.get("running"):
            break
        await asyncio.sleep(0.05)
    out_t = next(t for t in create_background_task_tools() if t.name == "task_output")
    out = await out_t.execute({"job_id": job_id})
    assert "claw-bg-ok" in out.output


def test_resolve_block_until_ms_rejects_negative():
    from clawagents.tools.exec import DEFAULT_TIMEOUT_MS, _resolve_block_until_ms

    assert _resolve_block_until_ms({"block_until_ms": 0}) == (0, True)
    assert _resolve_block_until_ms({"block_until_ms": -5}) == (DEFAULT_TIMEOUT_MS, False)
    assert _resolve_block_until_ms({"block_until_ms": 50})[0] == 100
    assert _resolve_block_until_ms({"timeout": -1}) == (DEFAULT_TIMEOUT_MS, False)


@pytest.mark.asyncio
async def test_execute_block_until_ms_zero_backgrounds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    tool = ExecTool(LocalBackend())
    r = await tool.execute(
        {"command": "echo via-block-until", "block_until_ms": 0, "description": "bg"}
    )
    assert r.success, r.error
    start = r.output.find("{")
    payload = json.loads(r.output[start:])
    assert payload["backgrounded"] is True
    assert payload["job_id"]


@pytest.mark.asyncio
async def test_execute_streaming_emits_tool_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_SHELL_SESSION", "0")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_STREAMING", "1")
    monkeypatch.setenv("CLAW_FEATURE_RTK_WRAP", "0")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.exec import ExecTool

    events: list[tuple[str, dict]] = []

    class Ctx:
        def on_event(self, kind: str, payload: dict) -> None:
            events.append((kind, payload))

    tool = ExecTool(LocalBackend(root=str(tmp_path)))
    r = await tool.execute(
        {"command": "printf 'stream-ok\\n'", "timeout": 5000},
        run_context=Ctx(),
    )
    assert r.success, r.error
    assert "stream-ok" in r.output
    progress = [p for k, p in events if k == "tool_progress"]
    assert progress, "expected tool_progress events"
    assert any("stream-ok" in (p.get("delta") or "") for p in progress)
