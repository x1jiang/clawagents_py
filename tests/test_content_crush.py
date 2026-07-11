"""Tests for content crushers + reversible tool artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawagents.memory.content_crush import crush_tool_output, detect_content_kind
from clawagents.tool_output_artifacts import (
    load_tool_artifact,
    prepare_tool_output_for_context,
    store_tool_artifact,
)
from clawagents.tools.retrieve_tool_result import RetrieveToolResultTool


def test_detect_json_array():
    text = json.dumps([{"a": 1}, {"a": 2}, {"a": 3}] * 100)
    assert detect_content_kind(text) == "json"


def test_detect_search_via_tool_name():
    assert detect_content_kind("x:1\ny:2\n" * 100, "grep") == "search"


def test_detect_log():
    lines = [f"INFO ok {i}" for i in range(20)] + [
        "ERROR boom failed",
        "WARNING retry",
        "FATAL crash",
        "ERROR again",
    ] * 5
    assert detect_content_kind("\n".join(lines)) == "log"


def test_crush_json_shrinks_large_array():
    data = [{"id": i, "name": f"item-{i}", "payload": "x" * 50} for i in range(200)]
    text = json.dumps(data)
    result = crush_tool_output(text, tool_name="api")
    assert result.did_crush
    assert result.kind == "json"
    assert result.crushed_chars < result.original_chars
    parsed = json.loads(result.text)
    assert parsed["_crushed"] == "json_array"
    assert parsed["length"] == 200
    assert len(parsed["sample"]) <= 3


def test_crush_log_keeps_errors():
    lines = [f"INFO line {i}" for i in range(500)]
    lines[100] = "ERROR something broke"
    lines[200] = "FATAL hard failure"
    text = "\n".join(lines)
    result = crush_tool_output(text, tool_name="execute")
    assert result.did_crush
    assert "ERROR something broke" in result.text
    assert "FATAL hard failure" in result.text
    assert result.crushed_chars < result.original_chars


def test_crush_search_omits_middle_lines():
    text = "\n".join(f"src/foo.py:{i}: match content blob {i} " + ("x" * 20) for i in range(400))
    result = crush_tool_output(text, tool_name="grep")
    assert result.did_crush
    assert "match content blob 0" in result.text
    assert "omitted" in result.text.lower()


def test_crush_html():
    html = (
        "<html><head><title>Docs</title></head><body>"
        + "<h1>Intro</h1><p>" + ("lorem " * 500) + "</p>"
        + "<script>alert(1)</script></body></html>"
    )
    result = crush_tool_output(html, tool_name="web_fetch")
    assert result.did_crush
    assert result.kind == "html"
    assert "Docs" in result.text
    assert "alert(1)" not in result.text


def test_crush_diff():
    lines = ["diff --git a/x.py b/x.py", "--- a/x.py", "+++ b/x.py", "@@ -1,3 +1,4 @@"]
    for i in range(200):
        lines.append(f"+added line {i} " + ("x" * 30))
        lines.append(f"-removed line {i}")
    text = "\n".join(lines)
    result = crush_tool_output(text, tool_name="execute")
    assert result.did_crush
    assert result.kind == "diff"
    assert "diff --git" in result.text


def test_crush_pytest_output():
    lines = ["============================= test session starts =============================="]
    lines += [f"tests/test_x.py::test_{i} PASSED" + (" detail" * 10) for i in range(120)]
    lines += [
        "tests/test_x.py::test_boom FAILED",
        "AssertionError: expected True",
        "E   assert False",
        "=========================== short test summary info ============================",
        "FAILED tests/test_x.py::test_boom",
    ]
    text = "\n".join(lines)
    assert len(text) > 2000
    result = crush_tool_output(text, tool_name="execute")
    assert result.did_crush
    assert result.kind == "test"
    assert "FAILED" in result.text


def test_small_output_not_crushed():
    text = "hello world"
    result = crush_tool_output(text)
    assert not result.did_crush
    assert result.text == text


def test_prepare_and_retrieve_roundtrip(tmp_path: Path):
    big = json.dumps([{"n": i, "blob": "y" * 40} for i in range(300)])
    prompt, aid = prepare_tool_output_for_context(
        tool_name="fetch",
        tool_use_id="call_abc",
        output=big,
        workspace=tmp_path,
    )
    assert aid is not None
    assert "Crushed" in prompt or "truncated" in prompt.lower()
    assert aid in prompt

    ok, full, meta = load_tool_artifact(aid, workspace=tmp_path)
    assert ok
    assert full == big
    assert meta is not None
    assert meta["tool_name"] == "fetch"


@pytest.mark.asyncio
async def test_retrieve_tool_result_tool(tmp_path: Path):
    body = "full secret output " + ("z" * 3000)
    aid, _ = store_tool_artifact(
        tool_name="execute",
        tool_use_id="tc1",
        output=body,
        kind="prose",
        workspace=tmp_path,
    )
    tool = RetrieveToolResultTool(workspace=str(tmp_path))
    result = await tool.execute({"id": aid})
    assert result.success
    assert "full secret output" in result.output


def test_compact_tool_results_honors_headroom():
    from clawagents.memory.compact_tool_results import compact_tool_results
    from clawagents.providers.llm import LLMMessage

    msgs = [
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="tool", content="x" * 50_000, tool_call_id="t1"),
        LLMMessage(role="tool", content="y" * 50_000, tool_call_id="t2"),
    ]
    tight, mod_a = compact_tool_results(
        msgs, max_input_tokens=8_000, headroom_ratio=0.3
    )
    loose, mod_b = compact_tool_results(
        msgs, max_input_tokens=8_000, headroom_ratio=0.9
    )
    assert mod_a and mod_b
    assert len(str(tight[1].content)) <= len(str(loose[1].content))
