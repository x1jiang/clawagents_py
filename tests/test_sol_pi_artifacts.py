"""Paged recall and source-verifiable diagnostic receipt regressions."""

import hashlib
import json
import re

import pytest

from clawagents.memory.content_crush import build_failed_diagnostic_receipt
from clawagents.tool_output_artifacts import (
    load_tool_artifact,
    load_tool_artifact_page,
    prepare_tool_output_for_context,
    store_tool_artifact,
)
from clawagents.tools.retrieve_tool_result import RetrieveToolResultTool


def _store(tmp_path, body):
    return store_tool_artifact(tool_name="execute", tool_use_id="paged", output=body, workspace=tmp_path)[0]


@pytest.mark.parametrize("body", ["", "😀α\r\n" * 900, "é" * 42000 + "\nend", "a\n" * 1000])
def test_pages_roundtrip_without_drop_or_repeat(tmp_path, body):
    aid = _store(tmp_path, body)
    offset = 0
    pieces = []
    while True:
        ok, page, meta = load_tool_artifact_page(aid, workspace=tmp_path, offset=offset, max_chars=137, line_count=17)
        assert ok and len(page) <= 137
        assert meta["offset"] == offset
        assert meta["next_offset"] == offset + len(page)
        pieces.append(page)
        if meta["eof"]:
            break
        assert meta["next_offset"] > offset
        offset = meta["next_offset"]
    assert "".join(pieces) == body


def test_line_start_and_legacy_load(tmp_path):
    body = "zero\nαβ\n" + "x" * 20000 + "\nlast"
    aid = _store(tmp_path, body)
    ok, page, info = load_tool_artifact_page(aid, workspace=tmp_path, line_start=2, line_count=1)
    assert ok and page == "αβ\n" and info["offset"] == 5 and not info["eof"]
    ok, page, info = load_tool_artifact_page(aid, workspace=tmp_path, line_start=99)
    assert ok and page == "" and info["eof"] and info["next_offset"] == len(body)
    assert load_tool_artifact(aid, workspace=tmp_path)[1] == body
    assert "truncated at 10 chars" in load_tool_artifact(aid, workspace=tmp_path, max_chars=10)[1]


@pytest.mark.asyncio
async def test_tool_default_page_and_invalid_ranges(tmp_path):
    aid = _store(tmp_path, "row\n" * 1000)
    tool = RetrieveToolResultTool(str(tmp_path))
    result = await tool.execute({"id": aid})
    header, details, page = result.output.split("\n", 2)
    assert "offset=0 next_offset=1600 eof=false" in header
    assert "continue with next_offset" in details and page == "row\n" * 400
    for args in ({"offset": -1}, {"line_start": 0}, {"line_count": 0}, {"offset": 1, "line_start": 2}, {"offset": "bad"}):
        assert not (await tool.execute({"id": aid, **args})).success


def _diagnostic_log():
    return "\n".join(["pytest session starts"] + [f"test_{i} PASSED" for i in range(900)] + ["FAILED tests/test_api.py::test_save - AssertionError: café", "1 failed, 900 passed"])


def test_failed_diagnostic_receipt_verified_and_retrievable(tmp_path):
    body = _diagnostic_log()
    efficiency = {}
    receipt, aid = prepare_tool_output_for_context(tool_name="execute", tool_use_id="failure", output=body, success=False, command="python -m pytest -q", workspace=tmp_path, efficiency=efficiency)
    assert receipt.startswith("clawagents_evidence_receipt_v1") and len(receipt) < len(body)
    assert "status=failure uncertain=true" in receipt
    assert hashlib.sha256(body.encode()).hexdigest() in receipt
    assert f"source_artifact={aid}" in receipt and "readback=retrieve_tool_result" in receipt
    items = re.findall(r"^- kind=(\w+) line=(\d+) quote_sha256=(\w+) quote=(.*)$", receipt, re.M)
    assert items and any(kind == "failure" for kind, *_ in items)
    for _, line, digest, encoded in items:
        quote = json.loads(encoded)
        assert quote and len(quote) <= 600
        assert quote in body.splitlines()[int(line) - 1]
        assert digest == hashlib.sha256(quote.encode()).hexdigest()
    assert load_tool_artifact(aid, workspace=tmp_path)[1] == body
    assert efficiency["reducer_bytes_saved"] == len(body.encode()) - len(receipt.encode())
    assert prepare_tool_output_for_context(tool_name="execute", tool_use_id="again", output=receipt, workspace=tmp_path)[0] == receipt


@pytest.mark.parametrize("command", [None, "cat file", "echo pytest", "python script.py"])
def test_non_diagnostic_failure_kept(tmp_path, command):
    body = _diagnostic_log()
    text, aid = prepare_tool_output_for_context(tool_name="execute", tool_use_id="no", output=body, success=False, command=command, workspace=tmp_path)
    assert text == body and aid is None


def test_receipt_fallback_and_long_failure_evidence(tmp_path):
    assert build_failed_diagnostic_receipt("ERROR short", artifact_id="id", command="pytest")[0] is None
    body = "x" * 5000 + " AssertionError: required evidence\n" + "ok\n" * 3000
    receipt, reason = build_failed_diagnostic_receipt(body, artifact_id="id", command="pytest")
    assert reason is None and "AssertionError: required evidence" in receipt
    receipt, reason = build_failed_diagnostic_receipt("ok\n" * 3000, artifact_id="id", command="pytest")
    assert receipt is None and reason == "missing-failure-evidence"


def test_page_does_not_follow_external_symlink(tmp_path):
    aid = _store(tmp_path, "inside")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    body = tmp_path / ".clawagents/tool-artifacts" / f"{aid}.txt"
    body.unlink()
    body.symlink_to(outside)
    assert not load_tool_artifact_page(aid, workspace=tmp_path)[0]


@pytest.mark.parametrize("status", ["failed", "succeeded"])
def test_fusion_preserves_confirmation_and_status(tmp_path, status):
    body = _diagnostic_log() if status == "failed" else "test passed\n" * 2000
    prefix = f"Error: exit 1\nOutput:\nApplied edit to src/app.py\n[syntax_gate:ok]\n[then_run:{status}]\n"
    output = prefix + json.dumps({"command_executed": True, "exit_code": int(status == "failed"), "stdout": body, "stderr": ""}) + "\nCommand exited nonzero"
    text, aid = prepare_tool_output_for_context(tool_name="edit_file", tool_use_id="fused", output=output, success=status == "succeeded", command="pytest -q", workspace=tmp_path)
    assert aid and text.startswith(prefix) and len(text) < len(output)
    if status == "failed":
        assert "clawagents_evidence_receipt_v1" in text
    assert body in load_tool_artifact(aid, workspace=tmp_path)[1]
    assert "Command exited nonzero" in load_tool_artifact(aid, workspace=tmp_path)[1]


def test_fusion_skipped_or_small_suffix_is_unchanged(tmp_path):
    for output in ("Applied edit\n[then_run:skipped]\nPermission denied", "Applied edit\n[then_run:failed]\nERROR short"):
        assert prepare_tool_output_for_context(tool_name="edit_file", tool_use_id="small", output=output, success=False, command="pytest", workspace=tmp_path) == (output, None)


def test_receipt_fallback_counters_and_storage_failure(tmp_path, monkeypatch):
    import clawagents.tool_output_artifacts as artifacts

    body = "neutral output\n" * 400
    efficiency = {}
    text, aid = prepare_tool_output_for_context(tool_name="execute", tool_use_id="unknown", output=body, success=False, command="pytest", workspace=tmp_path, efficiency=efficiency)
    assert text == body and aid is None
    assert efficiency["reducer_fallbacks"] == {"missing-failure-evidence": 1}

    def deny(**kwargs):
        raise OSError("cannot archive")

    monkeypatch.setattr(artifacts, "store_tool_artifact", deny)
    body = _diagnostic_log() * 5
    text, aid = prepare_tool_output_for_context(tool_name="execute", tool_use_id="denied", output=body, success=False, command="pytest", workspace=tmp_path, efficiency=efficiency)
    assert text == body and aid is None
    assert efficiency["reducer_fallbacks"]["artifact-write-failed"] == 1


def test_spilled_receipt_uses_complete_source_and_original_handle(tmp_path):
    source = _diagnostic_log()
    aid, _ = store_tool_artifact(tool_name="execute", tool_use_id="spill-source", output=source, workspace=tmp_path, extra_meta={"complete_command_output": True, "command": "pytest"})
    header = f'[Complete command output archived id={aid}; {len(source)} chars. Retrieve with retrieve_tool_result(id="{aid}").]\n'
    preview = header + source[:7000]
    receipt, returned_id = prepare_tool_output_for_context(tool_name="execute", tool_use_id="spill", output=preview, success=False, command="pytest", workspace=tmp_path)
    assert returned_id == aid and "FAILED tests/test_api.py" in receipt
    assert hashlib.sha256(source.encode()).hexdigest() in receipt
    assert len(receipt) < len(preview)


def test_default_character_limit_and_exact_final_page(tmp_path):
    aid = _store(tmp_path, "😀" * 32000)
    ok, first, meta = load_tool_artifact_page(aid, workspace=tmp_path)
    assert ok and len(first) == 16000 and not meta["eof"]
    ok, last, meta = load_tool_artifact_page(aid, workspace=tmp_path, offset=meta["next_offset"])
    assert ok and len(last) == 16000 and meta["eof"]


@pytest.mark.parametrize("failure", ["builder", "storage"])
def test_unexpected_transformation_exception_fails_open(tmp_path, monkeypatch, caplog, failure):
    import clawagents.tool_output_artifacts as artifacts

    efficiency = {"reducer_bytes_saved": 7, "reducer_fallbacks": {"existing": 2}}

    def broken(*args, **kwargs):
        raise ValueError("unexpected transformation failure")

    if failure == "builder":
        monkeypatch.setattr(artifacts, "build_failed_diagnostic_receipt", broken)
        body = _diagnostic_log()
    else:
        monkeypatch.setattr(artifacts, "store_tool_artifact", broken)
        # The rejected receipt increments its local fallback counter before
        # the archive raises; the public boundary must discard that mutation.
        body = "neutral output\n" * 4000
    with caplog.at_level("DEBUG", logger="clawagents.tool_output_artifacts"):
        result = prepare_tool_output_for_context(tool_name="execute", tool_use_id="unexpected", output=body, success=False, command="pytest", workspace=tmp_path, efficiency=efficiency)
    assert result == (body, None)
    assert efficiency == {"reducer_bytes_saved": 7, "reducer_fallbacks": {"existing": 2}}
    assert "preserving original output" in caplog.text
