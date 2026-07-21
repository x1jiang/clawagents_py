"""Strict SEARCH/REPLACE parser — empty REPLACE + fence corruption guard."""

from __future__ import annotations

import asyncio
from pathlib import Path

from clawagents.sandbox.local import LocalBackend
from clawagents.tools.apply_patch import (
    ApplyPatchTool,
    _parse_search_replace_hunks,
    _apply_search_replace,
    _nearest_search_hint,
)


def test_parse_empty_replace_deletion():
    patch = (
        "<<<<<<< SEARCH\n"
        "GONE\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    hunks, msg = _parse_search_replace_hunks(patch)
    assert msg == "ok"
    assert hunks == [("GONE", "")]


def test_parse_two_hunks_second_empty_replace():
    # The bug class: empty REPLACE must not swallow the next fence markers.
    patch = (
        "<<<<<<< SEARCH\n"
        "A\n"
        "=======\n"
        "B\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "C\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    hunks, msg = _parse_search_replace_hunks(patch)
    assert msg == "ok"
    assert hunks == [("A", "B"), ("C", "")]


def test_delete_applies():
    ok, out, _ = _apply_search_replace("keep\nGONE\nkeep2\n", "GONE", "")
    assert ok
    assert out == "keep\nkeep2\n"


def test_refuse_writing_fence_markers_into_file(tmp_path: Path):
    f = tmp_path / "deploy.sh"
    f.write_text("echo ok\n", encoding="utf-8")
    # Malformed-looking content that a buggy regex might inject — we simulate
    # a patch whose REPLACE intentionally contains a fence (must refuse).
    patch = (
        "<<<<<<< SEARCH\n"
        "echo ok\n"
        "=======\n"
        "echo ok\n"
        "<<<<<<< SEARCH\n"
        ">>>>>>> REPLACE\n"
    )
    # Parser should reject unexpected fence inside REPLACE
    hunks, msg = _parse_search_replace_hunks(patch)
    assert hunks is None
    assert "unexpected fence" in msg


def test_apply_patch_returns_diff(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello world\n", encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "<<<<<<< SEARCH\n"
        "hello world\n"
        "=======\n"
        "hello there\n"
        ">>>>>>> REPLACE\n"
    )
    result = asyncio.run(tool.execute({"path": "a.txt", "patch": patch}))
    assert result.success, result.error
    assert "hello there" in f.read_text(encoding="utf-8")
    assert "@@" in (result.output or "") or "hello there" in (result.output or "")


def test_multi_hunk_failure_reports_index_and_preserves_atomicity(tmp_path: Path):
    f = tmp_path / "config.json"
    original = '{\n  "first": false,\n  "second": false\n}\n'
    f.write_text(original, encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "<<<<<<< SEARCH\n"
        '  "first": false,\n'
        "=======\n"
        '  "first": true,\n'
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        '  "missing": false\n'
        "=======\n"
        '  "missing": true\n'
        ">>>>>>> REPLACE\n"
    )

    result = asyncio.run(tool.execute({"path": "config.json", "patch": patch}))

    assert result.success is False
    assert "hunk 2/2" in result.error.lower()
    assert "1 earlier hunk" in result.error.lower()
    assert "no changes written" in result.error.lower()
    assert "do not resend this patch unchanged" in result.error.lower()
    assert "one localized hunk per call" in result.error.lower()
    assert '"missing": false' in result.error
    assert f.read_text(encoding="utf-8") == original


def test_ambiguous_search_routes_to_hashline_without_retry(tmp_path: Path):
    f = tmp_path / "routes.js"
    f.write_text("close();\nkeep();\nclose();\n", encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "<<<<<<< SEARCH\nclose();\n=======\ndone();\n>>>>>>> REPLACE\n"
    )

    result = asyncio.run(tool.execute({"path": "routes.js", "patch": patch}))

    assert result.success is False
    assert "matches 2 locations" in result.error
    assert "Do not retry the same patch" in result.error
    assert "hashline_grep" in result.error


def test_apply_patch_refuses_invalid_json_before_write(tmp_path: Path):
    f = tmp_path / "config.json"
    original = '{\n  "enabled": false\n}\n'
    f.write_text(original, encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "<<<<<<< SEARCH\n"
        '  "enabled": false\n'
        "=======\n"
        '  \\"enabled\\": true\\n\n'
        ">>>>>>> REPLACE\n"
    )

    result = asyncio.run(tool.execute({"path": "config.json", "patch": patch}))

    assert result.success is False
    assert "invalid JSON" in result.error
    assert "literal escape" in result.error
    assert f.read_text(encoding="utf-8") == original


def test_nearest_hint_does_not_round_long_line_mismatch_to_100_percent():
    body = "Three target PDFs tested. " + ("validation detail " * 15)
    search = f"- `billing_img` | {body} |"
    content = f"| `billing_img` | {body} |\n"

    hint = _nearest_search_hint(content, search)

    assert "similarity 100%" not in hint
    assert "First difference at column 1" in hint
    assert "list marker" in hint
    assert "Markdown table row" in hint


def test_markdown_table_mismatch_reports_failed_hunk_and_stays_atomic(tmp_path: Path):
    f = tmp_path / "README.md"
    original = (
        "## Current validation\n\n"
        "| Profile | Result |\n"
        "| --- | --- |\n"
        "| `billing_img` | Three target PDFs tested. |\n\n"
        "- output filenames have no patient identifiers\n"
    )
    f.write_text(original, encoding="utf-8")
    tool = ApplyPatchTool(LocalBackend(root=str(tmp_path)))
    patch = (
        "<<<<<<< SEARCH\n## Current validation\n=======\n"
        "## Production naming\n\n## Current validation\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "- `billing_img` | Three target PDFs tested. |\n"
        "=======\n"
        "- `billing_img` | Three target PDFs tested and validated. |\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "- output filenames have no patient identifiers\n"
        "=======\n"
        "- production filenames follow approved naming\n"
        ">>>>>>> REPLACE\n"
    )

    result = asyncio.run(tool.execute({"path": "README.md", "patch": patch}))

    assert result.success is False
    assert "Hunk 2/3" in result.error
    assert "list marker" in result.error
    assert "Markdown table row" in result.error
    assert f.read_text(encoding="utf-8") == original
