"""Strict SEARCH/REPLACE parser — empty REPLACE + fence corruption guard."""

from __future__ import annotations

import asyncio
from pathlib import Path

from clawagents.sandbox.local import LocalBackend
from clawagents.tools.apply_patch import (
    ApplyPatchTool,
    _parse_search_replace_hunks,
    _apply_search_replace,
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
