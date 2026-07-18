"""Artifact load/search must not follow meta paths outside the storage dir."""

from __future__ import annotations

import json
from pathlib import Path

from clawagents.tool_output_artifacts import (
    load_tool_artifact,
    search_tool_artifacts,
    store_tool_artifact,
)


def test_store_and_load_uses_id_derived_body(tmp_path: Path):
    aid, body = store_tool_artifact(
        tool_name="execute",
        tool_use_id="call_abc",
        output="SECRET_FULL_OUTPUT_" + ("x" * 100),
        workspace=tmp_path,
    )
    assert body.parent == tmp_path / ".clawagents" / "tool-artifacts"
    assert body.name == f"{aid}.txt"
    meta = json.loads((body.parent / f"{aid}.meta.json").read_text())
    assert meta["path"] == body.name  # relative, not absolute escape

    ok, text, _ = load_tool_artifact(aid, workspace=tmp_path)
    assert ok
    assert "SECRET_FULL_OUTPUT_" in text


def test_load_ignores_meta_path_outside_artifact_dir(tmp_path: Path):
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("SHOULD_NOT_LEAK", encoding="utf-8")
    art_dir = tmp_path / ".clawagents" / "tool-artifacts"
    art_dir.mkdir(parents=True)
    aid = "evil_escape"
    meta = {
        "id": aid,
        "tool_name": "execute",
        "tool_use_id": aid,
        "kind": "raw",
        "chars": 10,
        "path": str(outside),  # absolute path outside storage
    }
    (art_dir / f"{aid}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # No body under artifact dir
    ok, text, _ = load_tool_artifact(aid, workspace=tmp_path)
    assert ok is False
    assert "SHOULD_NOT_LEAK" not in text


def test_search_ignores_escaped_meta_path(tmp_path: Path):
    outside = tmp_path / "leaky.txt"
    outside.write_text("unique_needle_escape_xyz", encoding="utf-8")
    art_dir = tmp_path / ".clawagents" / "tool-artifacts"
    art_dir.mkdir(parents=True)
    meta = {
        "id": "scan_me",
        "tool_name": "execute",
        "path": str(outside),
    }
    (art_dir / "scan_me.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    hits = search_tool_artifacts("unique_needle_escape_xyz", workspace=tmp_path)
    assert hits == []
