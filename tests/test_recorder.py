"""Tests for the trajectory recorder."""

import json
from pathlib import Path
from unittest.mock import patch
from clawagents.trajectory.recorder import (
    TrajectoryRecorder,
    ToolCallRecord,
    classify_failure,
    prune_trajectories,
)


def test_classify_failure_format():
    assert classify_failure("read_file", "invalid json in args", None) == "format"


def test_classify_failure_logic():
    assert classify_failure("read_file", "File not found: /tmp/missing.txt", None) == "logic"


def test_classify_failure_unknown():
    assert classify_failure("read_file", None, None) == "unknown"


def test_recorder_basic(tmp_path):
    with patch("clawagents.trajectory.recorder._get_trajectories_dir", return_value=tmp_path):
        rec = TrajectoryRecorder(task="test task", model="test-model")
        rec.record_turn(
            response_text="hello",
            model="test-model",
            tokens_used=100,
            tool_calls=[
                ToolCallRecord(
                    tool_name="read_file",
                    args={"path": "test.py"},
                    success=True,
                    output_preview="file contents...",
                )
            ],
        )
        summary = rec.finalize("success")
        assert summary.total_turns == 1
        assert summary.total_tool_calls == 1
        assert summary.outcome == "success"


def test_run_summary_appends_without_reading_existing_log(tmp_path, monkeypatch):
    with patch("clawagents.trajectory.recorder._get_trajectories_dir", return_value=tmp_path):
        runs_file = tmp_path / "runs.jsonl"
        runs_file.write_text('{"old": true}\n', encoding="utf-8")
        original_read_text = Path.read_text

        def forbidden_read_text(self, *args, **kwargs):
            if self == runs_file:
                raise AssertionError("runs.jsonl should be appended without reading prior content")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", forbidden_read_text)
        rec = TrajectoryRecorder(task="test task", model="test-model")
        rec.finalize("success")

        lines = original_read_text(runs_file, encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"old": True}
        assert json.loads(lines[1])["outcome"] == "success"


def test_prune_trajectories(tmp_path):
    import time
    with patch("clawagents.trajectory.recorder._get_trajectories_dir", return_value=tmp_path):
        old_file = tmp_path / "old_run.jsonl"
        old_file.write_text("{}")
        import os
        os.utime(old_file, (time.time() - 100 * 86400, time.time() - 100 * 86400))

        new_file = tmp_path / "new_run.jsonl"
        new_file.write_text("{}")

        removed = prune_trajectories(max_age_days=30)
        assert removed == 1
        assert new_file.exists()
        assert not old_file.exists()
