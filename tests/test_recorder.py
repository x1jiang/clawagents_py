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
