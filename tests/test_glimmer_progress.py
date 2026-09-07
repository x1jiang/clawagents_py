"""Opt-in progress checkpoints: observation churn is not limited to shell tools."""

from clawagents.graph.loop_tracker import _ToolCallTracker
from clawagents.loop_detection import LoopDetectionConfig


def result(tracker, tool, output="same evidence", *, success=True, **args):
    tracker.record(tool, args)
    return tracker.record_result(tool, args, output, success=success)


def test_mixed_read_search_planning_churn_gets_checkpoint():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    tools = ["read_file", "grep", "think", "write_todos"] * 2
    notices = [result(tracker, tool, path=f"cosmetic-{i}") for i, tool in enumerate(tools)]
    assert not any(notices[:-1])
    assert "Progress checkpoint" in notices[-1]
    assert "read-only" in notices[-1]
    assert "missing" in notices[-1]


def test_new_evidence_delays_but_does_not_disable_budget_checkpoint():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    notices = [result(tracker, "read_file", f"new evidence {i}", path=f"{i}.py") for i in range(16)]
    assert not any(notices[:-1])
    assert "Progress checkpoint" in notices[-1]
    assert "16" in notices[-1]


def test_new_evidence_beyond_legacy_hash_prefix_postpones_checkpoint():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    common = "unchanged header " * 50
    for i in range(7):
        result(tracker, "read_file", common + "old finding", path=f"{i}.py")
    assert result(tracker, "read_file", common + "new finding", path="a.py") is None


def test_changing_planning_text_is_not_new_external_evidence():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    notices = [result(tracker, "think", f"thought {i}", thought=f"plan {i}") for i in range(8)]
    assert notices[-1]


def test_successful_edit_resets_progress_checkpoint():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    for i in range(7):
        assert result(tracker, "read_file", path=f"{i}.py") is None
    assert result(tracker, "edit_file", "edited", path="a.py") is None
    assert result(tracker, "read_file", path="a.py") is None


def test_failed_edit_attempt_does_not_reset_progress_checkpoint():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    for i in range(7):
        result(tracker, "read_file", path=f"{i}.py")
    notice = result(tracker, "edit_file", "Error: old text not found", success=False, path="a.py")
    assert notice and "Progress checkpoint" in notice


def test_successful_delegation_is_not_a_confirmed_edit():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    for i in range(7):
        result(tracker, "read_file", path=f"{i}.py")
    assert result(tracker, "task", "read-only review complete", task="review") is None
    assert result(tracker, "read_file", path="a.py")


def test_checkpoint_budget_stays_bounded_even_after_edits():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    notices = []
    for epoch in range(8):
        result(tracker, "edit_file", f"edited {epoch}", path="a.py")
        notices.extend(result(tracker, "think", f"thought {i}") for i in range(24))
    assert sum(bool(notice) for notice in notices) == 2


def test_progress_checkpoint_does_not_turn_read_repeats_into_hard_stops():
    tracker = _ToolCallTracker(progress_nudge_after=8, hard_limit=3)
    for _ in range(16):
        result(tracker, "read_file", path="a.py")
    assert not tracker.is_hard_looping("read_file", {"path": "a.py"})


def test_default_profile_read_behavior_unchanged():
    tracker = _ToolCallTracker()
    assert not any(result(tracker, "read_file", path=f"{i}.py") for i in range(20))


def test_disabled_loop_detection_has_no_progress_checkpoint():
    tracker = _ToolCallTracker(progress_nudge_after=8, loop_config=LoopDetectionConfig(enabled=False))
    assert not any(result(tracker, "read_file", path=f"{i}.py") for i in range(20))


def test_running_task_poll_is_not_inspection_churn():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    assert not any(result(tracker, "task_wait", "still running", task_id="a") for _ in range(20))


def test_compaction_allows_fresh_evidence_without_refreshing_notice_budget():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    for i in range(7):
        result(tracker, "read_file", path=f"{i}.py")
    tracker.note_context_cleared()
    assert result(tracker, "read_file", path="a.py") is None


def test_failure_escalation_takes_priority_over_progress_notice():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    for i in range(6):
        result(tracker, "read_file", path=f"{i}.py")
    assert result(tracker, "execute", "Error: denied", success=False, command="a") is None
    notice = result(tracker, "execute", "Error: denied", success=False, command="b")
    assert "failed twice" in notice
    assert "Progress checkpoint" not in notice
    assert "Progress checkpoint" in result(tracker, "read_file", path="a.py")


def test_opt_in_replaces_the_legacy_shell_edit_instruction():
    tracker = _ToolCallTracker(progress_nudge_after=8)
    notices = [result(tracker, "execute", f"evidence {i}", command=f"probe {i}") for i in range(16)]
    assert not any(notices[:15])
    assert "Progress checkpoint" in notices[-1]
    assert "edit the code instead" not in notices[-1]
