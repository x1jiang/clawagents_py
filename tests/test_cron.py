"""Hermetic tests for clawagents.cron.

These tests:
- Pin a frozen clock via ``set_clock`` so schedule arithmetic is
  deterministic regardless of wall-clock time.
- Redirect ``CLAWAGENTS_HOME`` to a tmpdir per-test so jobs never
  touch the user's real cron storage.
- Never sleep on real time — the scheduler is exercised through
  :meth:`Scheduler.tick`, which is fully synchronous in effect.
- Do not import ``croniter``; cron-expression tests are skipped when
  the optional extra is missing.

Mirrors ``clawagents/src/cron/cron.test.ts``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clawagents import cron as cron_module  # noqa: E402
from clawagents.cron import jobs as jobs_mod  # noqa: E402


_FROZEN = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAWAGENTS_PROFILE", "test")

    current = {"now": _FROZEN}
    jobs_mod.set_clock(lambda: current["now"])
    yield current
    jobs_mod.reset_clock()


def _advance(state, seconds=0, minutes=0, hours=0):
    state["now"] = state["now"] + timedelta(seconds=seconds, minutes=minutes, hours=hours)


# ── parse_schedule ───────────────────────────────────────────────────


def test_parse_duration_minutes_hours_days():
    assert cron_module.parse_duration("30m") == 30
    assert cron_module.parse_duration("2h") == 120
    assert cron_module.parse_duration("1d") == 1440


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        cron_module.parse_duration("forever")


def test_parse_schedule_oneshot_relative():
    parsed = cron_module.parse_schedule("30m")
    assert parsed["kind"] == "once"
    assert "run_at" in parsed
    expected = (_FROZEN + timedelta(minutes=30)).isoformat()
    assert parsed["run_at"] == expected


def test_parse_schedule_interval():
    parsed = cron_module.parse_schedule("every 15m")
    assert parsed["kind"] == "interval"
    assert parsed["minutes"] == 15


def test_parse_schedule_iso_timestamp():
    parsed = cron_module.parse_schedule("2026-12-31T09:00:00+00:00")
    assert parsed["kind"] == "once"
    assert parsed["run_at"].startswith("2026-12-31T09:00:00")


def test_parse_schedule_invalid():
    with pytest.raises(ValueError):
        cron_module.parse_schedule("nonsense")


def test_parse_schedule_cron_requires_extra(monkeypatch):
    monkeypatch.setattr(jobs_mod, "CRONITER_AVAILABLE", False)
    monkeypatch.setattr(jobs_mod, "_croniter", None)
    with pytest.raises(ValueError, match="croniter"):
        cron_module.parse_schedule("0 9 * * *")


@pytest.mark.skipif(
    not jobs_mod.CRONITER_AVAILABLE, reason="croniter extra not installed"
)
def test_parse_schedule_cron_expression(_isolate):
    parsed = cron_module.parse_schedule("0 9 * * *")
    assert parsed["kind"] == "cron"
    assert parsed["expr"] == "0 9 * * *"
    nxt = cron_module.compute_next_run(parsed)
    assert nxt is not None
    assert datetime.fromisoformat(nxt) > _FROZEN


# ── compute_next_run ─────────────────────────────────────────────────


def test_compute_next_run_oneshot_returns_run_at_when_future():
    schedule = cron_module.parse_schedule("30m")
    nxt = cron_module.compute_next_run(schedule)
    assert nxt == schedule["run_at"]


def test_compute_next_run_oneshot_returns_none_after_run():
    schedule = cron_module.parse_schedule("30m")
    last = (_FROZEN + timedelta(minutes=30)).isoformat()
    assert cron_module.compute_next_run(schedule, last_run_at=last) is None


def test_compute_next_run_interval_first_run_in_future():
    schedule = cron_module.parse_schedule("every 15m")
    nxt = cron_module.compute_next_run(schedule)
    assert nxt == (_FROZEN + timedelta(minutes=15)).isoformat()


def test_compute_next_run_interval_advances_from_last_run(_isolate):
    schedule = cron_module.parse_schedule("every 15m")
    last = _FROZEN.isoformat()
    nxt = cron_module.compute_next_run(schedule, last_run_at=last)
    assert nxt == (_FROZEN + timedelta(minutes=15)).isoformat()


# ── CRUD ─────────────────────────────────────────────────────────────


def test_create_and_get_job(_isolate):
    job = cron_module.create_job("hello", "30m", name="greeting")
    assert job["id"]
    assert job["name"] == "greeting"
    assert job["enabled"] is True
    assert job["state"] == "scheduled"
    assert job["next_run_at"] == job["schedule"]["run_at"]
    assert job["repeat"]["times"] == 1  # auto for one-shot

    fetched = cron_module.get_job(job["id"])
    assert fetched is not None
    assert fetched["name"] == "greeting"


def test_list_jobs_filters_disabled(_isolate):
    a = cron_module.create_job("active", "30m")
    b = cron_module.create_job("paused", "30m")
    cron_module.pause_job(b["id"], reason="quiet hours")

    visible = cron_module.list_jobs()
    visible_ids = {j["id"] for j in visible}
    assert a["id"] in visible_ids
    assert b["id"] not in visible_ids

    all_jobs = cron_module.list_jobs(include_disabled=True)
    assert {j["id"] for j in all_jobs} == {a["id"], b["id"]}


def test_pause_resume_round_trip(_isolate):
    job = cron_module.create_job("ping", "every 1h")
    cron_module.pause_job(job["id"], reason="testing")

    paused = cron_module.get_job(job["id"])
    assert paused["enabled"] is False
    assert paused["state"] == "paused"
    assert paused["paused_reason"] == "testing"

    cron_module.resume_job(job["id"])
    resumed = cron_module.get_job(job["id"])
    assert resumed["enabled"] is True
    assert resumed["state"] == "scheduled"
    assert resumed["paused_at"] is None


def test_trigger_job_sets_next_run_now(_isolate):
    job = cron_module.create_job("ping", "every 1h")
    cron_module.trigger_job(job["id"])
    triggered = cron_module.get_job(job["id"])
    assert triggered["next_run_at"] == _FROZEN.isoformat()


def test_remove_job(_isolate):
    job = cron_module.create_job("delete-me", "30m")
    assert cron_module.remove_job(job["id"]) is True
    assert cron_module.get_job(job["id"]) is None
    assert cron_module.remove_job(job["id"]) is False


def test_update_job_changes_schedule(_isolate):
    job = cron_module.create_job("hello", "every 1h")
    updated = cron_module.update_job(job["id"], {"schedule": "every 30m"})
    assert updated["schedule"]["minutes"] == 30
    assert updated["schedule_display"] == "every 30m"


def test_update_unknown_returns_none(_isolate):
    assert cron_module.update_job("nope", {"name": "x"}) is None


# ── Persistence atomicity ───────────────────────────────────────────


def test_save_and_load_round_trip(_isolate):
    cron_module.create_job("a", "30m")
    cron_module.create_job("b", "every 2h")
    raw = cron_module.load_jobs()
    assert len(raw) == 2
    cron_module.save_jobs(raw)
    again = cron_module.load_jobs()
    assert again == raw


# ── Due jobs / mark_job_run / advance ───────────────────────────────


def test_get_due_jobs_returns_overdue_only(_isolate):
    state = _isolate
    overdue = cron_module.create_job("overdue", "30m")
    cron_module.create_job("future", "2h")
    _advance(state, hours=1)
    due = cron_module.get_due_jobs()
    due_ids = {j["id"] for j in due}
    assert overdue["id"] in due_ids
    assert len(due) == 1


def test_mark_job_run_advances_interval(_isolate):
    state = _isolate
    job = cron_module.create_job("ping", "every 30m")
    _advance(state, minutes=31)
    assert any(j["id"] == job["id"] for j in cron_module.get_due_jobs())
    cron_module.mark_job_run(job["id"], success=True)
    after = cron_module.get_job(job["id"])
    assert after["last_status"] == "ok"
    assert after["next_run_at"] == (state["now"] + timedelta(minutes=30)).isoformat()


def test_mark_job_run_deletes_oneshot_after_completion(_isolate):
    state = _isolate
    job = cron_module.create_job("oneshot", "30m")
    _advance(state, hours=1)
    cron_module.mark_job_run(job["id"], success=True)
    assert cron_module.get_job(job["id"]) is None


def test_mark_job_run_records_error(_isolate):
    state = _isolate
    job = cron_module.create_job("failing", "every 30m")
    _advance(state, minutes=31)
    cron_module.mark_job_run(job["id"], success=False, error="boom")
    after = cron_module.get_job(job["id"])
    assert after["last_status"] == "error"
    assert after["last_error"] == "boom"


def test_advance_next_run_only_for_recurring(_isolate):
    state = _isolate
    interval = cron_module.create_job("interval", "every 30m")
    oneshot = cron_module.create_job("oneshot", "30m")
    _advance(state, minutes=31)
    assert cron_module.advance_next_run(interval["id"]) is True
    assert cron_module.advance_next_run(oneshot["id"]) is False


def test_get_due_jobs_fast_forwards_stale_recurring(_isolate):
    state = _isolate
    job = cron_module.create_job("stale", "every 30m")
    _advance(state, hours=4)
    due = cron_module.get_due_jobs()
    assert all(j["id"] != job["id"] for j in due)
    after = cron_module.get_job(job["id"])
    assert after["next_run_at"] is not None
    assert datetime.fromisoformat(after["next_run_at"]) > state["now"]


# ── save_job_output ─────────────────────────────────────────────────


def test_save_job_output_writes_atomic_file(_isolate, tmp_path):
    job = cron_module.create_job("ping", "30m")
    target = cron_module.save_job_output(job["id"], "hello world")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello world"
    assert target.suffix == ".md"


# ── Scheduler ───────────────────────────────────────────────────────


def test_scheduler_tick_runs_due_jobs(_isolate):
    state = _isolate
    a = cron_module.create_job("a", "30m")
    b = cron_module.create_job("b", "every 30m")
    _advance(state, minutes=31)

    runs: list[str] = []

    async def runner(job):
        runs.append(job["id"])
        return f"output for {job['id']}"

    scheduler = cron_module.Scheduler(runner, interval_seconds=1)
    fired = asyncio.run(scheduler.tick())
    assert fired == 2
    assert {a["id"], b["id"]} == set(runs)

    assert cron_module.get_job(a["id"]) is None  # one-shot deleted
    after = cron_module.get_job(b["id"])
    assert after["last_status"] == "ok"


def test_scheduler_records_runner_failure(_isolate):
    state = _isolate
    job = cron_module.create_job("fail", "every 30m")
    _advance(state, minutes=31)

    async def runner(job):
        raise RuntimeError("boom")

    scheduler = cron_module.Scheduler(runner, interval_seconds=1)
    fired = asyncio.run(scheduler.tick())
    assert fired == 1
    after = cron_module.get_job(job["id"])
    assert after["last_status"] == "error"
    assert "boom" in (after["last_error"] or "")
    assert scheduler.stats["error_count"] == 1


def test_scheduler_rejects_invalid_interval():
    with pytest.raises(cron_module.SchedulerError):
        cron_module.Scheduler(lambda j: None, interval_seconds=0)  # type: ignore[arg-type]


# ── Cross-port surface check ────────────────────────────────────────


def test_public_surface_is_stable():
    expected = {
        "CRONITER_AVAILABLE",
        "Job",
        "JobRunner",
        "ParsedSchedule",
        "Scheduler",
        "SchedulerError",
        "advance_next_run",
        "compute_next_run",
        "create_job",
        "get_due_jobs",
        "get_job",
        "list_jobs",
        "load_jobs",
        "mark_job_run",
        "parse_duration",
        "parse_schedule",
        "pause_job",
        "remove_job",
        "resume_job",
        "save_job_output",
        "save_jobs",
        "trigger_job",
        "update_job",
    }
    assert expected.issubset(set(cron_module.__all__))
    for name in expected:
        assert hasattr(cron_module, name), f"missing public symbol: {name}"
