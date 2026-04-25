"""Cron job storage and schedule arithmetic.

Jobs are stored in ``<clawagents_home>/cron/jobs.json`` and outputs in
``<clawagents_home>/cron/output/<job_id>/<timestamp>.md``. Storage uses
atomic ``os.replace`` writes plus a per-process ``threading.Lock`` so
concurrent ``mark_job_run`` / ``advance_next_run`` calls do not race.

The schedule kinds supported here mirror Hermes' contract — ``once``,
``interval``, ``cron`` — but the public API is intentionally smaller:
no skill loaders, no delivery channels. ClawAgents' ``Scheduler`` lets
the caller plug in any runner via :class:`JobRunner`.

``croniter`` is loaded lazily so the rest of cron works without the
optional extra (``pip install clawagents[cron]``).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from clawagents.paths import get_clawagents_home

logger = logging.getLogger("clawagents.cron")

_croniter: Any = None
try:
    from croniter import croniter as _croniter_imported  # type: ignore[import-untyped]

    _croniter = _croniter_imported
    CRONITER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by tests via monkeypatch
    CRONITER_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Directories are resolved lazily — get_clawagents_home() respects
# CLAWAGENTS_HOME and CLAWAGENTS_PROFILE which tests flip during runs.

ONESHOT_GRACE_SECONDS = 120

_jobs_file_lock = threading.Lock()
"""Protects the ``load → modify → save`` cycle in this process."""

# Pluggable clock so tests can pin time without mocking the global ``datetime``.
NowFn = Callable[[], datetime]
_now_fn: NowFn = lambda: datetime.now(timezone.utc).astimezone()


def set_clock(fn: NowFn) -> None:
    """Override the cron module's notion of *now* (for tests)."""
    global _now_fn
    _now_fn = fn


def reset_clock() -> None:
    """Restore the default wall-clock clock."""
    global _now_fn
    _now_fn = lambda: datetime.now(timezone.utc).astimezone()


def _now() -> datetime:
    return _now_fn()


def _cron_dir(create: bool = True) -> Path:
    home = get_clawagents_home(create=create)
    cron = home / "cron"
    if create:
        cron.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(cron, 0o700)
        except (OSError, NotImplementedError):
            pass
    return cron


def _jobs_file() -> Path:
    return _cron_dir() / "jobs.json"


def _output_dir() -> Path:
    out = _cron_dir() / "output"
    out.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out, 0o700)
    except (OSError, NotImplementedError):
        pass
    return out


# ──────────────────────────────────────────────────────────────────────
# Schedule parsing
# ──────────────────────────────────────────────────────────────────────


class ParsedSchedule(TypedDict, total=False):
    """Normalized schedule dict produced by :func:`parse_schedule`."""

    kind: str  # "once" | "interval" | "cron"
    run_at: str  # for "once" — ISO timestamp
    minutes: int  # for "interval"
    expr: str  # for "cron"
    display: str


_DURATION_RE = re.compile(
    r"^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$"
)


def parse_duration(s: str) -> int:
    """Parse a duration like ``"30m"`` / ``"2h"`` / ``"1d"`` to minutes."""
    s = s.strip().lower()
    match = _DURATION_RE.match(s)
    if not match:
        raise ValueError(
            f"Invalid duration: {s!r}. Use format like '30m', '2h', or '1d'."
        )
    value = int(match.group(1))
    unit = match.group(2)[0]
    return value * {"m": 1, "h": 60, "d": 1440}[unit]


def parse_schedule(schedule: str) -> ParsedSchedule:
    """Parse a schedule string into a structured :class:`ParsedSchedule`.

    Accepted forms::

        "30m"               # one-shot in 30 minutes
        "2h"                # one-shot in 2 hours
        "every 30m"         # recurring every 30 minutes
        "every 2h"          # recurring every 2 hours
        "0 9 * * *"         # cron expression (requires `croniter` extra)
        "2026-02-03T14:00"  # one-shot at ISO timestamp
    """
    schedule = schedule.strip()
    original = schedule
    lower = schedule.lower()

    if lower.startswith("every "):
        minutes = parse_duration(schedule[6:].strip())
        return {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m",
        }

    parts = schedule.split()
    if len(parts) >= 5 and all(re.match(r"^[\d\*\-,/]+$", p) for p in parts[:5]):
        if not CRONITER_AVAILABLE or _croniter is None:
            raise ValueError(
                "Cron expressions require the optional `croniter` extra. "
                "Install with `pip install clawagents[cron]`."
            )
        try:
            _croniter(schedule)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid cron expression {schedule!r}: {exc}") from exc
        return {"kind": "cron", "expr": schedule, "display": schedule}

    if "T" in schedule or re.match(r"^\d{4}-\d{2}-\d{2}", schedule):
        try:
            dt = datetime.fromisoformat(schedule.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.astimezone()
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}",
            }
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp {schedule!r}: {exc}") from exc

    try:
        minutes = parse_duration(schedule)
        run_at = _now() + timedelta(minutes=minutes)
        return {
            "kind": "once",
            "run_at": run_at.isoformat(),
            "display": f"once in {original}",
        }
    except ValueError:
        pass

    raise ValueError(
        f"Invalid schedule {original!r}. Use:\n"
        "  - Duration: '30m', '2h', '1d' (one-shot)\n"
        "  - Interval: 'every 30m', 'every 2h' (recurring)\n"
        "  - Cron: '0 9 * * *' (requires `croniter` extra)\n"
        "  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    )


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz)
    return dt


def _recoverable_oneshot_run_at(
    schedule: ParsedSchedule,
    now: datetime,
    *,
    last_run_at: str | None = None,
) -> str | None:
    if schedule.get("kind") != "once":
        return None
    if last_run_at:
        return None
    run_at = schedule.get("run_at")
    if not run_at:
        return None
    run_at_dt = _ensure_aware(datetime.fromisoformat(run_at))
    if run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
        return run_at
    return None


def _grace_seconds(schedule: ParsedSchedule) -> int:
    """How late a recurring job can be before we fast-forward."""
    MIN_GRACE = 120
    MAX_GRACE = 7200  # 2h
    kind = schedule.get("kind")

    if kind == "interval":
        period = schedule.get("minutes", 1) * 60
        return max(MIN_GRACE, min(period // 2, MAX_GRACE))

    if kind == "cron" and CRONITER_AVAILABLE and _croniter is not None:
        try:
            cron = _croniter(schedule["expr"], _now())
            first = cron.get_next(datetime)
            second = cron.get_next(datetime)
            period = int((second - first).total_seconds())
            return max(MIN_GRACE, min(period // 2, MAX_GRACE))
        except Exception:
            return MIN_GRACE

    return MIN_GRACE


def compute_next_run(
    schedule: ParsedSchedule, last_run_at: str | None = None
) -> str | None:
    """Compute the next scheduled run time as an ISO timestamp."""
    now = _now()
    kind = schedule.get("kind")

    if kind == "once":
        return _recoverable_oneshot_run_at(schedule, now, last_run_at=last_run_at)

    if kind == "interval":
        minutes = schedule["minutes"]
        if last_run_at:
            last = _ensure_aware(datetime.fromisoformat(last_run_at))
            return (last + timedelta(minutes=minutes)).isoformat()
        return (now + timedelta(minutes=minutes)).isoformat()

    if kind == "cron":
        if not CRONITER_AVAILABLE or _croniter is None:
            return None
        cron = _croniter(schedule["expr"], now)
        return cron.get_next(datetime).isoformat()

    return None


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    """In-memory representation of a cron job. Mirrors the on-disk schema."""

    id: str
    name: str
    prompt: str
    schedule: ParsedSchedule
    schedule_display: str
    repeat: dict
    enabled: bool
    state: str
    created_at: str
    next_run_at: str | None
    last_run_at: str | None = None
    last_status: str | None = None
    last_error: str | None = None
    paused_at: str | None = None
    paused_reason: str | None = None
    workdir: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "schedule": dict(self.schedule),
            "schedule_display": self.schedule_display,
            "repeat": dict(self.repeat),
            "enabled": self.enabled,
            "state": self.state,
            "created_at": self.created_at,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "paused_at": self.paused_at,
            "paused_reason": self.paused_reason,
            "workdir": self.workdir,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        return cls(
            id=data["id"],
            name=data["name"],
            prompt=data.get("prompt", ""),
            schedule=data.get("schedule", {}),
            schedule_display=data.get("schedule_display", ""),
            repeat=data.get("repeat", {"times": None, "completed": 0}),
            enabled=bool(data.get("enabled", True)),
            state=data.get("state", "scheduled"),
            created_at=data.get("created_at", ""),
            next_run_at=data.get("next_run_at"),
            last_run_at=data.get("last_run_at"),
            last_status=data.get("last_status"),
            last_error=data.get("last_error"),
            paused_at=data.get("paused_at"),
            paused_reason=data.get("paused_reason"),
            workdir=data.get("workdir"),
            metadata=data.get("metadata", {}),
        )


def load_jobs() -> list[dict]:
    """Load all jobs from ``jobs.json`` (raw dict form)."""
    path = _jobs_file()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("jobs", [])
    except json.JSONDecodeError as exc:
        logger.error("jobs.json corrupted: %s", exc)
        raise


def save_jobs(jobs: list[dict]) -> None:
    """Atomically replace ``jobs.json`` with the given list."""
    cron = _cron_dir()
    target = _jobs_file()
    fd, tmp = tempfile.mkstemp(dir=str(cron), prefix=".jobs_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"jobs": jobs, "updated_at": _now().isoformat()}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        try:
            os.chmod(target, 0o600)
        except (OSError, NotImplementedError):
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ──────────────────────────────────────────────────────────────────────
# CRUD
# ──────────────────────────────────────────────────────────────────────


def _normalize_workdir(workdir: str | None) -> str | None:
    if workdir is None:
        return None
    raw = str(workdir).strip()
    if not raw:
        return None
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Cron workdir must be an absolute path (got {raw!r}). "
            "Cron jobs run detached from any shell cwd, so relative paths are ambiguous."
        )
    resolved = expanded.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Cron workdir does not exist or is not a directory: {resolved}")
    return str(resolved)


def create_job(
    prompt: str,
    schedule: str,
    *,
    name: str | None = None,
    repeat: int | None = None,
    workdir: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a new cron job and persist it."""
    parsed = parse_schedule(schedule)

    if repeat is not None and repeat <= 0:
        repeat = None
    if parsed["kind"] == "once" and repeat is None:
        repeat = 1

    job_id = uuid.uuid4().hex[:12]
    now_iso = _now().isoformat()
    label_source = (prompt or job_id).strip()

    job = {
        "id": job_id,
        "name": (name or label_source[:50]).strip(),
        "prompt": prompt,
        "schedule": dict(parsed),
        "schedule_display": parsed.get("display", schedule),
        "repeat": {"times": repeat, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": now_iso,
        "next_run_at": compute_next_run(parsed),
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "workdir": _normalize_workdir(workdir),
        "metadata": dict(metadata or {}),
    }

    with _jobs_file_lock:
        jobs = load_jobs()
        jobs.append(job)
        save_jobs(jobs)
    return job


def get_job(job_id: str) -> dict | None:
    for job in load_jobs():
        if job["id"] == job_id:
            return job
    return None


def list_jobs(include_disabled: bool = False) -> list[dict]:
    jobs = load_jobs()
    if include_disabled:
        return jobs
    return [j for j in jobs if j.get("enabled", True)]


def update_job(job_id: str, updates: dict) -> dict | None:
    """Apply *updates* to *job_id*; returns the new dict or None if missing."""
    with _jobs_file_lock:
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue

            if "workdir" in updates:
                wd = updates["workdir"]
                updates["workdir"] = (
                    None if wd in (None, "", False) else _normalize_workdir(wd)
                )

            updated = {**job, **updates}

            if "schedule" in updates:
                sched = updates["schedule"]
                if isinstance(sched, str):
                    sched = parse_schedule(sched)
                    updated["schedule"] = sched
                updated["schedule_display"] = updates.get(
                    "schedule_display",
                    sched.get("display", updated.get("schedule_display")),
                )
                if updated.get("state") != "paused":
                    updated["next_run_at"] = compute_next_run(sched)

            if (
                updated.get("enabled", True)
                and updated.get("state") != "paused"
                and not updated.get("next_run_at")
            ):
                updated["next_run_at"] = compute_next_run(updated["schedule"])

            jobs[i] = updated
            save_jobs(jobs)
            return updated
    return None


def pause_job(job_id: str, reason: str | None = None) -> dict | None:
    return update_job(
        job_id,
        {
            "enabled": False,
            "state": "paused",
            "paused_at": _now().isoformat(),
            "paused_reason": reason,
        },
    )


def resume_job(job_id: str) -> dict | None:
    job = get_job(job_id)
    if not job:
        return None
    return update_job(
        job_id,
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": compute_next_run(job["schedule"]),
        },
    )


def trigger_job(job_id: str) -> dict | None:
    """Schedule a job to fire on the next tick."""
    job = get_job(job_id)
    if not job:
        return None
    return update_job(
        job_id,
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": _now().isoformat(),
        },
    )


def remove_job(job_id: str) -> bool:
    with _jobs_file_lock:
        jobs = load_jobs()
        before = len(jobs)
        jobs = [j for j in jobs if j["id"] != job_id]
        if len(jobs) < before:
            save_jobs(jobs)
            return True
    return False


def mark_job_run(
    job_id: str,
    success: bool,
    *,
    error: str | None = None,
) -> None:
    """Record a run and advance the schedule. Auto-deletes finished one-shots."""
    with _jobs_file_lock:
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue
            now = _now().isoformat()
            job["last_run_at"] = now
            job["last_status"] = "ok" if success else "error"
            job["last_error"] = None if success else error

            repeat = job.get("repeat") or {}
            repeat["completed"] = repeat.get("completed", 0) + 1
            times = repeat.get("times")
            if times is not None and times > 0 and repeat["completed"] >= times:
                jobs.pop(i)
                save_jobs(jobs)
                return
            job["repeat"] = repeat

            job["next_run_at"] = compute_next_run(job["schedule"], now)
            if job["next_run_at"] is None:
                job["enabled"] = False
                job["state"] = "completed"
            elif job.get("state") != "paused":
                job["state"] = "scheduled"
            save_jobs(jobs)
            return
        logger.warning("mark_job_run: job_id %s not found", job_id)


def advance_next_run(job_id: str) -> bool:
    """Pre-advance ``next_run_at`` so a crash mid-run won't refire the job.

    Returns ``True`` if the timestamp was advanced. One-shot jobs are
    untouched so they remain eligible for retry on restart.
    """
    with _jobs_file_lock:
        jobs = load_jobs()
        for job in jobs:
            if job["id"] != job_id:
                continue
            kind = job.get("schedule", {}).get("kind")
            if kind not in ("cron", "interval"):
                return False
            new_next = compute_next_run(job["schedule"], _now().isoformat())
            if new_next and new_next != job.get("next_run_at"):
                job["next_run_at"] = new_next
                save_jobs(jobs)
                return True
            return False
    return False


def get_due_jobs() -> list[dict]:
    """Return all jobs whose ``next_run_at`` is at or before ``now``.

    Recurring jobs that are past their grace window are fast-forwarded
    instead of fired so a long downtime doesn't produce a burst.
    """
    now = _now()
    raw = load_jobs()
    jobs = copy.deepcopy(raw)
    due: list[dict] = []
    needs_save = False

    for job in jobs:
        if not job.get("enabled", True):
            continue

        next_run = job.get("next_run_at")
        if not next_run:
            recovered = _recoverable_oneshot_run_at(
                job.get("schedule", {}), now, last_run_at=job.get("last_run_at")
            )
            if not recovered:
                continue
            job["next_run_at"] = recovered
            next_run = recovered
            for rj in raw:
                if rj["id"] == job["id"]:
                    rj["next_run_at"] = recovered
                    needs_save = True
                    break

        next_run_dt = _ensure_aware(datetime.fromisoformat(next_run))
        if next_run_dt > now:
            continue

        sched = job.get("schedule", {})
        kind = sched.get("kind")
        grace = _grace_seconds(sched)
        if kind in ("cron", "interval") and (now - next_run_dt).total_seconds() > grace:
            new_next = compute_next_run(sched, now.isoformat())
            if new_next:
                logger.info(
                    "Job %r missed window (was %s, grace=%ds); "
                    "fast-forwarding to %s",
                    job.get("name", job["id"]),
                    next_run,
                    grace,
                    new_next,
                )
                for rj in raw:
                    if rj["id"] == job["id"]:
                        rj["next_run_at"] = new_next
                        needs_save = True
                        break
                continue

        due.append(job)

    if needs_save:
        with _jobs_file_lock:
            save_jobs(raw)

    return due


def save_job_output(job_id: str, output: str) -> Path:
    """Persist a run's output under ``cron/output/<job_id>/<ts>.md``."""
    base = _output_dir() / job_id
    base.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(base, 0o700)
    except (OSError, NotImplementedError):
        pass
    timestamp = _now().strftime("%Y-%m-%d_%H-%M-%S")
    target = base / f"{timestamp}.md"
    fd, tmp = tempfile.mkstemp(dir=str(base), prefix=".output_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        try:
            os.chmod(target, 0o600)
        except (OSError, NotImplementedError):
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target
