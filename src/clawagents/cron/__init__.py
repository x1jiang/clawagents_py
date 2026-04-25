"""``clawagents.cron`` — scheduled jobs for agent runs.

A small persistent job store on top of ``~/.clawagents/<profile>/cron``.
Three schedule kinds are supported:

- **once** — one-shot at an ISO timestamp or a relative duration like
  ``"30m"`` / ``"2h"`` / ``"1d"``.
- **interval** — recurring, e.g. ``"every 30m"`` or ``"every 2h"``.
- **cron** — full cron expressions like ``"0 9 * * *"`` (requires the
  optional ``croniter`` extra: ``pip install clawagents[cron]``).

Public API::

    from clawagents.cron import (
        Job,
        ParsedSchedule,
        parse_schedule,
        compute_next_run,
        create_job,
        list_jobs,
        get_job,
        update_job,
        pause_job,
        resume_job,
        trigger_job,
        remove_job,
        get_due_jobs,
        mark_job_run,
        advance_next_run,
        save_job_output,
        Scheduler,
        SchedulerError,
        CRONITER_AVAILABLE,
    )

The store is profile-aware (``~/.clawagents/<profile>/cron/jobs.json``)
so each profile sees its own schedule.

Lazy import: ``croniter`` is loaded only when a cron-expression schedule
is actually parsed. The other two kinds work without it.
"""

from clawagents.cron.errors import SchedulerError
from clawagents.cron.jobs import (
    CRONITER_AVAILABLE,
    Job,
    ParsedSchedule,
    advance_next_run,
    compute_next_run,
    create_job,
    get_due_jobs,
    get_job,
    list_jobs,
    load_jobs,
    mark_job_run,
    parse_duration,
    parse_schedule,
    pause_job,
    remove_job,
    resume_job,
    save_job_output,
    save_jobs,
    trigger_job,
    update_job,
)
from clawagents.cron.scheduler import JobRunner, Scheduler

__all__ = [
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
]
