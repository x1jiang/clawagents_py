"""Agent tools for background jobs."""

from __future__ import annotations

import json
from typing import Any

from clawagents.background import BackgroundJob, BackgroundJobManager
from clawagents.tools.registry import ToolResult


_DEFAULT_MANAGER = BackgroundJobManager()


# Cap so a burst of finished jobs cannot flood a tool result.
_MAX_ANNOUNCED = 10
_MAX_TAIL_CHARS = 400


def background_completion_notice(
    owned_job_ids: set[str],
    announced: set[str],
    *,
    manager: BackgroundJobManager | None = None,
) -> str:
    """A one-shot notice for owned jobs that finished since the last check.

    Without this the agent only discovers a job ended by remembering to call
    ``task_status``; a job that finishes while it is busy elsewhere goes
    unnoticed for the rest of the run. The tool layer appends this to the next
    tool result, so completion reaches the model on its own.

    ``owned_job_ids`` scopes the notice to jobs this caller started, and
    ``announced`` (mutated) keeps each completion to a single mention. Both are
    caller-held rather than manager-held because one process can run several
    agents/subagents against a shared manager, and one agent must never be
    told about — or silently consume — another's completions.

    Returns ``""`` when nothing new finished; callers append unconditionally.
    """
    if not owned_job_ids:
        return ""
    mgr = manager or _DEFAULT_MANAGER
    try:
        done = [
            job
            for job in mgr.completed_jobs(owned_job_ids)
            if job.id not in announced
        ]
    except Exception:  # noqa: BLE001 - notices must never break a tool call
        return ""
    if not done:
        return ""
    for job in done:
        announced.add(job.id)

    shown, hidden = done[:_MAX_ANNOUNCED], max(0, len(done) - _MAX_ANNOUNCED)
    lines = []
    for job in shown:
        verdict = (
            "cancelled"
            if job.cancelled
            else ("ok" if job.exit_code == 0 else f"exit {job.exit_code}")
        )
        line = f"- {job.id} ({verdict}): {' '.join(job.command)[:120]}"
        # A failure is the case worth acting on, so carry a little evidence.
        if not job.cancelled and job.exit_code not in (0, None):
            tail = (job.stderr or job.stdout or "").strip()
            if tail:
                line += f"\n    {tail[-_MAX_TAIL_CHARS:]}"
        lines.append(line)
    if hidden:
        lines.append(f"- …and {hidden} more (use task_status for details)")
    return (
        "\n\n<system-reminder>\nBackground job(s) finished since your last "
        "tool call:\n" + "\n".join(lines) + "\n</system-reminder>"
    )


def _job_json(job: BackgroundJob) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "command": job.command,
        "cwd": job.cwd,
        "pid": job.pid,
        "running": job.running,
        "exit_code": job.exit_code,
        "cancelled": job.cancelled,
    }


class _TaskCreateTool:
    name = "task_create"
    description = "Start a background command and return its job id."
    keywords = ["background", "job", "task", "process", "long-running"]
    parameters = {
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Command argv list.",
            "required": True,
        },
        "cwd": {"type": "string", "description": "Working directory."},
    }

    def __init__(self, manager: BackgroundJobManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        command = args.get("command")
        if not isinstance(command, list) or not command:
            return ToolResult(False, "", "command must be a non-empty argv list")
        job = await self._manager.start([str(part) for part in command], cwd=args.get("cwd") or None)
        return ToolResult(True, json.dumps(_job_json(job)))


class _TaskStatusTool:
    name = "task_status"
    description = "Return status for a background job."
    keywords = ["background", "job", "task", "status"]
    parameters = {"job_id": {"type": "string", "description": "Job id.", "required": True}}

    def __init__(self, manager: BackgroundJobManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            return ToolResult(True, json.dumps(_job_json(self._manager.status(str(args.get("job_id") or "")))))
        except Exception as exc:
            return ToolResult(False, "", str(exc))


class _TaskOutputTool:
    name = "task_output"
    description = "Return captured stdout and stderr for a background job."
    keywords = ["background", "job", "task", "output", "logs"]
    parameters = {"job_id": {"type": "string", "description": "Job id.", "required": True}}

    def __init__(self, manager: BackgroundJobManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            job = self._manager.status(str(args.get("job_id") or ""))
        except Exception as exc:
            return ToolResult(False, "", str(exc))
        return ToolResult(True, f"stdout:\n{job.stdout}\n\nstderr:\n{job.stderr}")


class _TaskStopTool:
    name = "task_stop"
    description = "Cancel a running background job."
    keywords = ["background", "job", "task", "stop", "cancel"]
    parameters = {"job_id": {"type": "string", "description": "Job id.", "required": True}}

    def __init__(self, manager: BackgroundJobManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            job = await self._manager.cancel(str(args.get("job_id") or ""))
        except Exception as exc:
            return ToolResult(False, "", str(exc))
        return ToolResult(True, json.dumps(_job_json(job)))


class _TaskListTool:
    name = "task_list"
    description = "List known background jobs."
    keywords = ["background", "job", "task", "list"]
    parameters: dict[str, dict[str, Any]] = {}

    def __init__(self, manager: BackgroundJobManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        del args
        return ToolResult(True, json.dumps([_job_json(job) for job in self._manager.list()]))


def create_background_task_tools(manager: BackgroundJobManager | None = None):
    mgr = manager or _DEFAULT_MANAGER
    return [
        _TaskCreateTool(mgr),
        _TaskStatusTool(mgr),
        _TaskOutputTool(mgr),
        _TaskStopTool(mgr),
        _TaskListTool(mgr),
    ]

