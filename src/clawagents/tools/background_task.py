"""Agent tools for background jobs."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from clawagents.background import BackgroundJob, BackgroundJobManager
from clawagents.tools.registry import ToolResult


_DEFAULT_MANAGER = BackgroundJobManager()


# Cap so a burst of finished jobs cannot flood a tool result.
_MAX_ANNOUNCED = 10
_MAX_TAIL_CHARS = 400

# Output budgets for task_output / task_wait. Background jobs are exactly the
# long, chatty ones, so the tail is returned rather than the head.
_DEFAULT_OUTPUT_CHARS = 4000
_MAX_OUTPUT_CHARS = 20000

# A single wait must not consume the whole tool timeout (registry default
# 120s), or the wait itself gets killed and the caller learns nothing.
_DEFAULT_WAIT_MS = 60_000
_MAX_WAIT_MS = 90_000


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
    # ``is None``, not ``or``: the manager defines __len__, so one holding no
    # jobs is falsy and would be swapped for the process default — exactly the
    # case where a caller's own manager needs to be honoured.
    mgr = _DEFAULT_MANAGER if manager is None else manager
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
        line = f"- {job.id} ({verdict}): {_display_command(job)[:120]}"
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


def _display_command(job: BackgroundJob) -> str:
    """Readable command text, tolerating job objects without a label.

    ``display_command`` arrived after this module; a manager from an older
    install still hands back plain jobs, and a status call is not worth an
    AttributeError.
    """
    text = getattr(job, "display_command", None)
    return text if isinstance(text, str) and text else " ".join(job.command)


def _job_json(job: BackgroundJob) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job.id,
        # The command as asked for, not the sandbox/session wrapper it became.
        "command": _display_command(job),
        "cwd": job.cwd,
        "pid": job.pid,
        "running": job.running,
        "exit_code": job.exit_code,
        "cancelled": job.cancelled,
    }
    # Elapsed time is what a caller actually reasons about when deciding
    # whether to keep waiting; raw epoch timestamps make it do the math.
    if job.started_at:
        end = job.ended_at or time.time()
        payload["elapsed_ms"] = int((end - job.started_at) * 1000)
    return payload


def _tail(text: str, limit: int) -> str:
    """Last ``limit`` chars, flagged when content was dropped."""
    if len(text) <= limit:
        return text
    return f"[…{len(text) - limit} earlier chars omitted]\n" + text[-limit:]


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
    description = (
        "Return captured stdout and stderr for a background job. "
        "Safe to call while the job is still running."
    )
    keywords = ["background", "job", "task", "output", "logs"]
    parameters = {
        "job_id": {"type": "string", "description": "Job id.", "required": True},
        "max_chars": {
            "type": "number",
            "description": (
                "Cap on each stream, keeping the tail. "
                f"Default: {_DEFAULT_OUTPUT_CHARS}."
            ),
        },
    }

    def __init__(self, manager: BackgroundJobManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            job = self._manager.status(str(args.get("job_id") or ""))
        except Exception as exc:
            return ToolResult(False, "", str(exc))
        try:
            limit = int(args.get("max_chars") or _DEFAULT_OUTPUT_CHARS)
        except (TypeError, ValueError):
            limit = _DEFAULT_OUTPUT_CHARS
        limit = max(200, min(limit, _MAX_OUTPUT_CHARS))
        # A build or test suite can emit megabytes; the tail is where the
        # verdict lives, and an uncapped dump would evict the conversation.
        header = "still running" if job.running else f"exit {job.exit_code}"
        return ToolResult(
            True,
            f"[{job.id}: {header}]\n"
            f"stdout:\n{_tail(job.stdout, limit)}\n\n"
            f"stderr:\n{_tail(job.stderr, limit)}",
        )


class _TaskWaitTool:
    name = "task_wait"
    description = (
        "Block until a background job exits, then return its exit code and "
        "output tail. Use this instead of ending your turn when the job's "
        "result is what the user asked for."
    )
    keywords = ["background", "job", "task", "wait", "await", "block", "join"]
    parameters = {
        "job_id": {"type": "string", "description": "Job id.", "required": True},
        "timeout_ms": {
            "type": "number",
            "description": (
                "Give up waiting after this long and report progress so far. "
                f"Default: {_DEFAULT_WAIT_MS}. Max: {_MAX_WAIT_MS}."
            ),
        },
        "max_chars": {
            "type": "number",
            "description": (
                f"Cap on each output stream, keeping the tail. Default: {_DEFAULT_OUTPUT_CHARS}."
            ),
        },
    }

    def __init__(self, manager: BackgroundJobManager) -> None:
        self._manager = manager

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        job_id = str(args.get("job_id") or "")
        try:
            job = self._manager.status(job_id)
        except Exception as exc:
            return ToolResult(False, "", str(exc))
        try:
            budget = int(args.get("timeout_ms") or _DEFAULT_WAIT_MS)
        except (TypeError, ValueError):
            budget = _DEFAULT_WAIT_MS
        budget = max(1000, min(budget, _MAX_WAIT_MS))
        try:
            limit = int(args.get("max_chars") or _DEFAULT_OUTPUT_CHARS)
        except (TypeError, ValueError):
            limit = _DEFAULT_OUTPUT_CHARS
        limit = max(200, min(limit, _MAX_OUTPUT_CHARS))

        timed_out = False
        try:
            job = await self._manager.await_complete(job_id, timeout=budget / 1000.0)
        except asyncio.TimeoutError:
            # Not an error: the job is simply still going. Report partial
            # output so the caller can decide to wait again or move on.
            timed_out = True
        except Exception as exc:
            return ToolResult(False, "", str(exc))

        payload = _job_json(job)
        payload["waited_ms"] = budget if timed_out else payload.get("elapsed_ms")
        payload["timed_out"] = timed_out
        if timed_out:
            payload["hint"] = (
                "Job is still running. Call task_wait again to keep waiting, "
                "or task_stop to give up on it."
            )
        return ToolResult(
            True,
            json.dumps(payload, indent=2)
            + f"\n\nstdout:\n{_tail(job.stdout, limit)}"
            + f"\n\nstderr:\n{_tail(job.stderr, limit)}",
        )


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
    # See background_completion_notice: an empty manager is falsy via __len__,
    # so a passed-in one must be detected with ``is None``.
    mgr = _DEFAULT_MANAGER if manager is None else manager
    return [
        _TaskCreateTool(mgr),
        _TaskStatusTool(mgr),
        _TaskOutputTool(mgr),
        _TaskWaitTool(mgr),
        _TaskStopTool(mgr),
        _TaskListTool(mgr),
    ]

