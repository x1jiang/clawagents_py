"""Agent tools for background jobs."""

from __future__ import annotations

import json
from typing import Any

from clawagents.background import BackgroundJob, BackgroundJobManager
from clawagents.tools.registry import ToolResult


_DEFAULT_MANAGER = BackgroundJobManager()


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
        "command": {"type": "array", "description": "Command argv list.", "required": True},
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

