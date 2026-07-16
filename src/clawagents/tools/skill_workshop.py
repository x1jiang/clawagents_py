"""skill_workshop tool — governed skill proposals."""

from __future__ import annotations

import json
import os
from typing import Any

from clawagents.permissions.mode import PermissionMode
from clawagents.skills.workshop.service import SkillWorkshopService
from clawagents.tools.registry import Tool, ToolResult


_WRITE_ACTIONS = frozenset(
    {"create", "update", "revise", "apply", "reject", "quarantine", "rollback"}
)


class SkillWorkshopTool:
    name = "skill_workshop"
    description = (
        "Governed skill authoring: create/update proposals, scan, apply, reject, quarantine, rollback. "
        "Never write live SKILL.md directly — use create/update then apply after review."
    )
    parameters = {
        "action": {
            "type": "string",
            "description": "Workshop action",
            "required": True,
        },
        "proposal_id": {"type": "string", "description": "Proposal id", "required": False},
        "rollback_id": {"type": "string", "description": "Rollback snapshot id", "required": False},
        "name": {"type": "string", "description": "Skill name", "required": False},
        "target_skill": {"type": "string", "description": "Target skill for update", "required": False},
        "description": {"type": "string", "description": "Skill description", "required": False},
        "body": {"type": "string", "description": "Proposal SKILL.md body", "required": False},
        "goal": {"type": "string", "description": "Authoring goal", "required": False},
        "evidence": {"type": "string", "description": "Supporting evidence", "required": False},
        "reason": {"type": "string", "description": "Reject/quarantine reason", "required": False},
        "support_files": {
            "type": "string",
            "description": "JSON array of {path, content}",
            "required": False,
        },
    }

    def __init__(
        self,
        workspace: str | None = None,
        skills_dir: str | None = None,
        *,
        on_reload: Any = None,
    ) -> None:
        self._service = SkillWorkshopService(workspace or os.getcwd(), skills_dir)
        self._on_reload = on_reload

    async def execute(self, args: dict[str, Any], run_context: Any = None) -> ToolResult:
        action = str(args.get("action", ""))
        if (
            run_context is not None
            and getattr(run_context, "permission_mode", PermissionMode.DEFAULT) == PermissionMode.PLAN
            and action in _WRITE_ACTIONS
        ):
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Refused: skill_workshop action '{action}' mutates state and "
                    "is unavailable in plan mode. Call exit_plan_mode first."
                ),
            )
        support_files = _parse_support_files(args.get("support_files"))
        try:
            if action == "create":
                result = self._service.create(
                    name=str(args.get("name", "")),
                    description=str(args.get("description", "")),
                    body=str(args.get("body", "")),
                    goal=str(args.get("goal", "")),
                    evidence=str(args.get("evidence", "")),
                    support_files=support_files,
                )
            elif action == "update":
                result = self._service.update(
                    target_skill=str(args.get("target_skill", args.get("name", ""))),
                    description=str(args.get("description", "")),
                    body=str(args.get("body", "")),
                    goal=str(args.get("goal", "")),
                    evidence=str(args.get("evidence", "")),
                    support_files=support_files,
                )
            elif action == "revise":
                result = self._service.revise(
                    str(args.get("proposal_id", "")),
                    body=str(args.get("body", "")),
                    description=args.get("description"),
                )
            elif action == "list":
                result = {"proposals": self._service.list()}
            elif action == "inspect":
                result = self._service.inspect(str(args.get("proposal_id", "")))
            elif action == "apply":
                result = self._service.apply(str(args.get("proposal_id", "")))
                if result.get("ok") and self._on_reload is not None:
                    try:
                        self._on_reload()
                        result = {**result, "skill_store_reloaded": True}
                    except Exception as reload_err:
                        result = {
                            **result,
                            "skill_store_reloaded": False,
                            "reload_error": str(reload_err),
                        }
            elif action == "reject":
                result = self._service.reject(str(args.get("proposal_id", "")), str(args.get("reason", "")))
            elif action == "quarantine":
                result = self._service.quarantine(str(args.get("proposal_id", "")), str(args.get("reason", "")))
            elif action == "rollback":
                result = self._service.rollback(str(args.get("rollback_id", "")))
                if result.get("ok") and self._on_reload is not None:
                    try:
                        self._on_reload()
                        result = {**result, "skill_store_reloaded": True}
                    except Exception as reload_err:
                        result = {
                            **result,
                            "skill_store_reloaded": False,
                            "reload_error": str(reload_err),
                        }
            else:
                return ToolResult(success=False, output="", error=f"unknown action {action}")
            return ToolResult(success=True, output=json.dumps(result, indent=2))
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


def _parse_support_files(raw: Any) -> list[dict[str, str]] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    return None


def create_skill_workshop_tool(
    workspace: str | None = None,
    skills_dir: str | None = None,
    *,
    on_reload: Any = None,
) -> Tool:
    return SkillWorkshopTool(workspace, skills_dir, on_reload=on_reload)
