"""Structured state preserved across context compaction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

CARRYOVER_METADATA_KEY = "compaction_carryover"


@dataclass
class CompactionCarryover:
    """State that should survive even when older transcript turns are compacted."""

    task_focus: str | None = None
    recent_files: list[str] = field(default_factory=list)
    recent_work_log: list[str] = field(default_factory=list)
    invoked_skills: list[str] = field(default_factory=list)
    active_workers: list[str] = field(default_factory=list)
    channel_log: list[dict[str, Any]] = field(default_factory=list)
    plan_reminder: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any((
            self.task_focus,
            self.recent_files,
            self.recent_work_log,
            self.invoked_skills,
            self.active_workers,
            self.channel_log,
            self.plan_reminder,
            self.metadata,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_focus": self.task_focus,
            "recent_files": list(self.recent_files),
            "recent_work_log": list(self.recent_work_log),
            "invoked_skills": list(self.invoked_skills),
            "active_workers": list(self.active_workers),
            "channel_log": list(self.channel_log),
            "plan_reminder": self.plan_reminder,
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if self.is_empty():
            return ""

        lines = ["## Carryover State"]
        if self.task_focus:
            lines.append(f"- Task focus: {_clip(self.task_focus, 500)}")
        if self.plan_reminder:
            lines.append("- Active plan:")
            lines.append(_clip(self.plan_reminder, 1500))
        if self.recent_files:
            lines.append("- Recent files: " + ", ".join(self.recent_files[:12]))
        if self.recent_work_log:
            lines.append("- Recent work:")
            lines.extend(f"  - {_clip(item, 500)}" for item in self.recent_work_log[:12])
        if self.invoked_skills:
            lines.append("- Invoked skills: " + ", ".join(self.invoked_skills[:12]))
        if self.active_workers:
            lines.append("- Active workers: " + ", ".join(self.active_workers[:12]))
        if self.channel_log:
            lines.append("- Recent channel messages:")
            for item in self.channel_log[:8]:
                channel = item.get("channel_id") or item.get("channelId") or "channel"
                conversation = item.get("conversation_id") or item.get("conversationId") or "conversation"
                body = _clip(str(item.get("body", "")), 300)
                lines.append(f"  - {channel}:{conversation}: {body}")
        if self.metadata:
            try:
                encoded = json.dumps(self.metadata, sort_keys=True, ensure_ascii=False)
            except TypeError:
                encoded = json.dumps({k: str(v) for k, v in self.metadata.items()}, sort_keys=True)
            lines.append(f"- Metadata: {_clip(encoded, 500)}")
        return "\n".join(lines)


def set_compaction_carryover(
    run_context: Any,
    *,
    task_focus: str | None = None,
    recent_files: list[str] | None = None,
    recent_work_log: list[str] | None = None,
    invoked_skills: list[str] | None = None,
    active_workers: list[str] | None = None,
    channel_log: list[dict[str, Any]] | None = None,
    plan_reminder: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CompactionCarryover:
    """Attach structured compaction carryover state to a RunContext."""

    carryover = CompactionCarryover(
        task_focus=task_focus,
        recent_files=_string_list(recent_files),
        recent_work_log=_string_list(recent_work_log),
        invoked_skills=_string_list(invoked_skills),
        active_workers=_string_list(active_workers),
        channel_log=[dict(item) for item in (channel_log or []) if isinstance(item, dict)],
        plan_reminder=_optional_str(plan_reminder),
        metadata=dict(metadata or {}),
    )
    _metadata(run_context)[CARRYOVER_METADATA_KEY] = carryover.to_dict()
    return carryover


def get_compaction_carryover(run_context: Any | None, *, task_context: str = "") -> CompactionCarryover:
    """Read carryover state from a RunContext metadata bag."""

    if run_context is None:
        return CompactionCarryover(task_focus=task_context or None)

    raw = _metadata(run_context).get(CARRYOVER_METADATA_KEY, {})
    carryover = normalize_compaction_carryover(raw)
    if not carryover.task_focus and task_context:
        carryover.task_focus = task_context
    # Re-inject active plan when feature enabled and not already set
    try:
        from clawagents.config.features import is_enabled

        if is_enabled("compact_reinject_plan") and not carryover.plan_reminder:
            from clawagents.tools.context_tools import load_plan_preamble

            workspace = None
            if hasattr(run_context, "_metadata"):
                workspace = run_context._metadata.get("workspace")
            plan = load_plan_preamble(workspace=workspace) if workspace else load_plan_preamble()
            if plan:
                carryover.plan_reminder = plan
    except Exception:
        pass
    return carryover


def normalize_compaction_carryover(value: Any) -> CompactionCarryover:
    if isinstance(value, CompactionCarryover):
        return value
    if not isinstance(value, dict):
        return CompactionCarryover()
    return CompactionCarryover(
        task_focus=_optional_str(value.get("task_focus") or value.get("taskFocus")),
        recent_files=_string_list(value.get("recent_files") or value.get("recentFiles")),
        recent_work_log=_string_list(value.get("recent_work_log") or value.get("recentWorkLog")),
        invoked_skills=_string_list(value.get("invoked_skills") or value.get("invokedSkills")),
        active_workers=_string_list(value.get("active_workers") or value.get("activeWorkers")),
        channel_log=[dict(item) for item in (value.get("channel_log") or value.get("channelLog") or []) if isinstance(item, dict)],
        plan_reminder=_optional_str(value.get("plan_reminder") or value.get("planReminder")),
        metadata=dict(value.get("metadata") or {}),
    )


def _metadata(run_context: Any) -> dict[str, Any]:
    meta = getattr(run_context, "_metadata", None)
    if isinstance(meta, dict):
        return meta
    setattr(run_context, "_metadata", {})
    return getattr(run_context, "_metadata")


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return []


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "..."
