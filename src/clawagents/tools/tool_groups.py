"""Active tool groups — shrink schemas without deleting registered tools.

Luna / GPT-5.6 start on a small *core* surface. Optional groups (web, git,
pty, …) stay registered but hidden from ``list()`` / native schemas until
``activate_tool_group`` brings them in.
"""

from __future__ import annotations

from typing import Any

from clawagents.tools.registry import ToolResult

# Shared discovery / HITL surface for every mode profile.
_CONTROL_PLANE: frozenset[str] = frozenset(
    {
        "think",
        "ask_user",
        "ask_user_question",
        "list_skills",
        "use_skill",
        "retrieve_tool_result",
        "tool_discover",
        "tool_describe",
        "tool_profile",
        "activate_tool_group",
        "search_history",
    }
)

# Explore / Plan — no mutating tools, no shell.
READ_ONLY_TOOL_NAMES: frozenset[str] = _CONTROL_PLANE | frozenset(
    {
        "ls",
        "read_file",
        "grep",
        "glob",
        "hashline_read",
        "hashline_grep",
        "write_plan",
        "enter_plan_mode",
        "exit_plan_mode",
        "tree",
    }
)

# Default coding — prefer hashline over overlapping edit APIs; drop rarely-needed
# helpers (insert_lines / tool_program / read_and_grep) until activated.
CODING_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | frozenset(
    {
        "write_file",
        "edit_file",
        "hashline_edit",
        "apply_patch",
        "execute",
        "write_todos",
        "update_todo",
        "diff",
    }
)

# Goal / autopilot — coding + subagent task + todos already included.
GOAL_TOOL_NAMES: frozenset[str] = CODING_TOOL_NAMES | frozenset({"task"})

# Back-compat alias (full historical core before mode split).
CORE_TOOL_NAMES: frozenset[str] = CODING_TOOL_NAMES | frozenset(
    {
        "insert_lines",
        "tool_program",
        "read_and_grep",
        "task",
    }
)

# Optional groups unlocked via activate_tool_group (includes overlapping editors).
TOOL_GROUPS_EXTRA_CODING: dict[str, frozenset[str]] = {
    "editors_extra": frozenset({"insert_lines", "tool_program", "read_and_grep"}),
}

TOOL_GROUPS: dict[str, frozenset[str]] = {
    "web": frozenset({"web_fetch", "web_search"}),
    "git": frozenset({"git_status", "git_diff", "git_commit", "git_undo_ai"}),
    "worktree": frozenset({"worktree_create", "worktree_list", "worktree_remove"}),
    "pty": frozenset({"pty_start", "pty_keys", "pty_screen", "pty_wait", "pty_stop"}),
    "marketplace": frozenset({"marketplace_install", "marketplace_list"}),
    "background": frozenset(
        {"task_create", "task_status", "task_output", "task_stop", "task_list"}
    ),
    "memory": frozenset(
        {
            "memory_search",
            "memory_view",
            "memory_replace",
            "memory_append",
            "rehydrate_ledger",
            "record_ledger",
            "repo_map",
        }
    ),
    "checkpoints": frozenset(
        {
            "checkpoint_create",
            "checkpoint_restore",
            "checkpoint_list",
            "checkpoint_diff",
            "snapshot_diff",
        }
    ),
    "hunks": frozenset({"hunk_list", "hunk_accept", "hunk_reject"}),
    "skills_extra": frozenset({"skill_workshop"}),
    "mcp": frozenset({"mcp_auth"}),
    **TOOL_GROUPS_EXTRA_CODING,
}


def group_names() -> list[str]:
    return sorted(TOOL_GROUPS.keys())


def tools_in_group(group: str) -> frozenset[str]:
    return TOOL_GROUPS.get(group.strip().lower(), frozenset())


def _profile_for_mode(chat_mode: str | None, *, goal_mode: bool = False) -> frozenset[str]:
    mode = (chat_mode or "").strip().lower()
    if goal_mode or mode == "goal":
        return GOAL_TOOL_NAMES
    if mode in ("read_only", "plan", "ask"):
        return READ_ONLY_TOOL_NAMES
    return CODING_TOOL_NAMES


def apply_core_active_profile(registry: Any) -> list[str]:
    """Restrict to coding profile (legacy entrypoint)."""
    return apply_mode_active_profile(registry, chat_mode="auto", goal_mode=False)


def apply_mode_active_profile(
    registry: Any,
    *,
    chat_mode: str | None = None,
    goal_mode: bool = False,
) -> list[str]:
    """Restrict schemas/execution to a mode profile ∩ registered tools."""
    registered = {t.name for t in registry.list_registered()}
    wanted = _profile_for_mode(chat_mode, goal_mode=goal_mode)
    active = (wanted & registered) | ({"activate_tool_group"} & registered)
    for name in ("tool_discover", "tool_describe", "tool_profile"):
        if name in registered:
            active.add(name)
    registry.set_active_tools(active)
    return sorted(active)


class ActivateToolGroupTool:
    """Expose optional tool groups on demand (does not unregister anything)."""

    name = "activate_tool_group"
    description = (
        "Activate an optional tool group so those tools appear in the schema "
        "and can execute. Groups: "
        + ", ".join(group_names())
        + ". Call with group='list' to see groups and which tools they unlock."
    )
    parameters = {
        "group": {
            "type": "string",
            "description": "Group name, or 'list' to enumerate groups.",
            "required": True,
        }
    }
    parallel_safe = True
    keywords = ["tools", "activate", "browser", "web", "git"]

    def __init__(self, registry: Any):
        self._registry = registry

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        import json

        group = str(args.get("group") or "").strip().lower()
        if not group or group in ("list", "help", "?"):
            registered = {t.name for t in self._registry.list_registered()}
            rows = []
            for g in group_names():
                members = sorted(tools_in_group(g) & registered)
                if not members:
                    continue
                rows.append({"group": g, "tools": members, "count": len(members)})
            active = sorted(self._registry.active_tool_names() or registered)
            return ToolResult(
                True,
                json.dumps({"groups": rows, "active_count": len(active), "active": active}),
            )
        members = tools_in_group(group)
        if not members:
            return ToolResult(
                False,
                "",
                f"Unknown group {group!r}. Use group='list'. Available: {', '.join(group_names())}",
            )
        registered = {t.name for t in self._registry.list_registered()}
        added = sorted(members & registered)
        if not added:
            return ToolResult(
                False, "", f"Group {group!r} has no registered tools in this session."
            )
        self._registry.activate_tools(added)
        return ToolResult(
            True,
            json.dumps(
                {
                    "activated": group,
                    "tools": added,
                    "active_count": len(self._registry.active_tool_names() or []),
                }
            ),
        )
