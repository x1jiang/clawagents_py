"""Active tool groups — shrink schemas without deleting registered tools.

Luna / GPT-5.6 start on a small *core* surface. Optional groups (web, git,
pty, …) stay registered but hidden from ``list()`` / native schemas until
``activate_tool_group`` brings them in.
"""

from __future__ import annotations

from typing import Any

from clawagents.tools.registry import ToolResult

# Always advertised for coding agents (intersection with what's registered).
CORE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "grep",
        "glob",
        "execute",
        "think",
        "write_todos",
        "update_todo",
        "ask_user",
        "ask_user_question",
        "list_skills",
        "use_skill",
        "retrieve_tool_result",
        "tool_discover",
        "tool_describe",
        "tool_profile",
        "activate_tool_group",
        "hashline_read",
        "hashline_grep",
        "hashline_edit",
        "apply_patch",
        "tree",
        "diff",
        "insert_lines",
        "write_plan",
        "enter_plan_mode",
        "exit_plan_mode",
        "search_history",
        "tool_program",
        "task",
        "read_and_grep",
    }
)

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
}


def group_names() -> list[str]:
    return sorted(TOOL_GROUPS.keys())


def tools_in_group(group: str) -> frozenset[str]:
    return TOOL_GROUPS.get(group.strip().lower(), frozenset())


def apply_core_active_profile(registry: Any) -> list[str]:
    """Restrict schemas/execution to core ∩ registered (+ activate_tool_group)."""
    registered = {t.name for t in registry.list_registered()}
    active = (CORE_TOOL_NAMES & registered) | ({"activate_tool_group"} & registered)
    # Keep discovery helpers if present so the model can find hidden groups.
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
