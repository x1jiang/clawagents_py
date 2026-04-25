"""PermissionMode enum + write-class tool registry.

The permission mode determines how aggressive the tool registry is at gating
state-changing operations. It lives on :class:`~clawagents.run_context.RunContext`
so that hooks, tools, and the registry can all consult the same value.

Modes (mirrors claude-code-main):

- ``DEFAULT`` — normal behavior, no extra gating.
- ``PLAN`` — read-only exploration only. Write-class tools refuse before
  executing. The model is expected to call ``exit_plan_mode`` to leave.
- ``ACCEPT_EDITS`` — auto-approve write-class edits without prompting.
- ``BYPASS`` — bypass all permission prompts (dangerous; opt-in).

The mode is set via the dedicated ``enter_plan_mode`` / ``exit_plan_mode``
tools (see :mod:`clawagents.tools.plan_mode`). Tools never reach into agent
state directly; they only mutate ``run_context.permission_mode``.
"""

from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    """Permission modes that gate write-class tools."""

    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS = "bypassPermissions"


# ─── Write-class tool registry ────────────────────────────────────────────
#
# Tools whose execution mutates state (filesystem, processes, network side
# effects). Listed by canonical tool name. The registry consults this set
# pre-execute when ``run_context.permission_mode == PLAN`` and refuses with
# a structured error.
#
# This list intentionally subsumes ``registry._WRITE_TOOLS`` (used for file
# snapshots) and adds ``execute`` so shell commands are also gated.

WRITE_CLASS_TOOLS: frozenset[str] = frozenset({
    # Filesystem writers
    "write_file",
    "edit_file",
    "create_file",
    "replace_in_file",
    "insert_in_file",
    "patch_file",
    "delete_file",
    # Shell / process
    "execute",
    "exec",
    "bash",
    # Composite / sub-agent tools that may issue writes.
    # (Sub-agents inherit permission_mode in their own run_context, so
    # gating at the parent dispatch site is defensive.)
    "subagent",
    "compose",
})


def is_write_class_tool(tool_name: str) -> bool:
    """Return True if the named tool counts as write-class for plan mode."""
    return tool_name in WRITE_CLASS_TOOLS


def permission_mode_from_string(value: str | None) -> PermissionMode:
    """Coerce a free-form string to a :class:`PermissionMode`.

    Accepts the canonical short names (``default``, ``plan``,
    ``acceptEdits``, ``bypassPermissions``) and the upper-case enum names
    (``DEFAULT``, ``PLAN``, ``ACCEPT_EDITS``, ``BYPASS``). Anything else
    falls back to ``DEFAULT``.
    """
    if not value:
        return PermissionMode.DEFAULT
    s = str(value).strip()
    # Try canonical wire value first.
    for m in PermissionMode:
        if m.value == s:
            return m
    # Try enum name (case-insensitive).
    name = s.upper().replace("-", "_")
    if name == "BYPASS":
        return PermissionMode.BYPASS
    if name == "ACCEPT_EDITS":
        return PermissionMode.ACCEPT_EDITS
    if name == "PLAN":
        return PermissionMode.PLAN
    if name == "DEFAULT":
        return PermissionMode.DEFAULT
    return PermissionMode.DEFAULT
