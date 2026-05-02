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

import fnmatch
from dataclasses import dataclass
from enum import Enum


class PermissionMode(str, Enum):
    """Permission modes that gate write-class tools."""

    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS = "bypassPermissions"


SENSITIVE_PATH_PATTERNS: tuple[str, ...] = (
    "*/.ssh/*",
    "*/.aws/credentials",
    "*/.aws/config",
    "*/.config/gcloud/*",
    "*/.azure/*",
    "*/.gnupg/*",
    "*/.docker/config.json",
    "*/.kube/config",
    "*/.clawagents/credentials.json",
)


@dataclass(frozen=True)
class PermissionDecision:
    """Structured tool permission result."""

    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""


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


def evaluate_tool_permission(
    tool_name: str,
    *,
    mode: PermissionMode = PermissionMode.DEFAULT,
    is_read_only: bool = False,
    file_path: str | None = None,
    command: str | None = None,
) -> PermissionDecision:
    """Return a structured permission decision for one tool call."""
    if file_path:
        for candidate in _policy_match_paths(file_path):
            for pattern in SENSITIVE_PATH_PATTERNS:
                if fnmatch.fnmatch(candidate, pattern):
                    return PermissionDecision(
                        allowed=False,
                        reason=(
                            f"Access denied: {file_path} is a sensitive credential path "
                            f"(matched built-in pattern '{pattern}')"
                        ),
                    )

    if mode == PermissionMode.BYPASS:
        return PermissionDecision(True, reason="bypassPermissions allows this tool")
    if is_read_only:
        return PermissionDecision(True, reason="read-only tools are allowed")
    if mode == PermissionMode.PLAN and is_write_class_tool(tool_name):
        return PermissionDecision(False, reason="Plan mode blocks mutating tools until exit_plan_mode")
    if mode == PermissionMode.ACCEPT_EDITS and is_write_class_tool(tool_name):
        return PermissionDecision(True, reason="acceptEdits allows write-class tools")

    reason = "Mutating tools require user confirmation in default mode."
    hint = _command_permission_hint(command)
    if hint:
        reason = f"{reason} {hint}"
    return PermissionDecision(False, requires_confirmation=True, reason=reason)


def _policy_match_paths(file_path: str) -> tuple[str, ...]:
    normalized = file_path.rstrip("/")
    if not normalized:
        return (file_path,)
    return (normalized, normalized + "/")


def _command_permission_hint(command: str | None) -> str:
    if not command:
        return ""
    lowered = command.lower()
    markers = (
        "npm install", "pnpm install", "yarn install", "bun install",
        "pip install", "uv pip install", "poetry install", "cargo install",
        "create-next-app", "npm create ", "pnpm create ", "yarn create ",
        "bun create ", "npx create-", "npm init ", "pnpm init ", "yarn init ",
    )
    if any(marker in lowered for marker in markers):
        return "Package installation and scaffolding commands change the workspace."
    return ""


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
