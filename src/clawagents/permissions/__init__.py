"""Permission system for clawagents.

Currently exposes :class:`PermissionMode` and :data:`WRITE_CLASS_TOOLS`.
Inspired by claude-code-main/src/utils/permissions/PermissionMode.ts.

Plan-approval helpers live in :mod:`clawagents.permissions.plan_approval`
and are imported from there (not re-exported here) to avoid a circular
import with :mod:`clawagents.run_context`.
"""

from clawagents.permissions.mode import (
    PermissionMode,
    PermissionDecision,
    SENSITIVE_PATH_PATTERNS,
    WRITE_CLASS_TOOLS,
    evaluate_tool_permission,
    is_write_class_tool,
    permission_mode_from_string,
)

__all__ = [
    "PermissionMode",
    "PermissionDecision",
    "SENSITIVE_PATH_PATTERNS",
    "WRITE_CLASS_TOOLS",
    "evaluate_tool_permission",
    "is_write_class_tool",
    "permission_mode_from_string",
]
