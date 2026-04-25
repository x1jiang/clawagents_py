"""Permission system for clawagents.

Currently exposes :class:`PermissionMode` and :data:`WRITE_CLASS_TOOLS`.
Inspired by claude-code-main/src/utils/permissions/PermissionMode.ts.
"""

from clawagents.permissions.mode import (
    PermissionMode,
    WRITE_CLASS_TOOLS,
    is_write_class_tool,
    permission_mode_from_string,
)

__all__ = [
    "PermissionMode",
    "WRITE_CLASS_TOOLS",
    "is_write_class_tool",
    "permission_mode_from_string",
]
