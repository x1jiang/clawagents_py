"""Versioned capability flags for host/sidecar integration.

VS Code (and other clients) should probe ``GET /capabilities`` (or import this
module) instead of monkey-patching private helpers or sniffing
``inspect.signature`` for every feature.
"""

from __future__ import annotations

from typing import Any

# Bump the contract version when removing/renaming keys (additive is fine).
CAPABILITIES_CONTRACT_VERSION = 1

CAPABILITIES: dict[str, Any] = {
    "contract_version": CAPABILITIES_CONTRACT_VERSION,
    # Gemini tool schemas always emit ARRAY ``items`` (no sidecar patch needed).
    "gemini_array_items": True,
    # create_claw_agent(workspace=…) scopes tools without process chdir.
    "workspace_scoped_agent": True,
    # ToolResult.raw_output preserves full dumps for artifact archival.
    "raw_tool_output": True,
    # prepare_tool_output_for_context accepts workspace=
    "artifact_workspace_arg": True,
}


def get_capabilities() -> dict[str, Any]:
    """Return a copy of the capability map (safe to serialize)."""
    return dict(CAPABILITIES)
