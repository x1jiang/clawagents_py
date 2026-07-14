"""ATLAS adaptive failure-taxonomy integration for ClawAgents.

Optional dependency: install ``atlas-skill`` from the ATLAS GitHub repo.

Enable with ``create_claw_agent(atlas=True)`` or ``CLAW_ATLAS=1`` and an
``atlas.json`` (or ``CLAW_ATLAS_CONFIG``) in the project.
"""

from __future__ import annotations

from clawagents.atlas._runtime import require_atlas_runtime
from clawagents.atlas.adapter import AtlasAdapter
from clawagents.atlas.config import resolve_atlas_config, resolve_atlas_enabled

__all__ = [
    "AtlasAdapter",
    "require_atlas_runtime",
    "resolve_atlas_config",
    "resolve_atlas_enabled",
]
