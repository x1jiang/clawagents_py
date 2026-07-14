"""Resolve ATLAS enablement and ``atlas.json`` for ClawAgents runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_TRACE_OUTPUT = Path(".clawagents") / "atlas-program"
DEFAULT_CONFIG_NAME = "atlas.json"
_ENV_ATLAS = "CLAW_ATLAS"
_ENV_ATLAS_CONFIG = "CLAW_ATLAS_CONFIG"


def resolve_atlas_enabled(atlas: bool | None) -> bool:
    """Resolve ``atlas`` constructor/env flag (default off)."""
    if atlas is not None:
        return bool(atlas)
    return os.environ.get(_ENV_ATLAS, "").lower() in ("1", "true", "yes", "on")


def resolve_atlas_config_path(atlas_config: str | Path | None = None) -> Path | None:
    """Return the config path to load, or ``None`` when no file should be read.

    Precedence: explicit ``atlas_config`` → ``CLAW_ATLAS_CONFIG`` → ``./atlas.json``
    if present.
    """
    if atlas_config is not None:
        return Path(atlas_config).expanduser()
    env_path = os.environ.get(_ENV_ATLAS_CONFIG, "").strip()
    if env_path:
        return Path(env_path).expanduser()
    default = Path(DEFAULT_CONFIG_NAME)
    if default.is_file():
        return default
    return None


def resolve_atlas_config(atlas_config: str | Path | None = None) -> dict[str, Any]:
    """Load and normalize ATLAS config, applying ClawAgents defaults.

    When no config file exists, returns a minimal in-memory config with
    ``trace_output`` under ``.clawagents/atlas-program``. ``atlas_model`` remains
    optional until learning calls need it (ATLAS allows MAST warm-up without it
    for some paths; generation requires it).
    """
    from clawagents.atlas._runtime import require_atlas_runtime

    require_atlas_runtime()
    from atlas_runtime import load_atlas_config

    path = resolve_atlas_config_path(atlas_config)
    if path is None:
        cfg: dict[str, Any] = {}
    else:
        cfg = dict(load_atlas_config(path))

    if "trace_output" not in cfg:
        cfg["trace_output"] = (Path.cwd() / DEFAULT_TRACE_OUTPUT).resolve()
    if "gate_exhaustion_policy" not in cfg:
        # Benchmark-friendly default: release best answer when repair budget hits.
        cfg["gate_exhaustion_policy"] = "release"
    if "redact_traces" not in cfg:
        cfg["redact_traces"] = True
    if "recent_activity_messages" not in cfg:
        cfg["recent_activity_messages"] = 8
    if "recent_activity_chars" not in cfg:
        cfg["recent_activity_chars"] = 12_000
    if "failure_throttle_calls" not in cfg:
        cfg["failure_throttle_calls"] = 5
    if "failure_recency_seconds" not in cfg:
        cfg["failure_recency_seconds"] = 30
    if "format_retries" not in cfg:
        cfg["format_retries"] = 2
    if "repair_rounds" not in cfg:
        cfg["repair_rounds"] = int(cfg.get("max_retries") or 3)
    if "dashboard" not in cfg:
        cfg["dashboard"] = False  # opt-in for library use; avoid surprise local servers
    return cfg
