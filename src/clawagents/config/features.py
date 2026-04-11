"""Feature flags for ClawAgents.

Inspired by Claude Code's build-time feature() system — but runtime-based
so features can be toggled via environment variables without rebuilding.

Usage:
    from clawagents.config.features import is_enabled

    if is_enabled("micro_compact"):
        messages = _micro_compact_tool_results(messages)
"""

from __future__ import annotations

import os


# ─── Feature Registry ─────────────────────────────────────────────────────
# Each feature maps to an env var. Default values control whether the feature
# is opt-in (default "0") or opt-out (default "1").

_FEATURE_DEFAULTS: dict[str, str] = {
    # Quick wins — enabled by default
    "micro_compact":        "1",   # Clear old tool result content aggressively
    "file_snapshots":       "1",   # Snapshot files before write tools modify them
    "cache_tracking":       "0",   # Log prompt cache hit rates from API responses

    # Medium effort — opt-in
    "typed_memory":         "0",   # Parse frontmatter in memory files for type-based recall
    "wal":                  "0",   # Write-ahead logging for crash recovery
    "permission_rules":     "0",   # Declarative tool permission rules
    "background_memory":    "0",   # Continuous memory extraction every N turns

    # New features (inspired by claw-code-main)
    "cache_boundary":       "1",   # Prompt cache boundary optimization for Anthropic
    "session_persistence":  "0",   # Session persistence + resume
    "error_taxonomy":       "1",   # Structured error classification + recovery recipes
    "external_hooks":       "0",   # External shell hook system

    # Complex — opt-in
    "forked_agents":        "0",   # Background forked agent pattern
    "coordinator":          "0",   # Coordinator/swarm orchestration mode
    "transcript_archival":  "0",   # Archive full messages to markdown before compaction
    "credential_proxy":     "0",   # Credential proxy for sandboxed sub-agents
}

# Env var prefix: CLAW_FEATURE_MICRO_COMPACT=1
_ENV_PREFIX = "CLAW_FEATURE_"


def _resolve_features() -> dict[str, bool]:
    """Resolve all feature flags from environment, with defaults."""
    result: dict[str, bool] = {}
    for name, default in _FEATURE_DEFAULTS.items():
        env_key = _ENV_PREFIX + name.upper()
        value = os.environ.get(env_key, default)
        result[name] = value in ("1", "true", "yes", "on")
    return result


# Lazy singleton — resolved once on first access
_resolved: dict[str, bool] | None = None


def _get_features() -> dict[str, bool]:
    global _resolved
    if _resolved is None:
        _resolved = _resolve_features()
    return _resolved


def is_enabled(feature: str) -> bool:
    """Check if a feature flag is enabled.

    Args:
        feature: Feature name (e.g., "micro_compact", "file_snapshots")

    Returns:
        True if the feature is enabled via env var or default.
    """
    return _get_features().get(feature, False)


def all_features() -> dict[str, bool]:
    """Return a copy of all feature flags and their current state."""
    return dict(_get_features())


def reset() -> None:
    """Reset cached features (useful for testing)."""
    global _resolved
    _resolved = None

def set_overrides(overrides: dict[str, bool]) -> None:
    """Explicitly override feature flags (useful for constructor injection)."""
    global _resolved
    if _resolved is None:
        _resolved = _resolve_features()
    for k, v in overrides.items():
        _resolved[k] = bool(v)
