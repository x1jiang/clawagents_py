"""Feature flags for ClawAgents.

Inspired by Claude Code's build-time feature() system — but runtime-based
so features can be toggled via environment variables without rebuilding.

Usage:
    from clawagents.config.features import is_enabled

    if is_enabled("micro_compact"):
        messages = _micro_compact_tool_results(messages)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


# ─── Feature Registry ─────────────────────────────────────────────────────
# Each feature maps to an env var. Default values control whether the feature
# is opt-in (default "0") or opt-out (default "1").

_FEATURE_DEFAULTS: dict[str, str] = {
    # Quick wins — enabled by default
    "micro_compact":        "1",   # Clear old tool result content aggressively
    "file_snapshots":       "1",   # Snapshot files before write tools modify them
    "cache_tracking":       "0",   # Log prompt cache hit rates from API responses
    "context_ledger":       "1",   # Commit-boundary restorable ledger
    "core_memory":          "1",   # Editable core memory blocks
    "shadow_checkpoints":   "1",   # Shadow-git turn checkpoints after mutating tools
    "auto_verify":          "0",   # Auto lint/test after edits (can be slow)
    "repo_map_inject":      "0",   # Inject ranked repo map into dynamic prompt
    "memory_bank":          "1",   # Optional .clawagents/memory-bank/* briefs
    "fact_store":           "1",   # Local superseding facts from lessons
    "codeact":              "1",   # Allow action_mode=code CodeAct loop

    # Medium effort — opt-in
    "typed_memory":         "0",   # Parse frontmatter in memory files for type-based recall
    "wal":                  "0",   # Write-ahead logging for crash recovery
    "permission_rules":     "1",   # Declarative tool permission rules (deny wins)
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

    # Grok-Build inspired (v6.14) — on by default where safe
    "plan_approval":        "1",   # Host gate on exit_plan_mode when callback set
    "task_worktree":        "1",   # Allow task(isolation=worktree)
    "hunk_review":          "1",   # Attributed hunk accept/reject tools
    "compact_reinject_plan": "1",  # Re-inject plan reminder into compaction carryover
    "compact_tool_pair_safe": "1", # Snap protect_last_n so tool pairs stay intact
    "full_replace_compaction": "1",  # Grok-style full-replace assemble after summarize
    "marketplace":          "1",   # Skill/plugin install from path or git
    "os_sandbox_profiles":  "1",   # Named OS sandbox profile abstraction
    "incremental_repo_map": "1",   # mtime-cached scope graph for repo_map
    "autopilot_loop":       "1",   # plan→execute→verify autopilot driver

    # Skills strategy (Grok-inspired, layered on progressive disclosure) — v6.14.2
    "skill_when_to_use":    "1",   # Parse/list when-to-use; boost ranking
    "skill_path_gating":    "1",   # Hide path-gated skills until matching files touched
    "skill_substitutions":  "1",   # $ARGUMENTS / ${SKILL_DIR} / ${SESSION_ID} in bodies
    "skill_hot_reload":     "1",   # Rescan skill dirs on mtime / workshop apply
    "skill_auto_suggest":   "1",   # High-confidence use_skill nudge (no auto-load)

    # v6.15 product surfaces
    "goal_autopilot":       "1",   # Grok-style /goal planner→verify→strategist
    "prefire_compaction":   "1",   # Two-pass: summarize before hard cliff
    "mid_turn_interject":   "1",   # Accept queued user redirect mid-loop

    # v6.17 Grok-Build parity pack
    "smart_memory":         "1",   # Access boost + temporal decay + blake2 dedup
    "memory_dream":         "1",   # Dream consolidation into MEMORY.md
    "memory_flush":         "1",   # Pre-compaction memory flush
    "hybrid_memory_search": "1",   # FTS5 BM25-style + MMR hybrid recall
    "pty_sessions":         "1",   # Interactive PTY shell sessions
    "hashline_tools":       "1",   # Grok hashline_read / hashline_edit (additive)
    "execute_background":   "1",   # Optional is_background on execute tool
    "rtk_wrap":             "1",   # Auto-wrap noisy execute cmds with rtk (if installed)
    "aggressive_tool_crush": "1",  # Lower crush thresholds in agent_loop (not hooks)
    "execute_shell_session": "1",  # Persist cwd across execute (Grok shell-state slice)
    "execute_shell_env":    "1",   # Sticky env overlay across execute (with shell_session)
    "execute_auto_background": "1",  # On FG timeout, adopt process as background job
    "execute_streaming":    "1",   # Progressive stdout/stderr via tool_progress events
    "edit_file_create_empty": "1",  # Advertise create_if_missing on edit_file
    "structured_output":    "1",   # Native provider json_schema / response_format
    "doom_loop":            "1",   # Generation tail-repetition resample
    "history_then_steps":   "1",   # Graduated compaction mode
    "compaction_segments":  "1",   # Greppable segment_NNN.md + INDEX.md
    "hunk_watcher":         "1",   # External edit attribution via mtime watch
    "session_rewind":       "1",   # Rewind to prompt N (files + conversation)
    "hook_taxonomy":        "0",   # Opt-in; requires external_hooks too (was RCE default-on)
    "sandbox_fail_closed":  "0",   # Refuse soft-fallback; secret path deny binds
    "provider_circuit_breaker": "0",  # Off by default — concurrency burns retries on BreakerOpen
    "tool_error_traceback": "0",   # Include short traceback in ToolResult.error (also CLAW_DEBUG)
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
    """Explicitly override feature flags (useful for constructor injection).

    Unknown flag names are applied but logged at WARNING — a typo like
    ``micro_compat`` (for ``micro_compact``) is otherwise a silent no-op that
    leaves the developer believing they toggled a feature they didn't.
    """
    global _resolved
    if _resolved is None:
        _resolved = _resolve_features()
    unknown = [k for k in overrides if k not in _FEATURE_DEFAULTS]
    if unknown:
        logger.warning(
            "set_overrides: unknown feature flag(s) %s — check spelling against "
            "clawagents.config.features._FEATURE_DEFAULTS (known: %s)",
            ", ".join(sorted(unknown)),
            ", ".join(sorted(_FEATURE_DEFAULTS)),
        )
    for k, v in overrides.items():
        _resolved[k] = bool(v)


@contextmanager
def temporary_overrides(overrides: dict[str, bool]) -> Iterator[None]:
    """Apply feature overrides for a scope, restoring prior values on exit.

    ``run_agent_graph(features=...)`` uses this so per-invoke flags do not
    leak into subsequent runs in the same process.
    """
    global _resolved
    if _resolved is None:
        _resolved = _resolve_features()
    prior = dict(_resolved)
    set_overrides(overrides)
    try:
        yield
    finally:
        _resolved.clear()
        _resolved.update(prior)
