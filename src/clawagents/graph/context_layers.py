"""Pluggable context injection layers for the agent system prompt.

Each layer encapsulates a discrete source of dynamic context (memories,
goals, workspace facts, repo maps, etc.) behind a uniform protocol.  The
bootstrapper iterates layers in registration order; a layer whose
``inject()`` returns ``None`` is silently skipped.

Adding a new context source only requires implementing a ``ContextLayer``
and registering it in ``DEFAULT_LAYERS`` — the agent loop itself stays
untouched.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class ContextLayer(Protocol):
    """A pluggable system-prompt context injection layer.

    Implementations check their own applicability (feature flags,
    ``skip_memory``, etc.) inside ``inject()`` and return ``None`` to skip.
    """

    @property
    def name(self) -> str:
        """Human-readable name for logging / observability."""
        ...

    def inject(self, run_context: Any) -> str | None:
        """Return context text to append to the system prompt, or ``None``."""
        ...


# ── Layer Implementations ───────────────────────────────────────────────


class LessonLayer:
    """PTRL Layer 1: Pre-run lesson injection from past trajectories."""

    name = "lessons"

    def __init__(self, *, learn: bool = False) -> None:
        self._learn = learn

    def inject(self, run_context: Any) -> str | None:
        if not self._learn:
            return None
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.trajectory.lessons import build_lesson_preamble

            return build_lesson_preamble() or None
        except Exception:
            logger.debug("lesson layer failed", exc_info=True)
            return None


class GoalReminderLayer:
    """Goal autopilot standing reminder (preferred long-horizon gate)."""

    name = "goal_reminder"

    def inject(self, run_context: Any) -> str | None:
        try:
            from clawagents.config.features import is_enabled
            from clawagents.goal import get_goal_tracker, goal_system_reminder

            meta = getattr(run_context, "_metadata", None)
            if not (isinstance(meta, dict) and meta.get("goal_mode")):
                return None
            if not is_enabled("goal_autopilot"):
                return None
            tracker = get_goal_tracker(run_context)
            reminder = goal_system_reminder(tracker.state if tracker else None)
            if not reminder:
                return None
            return (
                "<!--claw:goal-reminder-->\n"
                + reminder
                + "\n<!--/claw:goal-reminder-->"
            )
        except Exception:
            logger.debug("goal reminder layer failed", exc_info=True)
            return None


class CoreMemoryLayer:
    """Core memory: persistent user/project facts."""

    name = "core_memory"

    def inject(self, run_context: Any) -> str | None:
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("core_memory"):
                return None
            from clawagents.memory.core_memory import load_core_memory

            return load_core_memory() or None
        except Exception:
            logger.debug("core memory layer failed", exc_info=True)
            return None


class ContextLedgerLayer:
    """Context ledger: structured key-value context store."""

    name = "context_ledger"

    def inject(self, run_context: Any) -> str | None:
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("context_ledger"):
                return None
            from clawagents.memory.context_ledger import load_ledger_preamble

            return load_ledger_preamble() or None
        except Exception:
            logger.debug("context ledger layer failed", exc_info=True)
            return None


class MemoryBankLayer:
    """Memory bank: categorised long-term memory."""

    name = "memory_bank"

    def inject(self, run_context: Any) -> str | None:
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("memory_bank"):
                return None
            from clawagents.memory.core_memory import (
                ensure_memory_bank_stubs,
                load_memory_bank_preamble,
            )

            ensure_memory_bank_stubs()
            return load_memory_bank_preamble() or None
        except Exception:
            logger.debug("memory bank layer failed", exc_info=True)
            return None


class FactStoreLayer:
    """Fact store: structured factual assertions."""

    name = "fact_store"

    def inject(self, run_context: Any) -> str | None:
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("fact_store"):
                return None
            from clawagents.memory.facts import live_facts_preamble

            return live_facts_preamble() or None
        except Exception:
            logger.debug("fact store layer failed", exc_info=True)
            return None


class PlanLayer:
    """Active plan preamble (always injected when available)."""

    name = "plan"

    def inject(self, run_context: Any) -> str | None:
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.tools.context_tools import load_plan_preamble

            return load_plan_preamble() or None
        except Exception:
            logger.debug("plan layer failed", exc_info=True)
            return None


class RepoMapLayer:
    """Ranked repository map for workspace orientation."""

    name = "repo_map"

    def inject(self, run_context: Any) -> str | None:
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("repo_map_inject"):
                return None
            from clawagents.memory.repo_map import build_repo_map

            return build_repo_map(max_chars=3_500) or None
        except Exception:
            logger.debug("repo map layer failed", exc_info=True)
            return None


class WorkspaceEnvLayer:
    """Workspace environment facts (cwd, git status, sandbox profile)."""

    name = "workspace_env"

    def inject(self, run_context: Any) -> str | None:
        if getattr(run_context, "skip_memory", False):
            return None
        try:
            from clawagents.tools.git_tools import is_git_work_tree

            ws = str(getattr(run_context, "workspace", None) or Path.cwd())
            git_ok = is_git_work_tree(ws)
            scratch = tempfile.gettempdir()
            meta = getattr(run_context, "_metadata", None)
            sb_name = "workspace"
            if isinstance(meta, dict):
                sb_name = str(meta.get("sandbox_profile") or sb_name)
            lines = [
                "## Workspace env",
                f"- workspace: `{ws}`",
                f"- is_git_repo: {'true' if git_ok else 'false'}",
                f"- sandbox: `{sb_name}`",
                f"- scratch_dir: `{scratch}` (also /tmp when sandbox allows)",
            ]
            if not git_ok:
                lines.append(
                    "- Prefer `snapshot_diff` to review edits (no git)."
                )
            else:
                lines.append(
                    "- Prefer `git_status` / `git_diff` to review edits."
                )
            lines.append(
                "- Do not chain `&& git …` after syntax checks when is_git_repo is false."
            )
            if sb_name == "off":
                lines.append(
                    "- OS sandbox is off — home config CLIs (gcloud/aws/docker) may run."
                )
            return "\n".join(lines)
        except Exception:
            logger.debug("workspace env layer failed", exc_info=True)
            return None


# ── Layer Registry ──────────────────────────────────────────────────────


def build_default_layers(*, learn: bool = False) -> list[ContextLayer]:
    """Construct the default layer stack in canonical injection order."""
    return [
        LessonLayer(learn=learn),
        GoalReminderLayer(),
        CoreMemoryLayer(),
        ContextLedgerLayer(),
        MemoryBankLayer(),
        FactStoreLayer(),
        PlanLayer(),
        RepoMapLayer(),
        WorkspaceEnvLayer(),
    ]


def collect_dynamic_context(
    layers: list[ContextLayer],
    run_context: Any,
    *,
    emit: Any = None,
) -> str:
    """Run all layers and return the joined dynamic preamble.

    Each layer is individually error-guarded — a failing layer never
    blocks the agent run.
    """
    parts: list[str] = []
    for layer in layers:
        try:
            part = layer.inject(run_context)
            if part:
                parts.append(part)
                if emit is not None:
                    emit("context", {"message": f"injected {layer.name}"})
        except Exception:
            logger.debug("context layer %s failed", layer.name, exc_info=True)
    return "\n\n".join(parts)
