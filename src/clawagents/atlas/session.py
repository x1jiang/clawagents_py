"""Per-run ATLAS session state for the ClawAgents harness."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PendingKind(str, Enum):
    ADVISORY = "advisory"
    FINAL = "final"


@dataclass
class PendingReflection:
    kind: PendingKind
    checkpoint_id: str
    gate_label: str
    full: bool
    format_attempt: int = 0
    pinned_status: str | None = None


@dataclass
class AtlasRunSession:
    """Mutable state for one ClawAgents run under ATLAS supervision."""

    atlas_session: Any  # atlas_runtime.Session
    config: dict[str, Any]
    run_id: str
    task: str
    segment_start: int = 1  # after system + first user
    checkpoint_count: int = 0
    repair_attempts: int = 0
    tool_calls_since_failure_nudge: int = 0
    last_failure_nudge_at: float = 0.0
    pending: PendingReflection | None = None
    final_gate_allowed: bool | None = None
    finalized: bool = False
    skip_rethink_this_round: bool = False

    @property
    def taxonomy_id(self) -> str:
        return str(self.atlas_session.delivery.taxonomy_id)

    @property
    def taxonomy(self) -> dict[str, Any]:
        return dict(self.atlas_session.delivery.taxonomy)

    @property
    def runtime_protocol(self) -> str:
        return str(self.atlas_session.delivery.runtime_protocol or "")

    @property
    def known_code_ids(self) -> list[str]:
        codes = self.taxonomy.get("codes") or []
        return [str(c["id"]) for c in codes if isinstance(c, dict) and "id" in c]

    def new_checkpoint_id(self) -> str:
        return uuid.uuid4().hex

    def failure_throttle_ok(self) -> bool:
        # First failure nudge is always allowed; later ones need spacing.
        if self.last_failure_nudge_at == 0.0:
            return True
        throttle = int(self.config.get("failure_throttle_calls") or 5)
        recency = float(self.config.get("failure_recency_seconds") or 30)
        if self.tool_calls_since_failure_nudge < throttle:
            return False
        if (time.monotonic() - self.last_failure_nudge_at) < recency:
            return False
        return True

    def mark_failure_nudge(self) -> None:
        self.tool_calls_since_failure_nudge = 0
        self.last_failure_nudge_at = time.monotonic()

    def note_tool_call(self) -> None:
        self.tool_calls_since_failure_nudge += 1
