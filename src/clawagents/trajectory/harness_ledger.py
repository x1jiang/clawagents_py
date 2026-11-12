"""Budget accounting and cost guardrails for Harness² Propose–Probe–Compose.

Tracks ``R = 4k`` rollouts and ``P = 3k`` edit sessions per task, separating
rollout vs. proposer session USD cost, cached hits, retry attempts, and optional
DSpark-style confidence-scheduled skipped stages.

``assert_identity()`` checks for duplicate stage labels *before* checking counts,
catching bugs where two stages collide on the same run ID and silently overwrite
each other while still summing to ``4k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EntryKind = Literal["rollout", "session"]


def expected_rollouts(k: int) -> int:
    """``R = 4k``: ``k`` base + ``2k`` INV/DOM candidates + ``k`` compositions."""
    return 4 * int(k)


def expected_sessions(k: int) -> int:
    """``P = 3k``: ``k`` INV + ``k`` DOM + ``k`` compositions."""
    return 3 * int(k)


class BudgetError(AssertionError):
    """Raised when recorded stages violate the ``4k / 3k`` budget identity."""


class BudgetExceededError(RuntimeError):
    """Raised when cumulative task spend exceeds ``max_usd``."""


@dataclass(frozen=True)
class LedgerEntry:
    kind: EntryKind
    label: str
    usd: float = 0.0
    cached: bool = False
    reused_fingerprint: str = ""
    attempts: int = 1
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class HarnessLedger:
    """Per-task execution, cache, and cost ledger for Propose–Probe–Compose."""

    mode: str
    task_id: str
    k: int
    max_usd: float | None = None
    adaptive_probe: bool = False
    entries: list[LedgerEntry] = field(default_factory=list)

    def _check_budget_before_spend(self) -> None:
        if self.max_usd is not None and self.usd > self.max_usd + 1e-9:
            raise BudgetExceededError(
                f"Harness budget exceeded on {self.task_id}: "
                f"spent ${self.usd:.4f} > max_usd=${self.max_usd:.4f}"
            )

    def check_can_proceed(self) -> None:
        """Raise :class:`BudgetExceededError` if ``max_usd`` has already been reached."""
        if self.max_usd is not None and self.usd >= self.max_usd and self.max_usd >= 0:
            raise BudgetExceededError(
                f"Harness budget cap reached on {self.task_id}: "
                f"spent ${self.usd:.4f} >= max_usd=${self.max_usd:.4f}"
            )

    def record_rollout(
        self,
        label: str,
        usd: float = 0.0,
        *,
        cached: bool = False,
        reused_fingerprint: str = "",
        skipped: bool = False,
        skip_reason: str = "",
    ) -> None:
        entry = LedgerEntry(
            kind="rollout",
            label=label,
            usd=0.0 if (cached or skipped or reused_fingerprint) else float(usd),
            cached=cached,
            reused_fingerprint=reused_fingerprint,
            skipped=skipped,
            skip_reason=skip_reason,
        )
        self.entries.append(entry)
        self._check_budget_before_spend()

    def record_session(
        self,
        label: str,
        usd: float = 0.0,
        *,
        cached: bool = False,
        attempts: int = 1,
        skipped: bool = False,
        skip_reason: str = "",
    ) -> None:
        entry = LedgerEntry(
            kind="session",
            label=label,
            usd=0.0 if (cached or skipped) else float(usd),
            cached=cached,
            attempts=max(1, int(attempts)),
            skipped=skipped,
            skip_reason=skip_reason,
        )
        self.entries.append(entry)
        self._check_budget_before_spend()

    @property
    def rollouts_executed(self) -> int:
        return sum(
            1
            for e in self.entries
            if e.kind == "rollout"
            and not e.skipped
            and not e.reused_fingerprint
            and not e.cached
        )

    @property
    def rollouts_total(self) -> int:
        return sum(1 for e in self.entries if e.kind == "rollout")

    @property
    def rollouts_saved(self) -> int:
        return sum(
            1
            for e in self.entries
            if e.kind == "rollout" and (e.skipped or bool(e.reused_fingerprint) or e.cached)
        )

    @property
    def sessions_executed(self) -> int:
        return sum(
            1
            for e in self.entries
            if e.kind == "session" and not e.skipped and not e.cached
        )

    @property
    def sessions_total(self) -> int:
        return sum(1 for e in self.entries if e.kind == "session")

    @property
    def sessions_saved(self) -> int:
        return sum(1 for e in self.entries if e.kind == "session" and (e.skipped or e.cached))

    @property
    def R(self) -> int:
        return self.rollouts_total

    @property
    def P(self) -> int:
        return self.sessions_total

    @property
    def usd(self) -> float:
        return round(sum(e.usd for e in self.entries), 6)

    @property
    def rollout_usd(self) -> float:
        return round(sum(e.usd for e in self.entries if e.kind == "rollout"), 6)

    @property
    def session_usd(self) -> float:
        return round(sum(e.usd for e in self.entries if e.kind == "session"), 6)

    def labels(self, kind: EntryKind) -> list[str]:
        return [e.label for e in self.entries if e.kind == kind]

    def duplicate_labels(self) -> list[str]:
        rl = self.labels("rollout")
        sl = self.labels("session")
        return sorted(
            {label for label in rl if rl.count(label) > 1}
            | {label for label in sl if sl.count(label) > 1}
        )

    def assert_identity(self) -> None:
        """Verify ``R = 4k`` and ``P = 3k`` with zero duplicate stage labels.

        Duplicates are checked *before* counts because a repeated stage label can
        mask a missing stage while silently overwriting another candidate's directory.
        """
        want_r, want_p = expected_rollouts(self.k), expected_sessions(self.k)
        dupes = self.duplicate_labels()
        if not dupes and self.R == want_r and self.P == want_p:
            return
        detail = [
            f"{self.mode}-{self.k} {self.task_id}: "
            f"R={self.R} (want {want_r}), P={self.P} (want {want_p})",
            f"  rollouts: {', '.join(self.labels('rollout')) or '(none)'}",
            f"  sessions: {', '.join(self.labels('session')) or '(none)'}",
        ]
        if dupes:
            detail.append(
                f"  DUPLICATE labels: {dupes} -- two stages wrote the same run id "
                "and one silently overwrote the other"
            )
        raise BudgetError("\n".join(detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task_id,
            "k": self.k,
            "rollouts": self.rollouts_executed,
            "stages": self.sessions_executed,
            "rollouts_total": self.rollouts_total,
            "stages_total": self.sessions_total,
            "rollouts_saved": self.rollouts_saved,
            "stages_saved": self.sessions_saved,
            "expected_rollouts": expected_rollouts(self.k),
            "expected_stages": expected_sessions(self.k),
            "usd": self.usd,
            "rollout_usd": self.rollout_usd,
            "session_usd": self.session_usd,
            "max_usd": self.max_usd,
            "adaptive_probe": self.adaptive_probe,
            "entries": [
                {
                    "kind": e.kind,
                    "label": e.label,
                    "usd": e.usd,
                    "cached": e.cached,
                    "reused_fingerprint": e.reused_fingerprint,
                    "attempts": e.attempts,
                    "skipped": e.skipped,
                    "skip_reason": e.skip_reason,
                }
                for e in self.entries
            ],
        }


__all__ = [
    "BudgetError",
    "BudgetExceededError",
    "EntryKind",
    "HarnessLedger",
    "LedgerEntry",
    "expected_rollouts",
    "expected_sessions",
]
