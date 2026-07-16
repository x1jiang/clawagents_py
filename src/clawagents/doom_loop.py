"""Doom-loop detection — generation tail-repetition with resample policy.

Grok Build parity (sampling-types doom_loop.rs): confident trigger is
``tail_repetition`` on the thinking channel at/under threshold; host may
resample up to max_retries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional


Channel = Literal["thinking", "response"]


@dataclass
class DoomLoopSignal:
    kind: str  # tail_repetition | low_logprob
    threshold: int
    channel: Channel
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.kind}:{self.threshold}@{self.channel}"


@dataclass
class DoomLoopRecoveryPolicy:
    max_threshold: int = 8  # clamp [2, 64]
    max_retries: int = 2  # clamp [0, 5]

    def __post_init__(self) -> None:
        self.max_threshold = max(2, min(64, int(self.max_threshold)))
        self.max_retries = max(0, min(5, int(self.max_retries)))


@dataclass
class DoomLoopState:
    retry_count: int = 0
    triggers: list[DoomLoopSignal] = field(default_factory=list)


def is_confident_trigger(
    signal: DoomLoopSignal, policy: DoomLoopRecoveryPolicy | None = None
) -> bool:
    """Thinking or response-channel tail_repetition at/under max_threshold."""
    pol = policy or DoomLoopRecoveryPolicy()
    return (
        signal.channel in ("thinking", "response")
        and signal.kind == "tail_repetition"
        and signal.threshold <= pol.max_threshold
    )


def detect_tail_repetition(
    text: str,
    *,
    channel: Channel = "thinking",
    min_unit: int = 8,
    max_unit: int = 80,
) -> DoomLoopSignal | None:
    """Detect repeated trailing substrings (generation doom loop).

    Looks for the same chunk repeating at the end of ``text``.
    Threshold ≈ number of consecutive repeats of the unit.
    """
    body = (text or "").strip()
    if len(body) < min_unit * 2:
        return None
    # Prefer line-based repetition (common thinking-loop pattern)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) >= 4:
        last = lines[-1].strip()
        if len(last) >= 4:
            count = 0
            for ln in reversed(lines):
                if ln.strip() == last:
                    count += 1
                else:
                    break
            if count >= 3:
                return DoomLoopSignal(
                    kind="tail_repetition",
                    threshold=count,
                    channel=channel,
                )

    # Character n-gram tail repetition
    best: DoomLoopSignal | None = None
    for unit_len in range(min_unit, min(max_unit, len(body) // 2) + 1):
        unit = body[-unit_len:]
        count = 1
        pos = len(body) - unit_len
        while pos - unit_len >= 0 and body[pos - unit_len : pos] == unit:
            count += 1
            pos -= unit_len
        if count >= 3:
            sig = DoomLoopSignal(
                kind="tail_repetition", threshold=count, channel=channel
            )
            if best is None or sig.threshold > best.threshold:
                best = sig
    return best


def should_resample(
    signal: DoomLoopSignal,
    state: DoomLoopState,
    policy: DoomLoopRecoveryPolicy | None = None,
) -> bool:
    pol = policy or DoomLoopRecoveryPolicy()
    if not is_confident_trigger(signal, pol):
        return False
    return state.retry_count < pol.max_retries


def note_trigger(state: DoomLoopState, signal: DoomLoopSignal) -> None:
    state.triggers.append(signal)


def parse_label(label: str) -> DoomLoopSignal | None:
    m = re.match(
        r"^(tail_repetition|low_logprob):(\d+)@(thinking|response)$",
        (label or "").strip(),
    )
    if not m:
        return None
    return DoomLoopSignal(
        kind=m.group(1),
        threshold=int(m.group(2)),
        channel=m.group(3),  # type: ignore[arg-type]
        label=label.strip(),
    )


__all__ = [
    "DoomLoopSignal",
    "DoomLoopRecoveryPolicy",
    "DoomLoopState",
    "is_confident_trigger",
    "detect_tail_repetition",
    "should_resample",
    "note_trigger",
    "parse_label",
]
