"""Pluggable reward scorers for trajectories.

A scorer is any callable that takes a :class:`Trajectory` and returns a
``float`` in roughly ``[-1.0, 1.0]``. The included scorers cover the
common heuristics — string matching, regex, length penalties — and
:class:`CompositeScorer` lets you blend several into a single reward.

Custom scorers simply implement ``__call__(traj) -> float`` (or are
plain functions of the same shape). They never need to inherit from a
base class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from clawagents.rl.trajectory import Trajectory


class RewardScorer(Protocol):
    """Anything callable as ``scorer(traj) -> float``."""

    def __call__(self, traj: Trajectory) -> float:  # pragma: no cover - protocol
        ...


@dataclass
class ContainsScorer:
    """+1 if every required substring appears in the assistant output, else -1.

    Args:
        needles: substrings that must appear (any order). Empty list ⇒ always 0.
        case_sensitive: if False, comparison is lowercased.
        partial_credit: if True, scale by fraction of matched needles.
    """

    needles: list[str]
    case_sensitive: bool = False
    partial_credit: bool = False

    def __call__(self, traj: Trajectory) -> float:
        if not self.needles:
            return 0.0
        text = traj.assistant_text
        if not self.case_sensitive:
            text = text.lower()
        matches = 0
        for n in self.needles:
            probe = n if self.case_sensitive else n.lower()
            if probe in text:
                matches += 1
        if self.partial_credit:
            return (matches / len(self.needles)) * 2.0 - 1.0
        return 1.0 if matches == len(self.needles) else -1.0


@dataclass
class ExactMatchScorer:
    """+1 if assistant's final content matches ``expected`` exactly, else -1."""

    expected: str
    strip: bool = True
    case_sensitive: bool = True

    def __call__(self, traj: Trajectory) -> float:
        final = traj.final_assistant
        if final is None:
            return -1.0
        actual = final.content
        expected = self.expected
        if self.strip:
            actual = actual.strip()
            expected = expected.strip()
        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()
        return 1.0 if actual == expected else -1.0


@dataclass
class RegexScorer:
    """+1 if regex matches anywhere in the assistant output, else -1.

    Compiles ``pattern`` lazily (so the dataclass remains pickleable).
    """

    pattern: str
    flags: int = 0

    def __call__(self, traj: Trajectory) -> float:
        try:
            rx = re.compile(self.pattern, self.flags)
        except re.error:
            return 0.0
        return 1.0 if rx.search(traj.assistant_text) else -1.0


@dataclass
class LengthPenaltyScorer:
    """Penalise responses outside a target window.

    ``target_chars`` is the ideal length; the scorer returns ``1.0``
    when length matches, decaying linearly to ``-1.0`` at the bounds.
    """

    target_chars: int = 400
    min_chars: int = 0
    max_chars: int = 4000

    def __call__(self, traj: Trajectory) -> float:
        n = len(traj.assistant_text)
        if n <= 0:
            return -1.0
        if n < self.min_chars or n > self.max_chars:
            return -1.0
        if n == self.target_chars:
            return 1.0
        if n < self.target_chars:
            span = self.target_chars - self.min_chars
            if span <= 0:
                return 0.0
            return 1.0 - 2.0 * ((self.target_chars - n) / span)
        # n > target
        span = self.max_chars - self.target_chars
        if span <= 0:
            return 0.0
        return 1.0 - 2.0 * ((n - self.target_chars) / span)


@dataclass
class CompositeScorer:
    """Weighted blend of multiple scorers.

    ``weights`` may be a list (parallel to ``scorers``) or omitted, in
    which case all scorers are weighted equally. The result is
    *normalised by the sum of weights* so it remains in roughly
    ``[-1, 1]`` no matter how many components you stack.
    """

    scorers: list[RewardScorer]
    weights: list[float] | None = None
    name: str = "composite"

    def __call__(self, traj: Trajectory) -> float:
        if not self.scorers:
            return 0.0
        weights = self.weights or [1.0] * len(self.scorers)
        if len(weights) != len(self.scorers):
            raise ValueError("CompositeScorer: weights length must match scorers")
        total_w = sum(abs(w) for w in weights) or 1.0
        score = 0.0
        for scorer, w in zip(self.scorers, weights):
            score += w * float(scorer(traj))
        return score / total_w


def score_all(
    traj: Trajectory,
    scorers: dict[str, RewardScorer | Callable[[Trajectory], float]],
) -> dict[str, float]:
    """Run a name → scorer mapping and stash results on ``traj.rewards``.

    The returned dict is also written to ``traj.rewards`` (overwriting
    any pre-existing keys); ``traj.reward`` is set to the *mean* across
    components if it isn't already set.
    """
    results = {name: float(s(traj)) for name, s in scorers.items()}
    traj.rewards.update(results)
    if traj.reward is None and results:
        traj.reward = sum(results.values()) / len(results)
    return results
