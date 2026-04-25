"""Optional adapter wrappers for TRL and Atropos.

Each adapter probes for its dependency lazily. Importing this module
never imports a training framework — the import only happens inside
:meth:`build_dataset` / :meth:`stream_rollouts` calls. If the user
hasn't installed the dependency we raise
:class:`MissingRLDependencyError` with a clear hint.

To avoid forcing every user to install heavyweight ML stacks, both
adapters are entirely optional. They're useful when you want a
one-liner ``adapter.build_dataset(trajs)`` instead of writing JSONL
plumbing yourself.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from clawagents.rl.errors import MissingRLDependencyError
from clawagents.rl.export import (
    to_atropos_rollout,
    to_trl_dpo,
    to_trl_sft,
)
from clawagents.rl.trajectory import Trajectory


def _have(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


TRL_AVAILABLE = _have("trl")
"""True iff the optional :mod:`trl` package is importable."""

ATROPOS_AVAILABLE = _have("atroposlib") or _have("atropos")
"""True iff an Atropos-compatible package is importable."""


@dataclass
class TrlAdapter:
    """Convert trajectories into ``datasets.Dataset`` rows for TRL.

    Example::

        adapter = TrlAdapter()
        ds = adapter.build_sft_dataset(trajectories)
        # ds is a `datasets.Dataset` ready for SFTTrainer

    The adapter requires the optional ``trl`` and ``datasets`` packages;
    install them with ``pip install trl datasets``.
    """

    require_install_hint: str = "pip install trl datasets"

    def _require(self) -> Any:
        try:
            import datasets
        except Exception as exc:  # pragma: no cover - exercised in tests
            raise MissingRLDependencyError(
                framework="trl", install_hint=self.require_install_hint
            ) from exc
        return datasets

    def build_sft_dataset(self, trajectories: Iterable[Trajectory]) -> Any:
        """Return a HuggingFace ``Dataset`` shaped for ``SFTTrainer``."""
        datasets = self._require()
        rows = [to_trl_sft(t) for t in trajectories]
        return datasets.Dataset.from_list(rows)

    def build_dpo_dataset(
        self, pairs: Sequence[tuple[Trajectory, Trajectory]]
    ) -> Any:
        """Return a HuggingFace ``Dataset`` shaped for ``DPOTrainer``."""
        datasets = self._require()
        rows = [to_trl_dpo(c, r) for (c, r) in pairs]
        return datasets.Dataset.from_list(rows)


@dataclass
class AtroposAdapter:
    """Stream trajectories into an Atropos rollout collector.

    Atropos's interface is intentionally generic — it consumes
    rollouts as dicts with ``messages``, ``score``, and ``metadata``
    fields. This adapter just converts trajectories one-by-one.

    Note: Atropos's exact import path moves between releases; we probe
    a couple of common module names. If neither resolves we raise.
    """

    require_install_hint: str = "pip install atroposlib  # or pip install atropos"

    def _require(self) -> Any:
        last_exc: Exception | None = None
        for mod in ("atroposlib", "atropos"):
            try:
                return importlib.import_module(mod)
            except Exception as exc:
                last_exc = exc
        raise MissingRLDependencyError(
            framework="atropos", install_hint=self.require_install_hint
        ) from last_exc

    def to_rollouts(self, trajectories: Iterable[Trajectory]) -> list[dict[str, Any]]:
        """Convert trajectories to Atropos rollout dicts (no Atropos required)."""
        return [to_atropos_rollout(t) for t in trajectories]

    def submit(
        self,
        trajectories: Iterable[Trajectory],
        *,
        sink: Any | None = None,
    ) -> int:
        """Push rollouts at an Atropos collector.

        ``sink`` may be:

        * ``None`` — uses ``atroposlib.RolloutCollector()`` if available.
        * any object exposing ``.submit(rollout: dict) -> None`` —
          the adapter just forwards to it.

        Returns the number of rollouts submitted.
        """
        rollouts = self.to_rollouts(trajectories)
        if sink is None:
            atropos = self._require()
            collector_cls = getattr(atropos, "RolloutCollector", None)
            if collector_cls is None:
                raise MissingRLDependencyError(
                    framework="atropos",
                    install_hint=self.require_install_hint,
                )
            sink = collector_cls()
        for r in rollouts:
            sink.submit(r)
        return len(rollouts)
