"""Auxiliary model registry — pick the right model for each task.

Real agent runs do more than just "the main reasoning LLM". They also:

- **Compress** long conversations into a running summary (cheap +
  fast model is a better fit than the flagship reasoner).
- **Title** threads / runs (a tiny model is plenty).
- **Process images** (only the multimodal models can do this).
- **Judge / evaluate** trajectories or outputs.

Hermes-style frameworks let operators pin a *different* model to each
of these auxiliary tasks. This module gives ClawAgents the same
capability via a small, dependency-free registry.

Example
-------
::

    from clawagents.aux_models import (
        AuxModelRegistry, AuxModelTask, AuxModelSpec,
    )

    aux = AuxModelRegistry.from_env("gpt-5.4")  # primary
    aux.set(AuxModelTask.COMPRESSION, "gpt-5.4-mini")
    aux.set(AuxModelTask.TITLE, AuxModelSpec(model="gpt-5.4-mini",
                                             max_tokens=20))

    spec = aux.get(AuxModelTask.COMPRESSION)
    # → AuxModelSpec(model="gpt-5.4-mini", ...)

The registry is a *lookup table* — it never calls the LLM itself. Any
component that wants to pick a task-specific model imports this module,
asks for the spec, and feeds the result into its provider call.

Environment overrides
---------------------
- ``CLAW_MODEL_COMPRESSION`` — compression task default
- ``CLAW_MODEL_TITLE``       — title task default
- ``CLAW_MODEL_VISION``      — vision task default
- ``CLAW_MODEL_JUDGE``       — judge task default

Each can be either a model id (``"gpt-5.4-mini"``) or a ``model@base_url``
shorthand (``"llama3.2:3b@http://localhost:11434"``). Anything else
should be configured programmatically via :class:`AuxModelSpec`.

Mirrors ``clawagents/src/aux-models.ts``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum


class AuxModelTask(str, Enum):
    """A named slot for an auxiliary model role.

    ``PRIMARY`` is the canonical "main reasoning" model. The other
    members are auxiliary slots that consumers may override
    independently. Subclassing isn't supported; if you need a custom
    task, register a string key directly with
    :meth:`AuxModelRegistry.set`.
    """

    PRIMARY = "primary"
    COMPRESSION = "compression"
    TITLE = "title"
    VISION = "vision"
    JUDGE = "judge"


_ENV_VAR = {
    AuxModelTask.COMPRESSION: "CLAW_MODEL_COMPRESSION",
    AuxModelTask.TITLE: "CLAW_MODEL_TITLE",
    AuxModelTask.VISION: "CLAW_MODEL_VISION",
    AuxModelTask.JUDGE: "CLAW_MODEL_JUDGE",
}


@dataclass(frozen=True)
class AuxModelSpec:
    """Provider-agnostic description of *which* model to use for a task.

    Attributes:
        model: Required. The model identifier (``"gpt-5.4-mini"``,
            ``"claude-4.5-sonnet"``, ``"llama3.2:3b"``).
        base_url: Optional non-default endpoint (e.g. an Ollama host or
            a self-hosted OpenAI-compatible gateway).
        api_key: Optional API key override. Use sparingly — most
            consumers should rely on ambient env credentials so this
            object doesn't carry secrets.
        temperature: Optional sampling temperature.
        max_tokens: Optional response length cap. Title tasks can use
            very small caps (``20``) for snappier latency.
        extra: Free-form key/value bag for provider-specific knobs.
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def with_overrides(self, **changes: object) -> "AuxModelSpec":
        """Return a new spec with the given fields overridden."""
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def coerce(cls, value: "str | AuxModelSpec") -> "AuxModelSpec":
        """Promote a bare ``str`` model id (or ``model@base_url``) to a spec."""
        if isinstance(value, AuxModelSpec):
            return value
        if not isinstance(value, str):
            raise TypeError(f"AuxModelSpec.coerce: expected str | AuxModelSpec, got {type(value).__name__}")
        s = value.strip()
        if not s:
            raise ValueError("AuxModelSpec.coerce: empty model string")
        if "@" in s:
            model, _, base = s.partition("@")
            return cls(model=model.strip(), base_url=base.strip() or None)
        return cls(model=s)


class AuxModelRegistry:
    """Per-run lookup table from :class:`AuxModelTask` to :class:`AuxModelSpec`.

    Construct one for each agent run. Fallback rule: if a task has no
    explicit binding, :meth:`get` returns the ``PRIMARY`` spec. This
    means callers can ask for ``COMPRESSION`` even when the operator
    hasn't configured one — they still get *a* model.

    The registry is intentionally permissive — task ids are stored as
    plain strings under the hood, so consumers may register custom
    tasks beyond the :class:`AuxModelTask` enum.
    """

    def __init__(self, primary: "str | AuxModelSpec") -> None:
        self._slots: dict[str, AuxModelSpec] = {}
        self.set(AuxModelTask.PRIMARY, primary)

    def set(self, task: "AuxModelTask | str", spec: "str | AuxModelSpec") -> None:
        """Bind a model to a task slot. Overwrites any existing binding."""
        key = task.value if isinstance(task, AuxModelTask) else str(task)
        self._slots[key] = AuxModelSpec.coerce(spec)

    def unset(self, task: "AuxModelTask | str") -> None:
        """Remove a binding (so :meth:`get` falls back to PRIMARY)."""
        key = task.value if isinstance(task, AuxModelTask) else str(task)
        if key == AuxModelTask.PRIMARY.value:
            raise ValueError("PRIMARY is required and cannot be unset")
        self._slots.pop(key, None)

    def get(self, task: "AuxModelTask | str") -> AuxModelSpec:
        """Return the spec for a task, falling back to PRIMARY."""
        key = task.value if isinstance(task, AuxModelTask) else str(task)
        spec = self._slots.get(key)
        if spec is not None:
            return spec
        return self._slots[AuxModelTask.PRIMARY.value]

    def has(self, task: "AuxModelTask | str") -> bool:
        """True if an explicit binding exists for ``task`` (not via fallback)."""
        key = task.value if isinstance(task, AuxModelTask) else str(task)
        return key in self._slots

    def primary(self) -> AuxModelSpec:
        """Shorthand for ``self.get(AuxModelTask.PRIMARY)``."""
        return self._slots[AuxModelTask.PRIMARY.value]

    def slots(self) -> dict[str, AuxModelSpec]:
        """Return a shallow copy of the binding map."""
        return dict(self._slots)

    @classmethod
    def from_env(
        cls,
        primary: "str | AuxModelSpec",
        *,
        env: dict[str, str] | None = None,
    ) -> "AuxModelRegistry":
        """Build a registry, populating known tasks from env vars.

        Args:
            primary: The primary model (always required — env vars are
                only consulted for the auxiliary slots).
            env: Optional environment dict (defaults to ``os.environ``).
                Mainly useful for tests.
        """
        env_map = env if env is not None else dict(os.environ)
        reg = cls(primary)
        for task, var in _ENV_VAR.items():
            raw = env_map.get(var)
            if raw and raw.strip():
                reg.set(task, raw)
        return reg


__all__ = [
    "AuxModelTask",
    "AuxModelSpec",
    "AuxModelRegistry",
]
