"""Errors for the ClawAgents RL adapter."""

from __future__ import annotations


class RLError(RuntimeError):
    """Base class for errors raised by :mod:`clawagents.rl`."""


class MissingRLDependencyError(RLError):
    """Raised when an optional RL training dependency isn't installed.

    The RL adapters (TRL, Atropos, …) are intentionally lazy. We only
    import them when the user actually wants to push trajectories into
    a training run; if the package isn't installed we raise this error
    with a clear message explaining how to install it.
    """

    def __init__(self, framework: str, install_hint: str) -> None:
        super().__init__(
            f"clawagents.rl: optional dependency for '{framework}' is not "
            f"installed. Install it with: {install_hint}"
        )
        self.framework = framework
        self.install_hint = install_hint
