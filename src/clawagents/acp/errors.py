"""ACP adapter errors."""

from __future__ import annotations


class AcpError(Exception):
    """Base class for ACP adapter errors."""


class MissingAcpDependencyError(AcpError):
    """Raised when the optional ``agent-client-protocol`` package is missing.

    Install with::

        pip install "clawagents[acp]"
    """

    def __init__(self, original: BaseException | None = None) -> None:
        msg = (
            "The ACP adapter requires the optional 'agent-client-protocol' "
            "package. Install it with: pip install \"clawagents[acp]\""
        )
        if original is not None:
            msg += f" (original error: {original})"
        super().__init__(msg)
        self.original = original
