"""HTTP + WebSocket gateway (optional: ``pip install 'clawagents[gateway]'``).

The gateway needs FastAPI and uvicorn, which are not installed with the core
package. Names are resolved lazily so importing ``clawagents.gateway`` never
fails; using it without the extra raises an error that says what to install.
"""

from __future__ import annotations

from typing import Any

GATEWAY_INSTALL_HINT = (
    "The ClawAgents gateway needs FastAPI and uvicorn, which are not installed "
    "with the core package. Install them with:\n"
    "  pip install 'clawagents[gateway]'"
)

_EXPORTS = ("create_app", "start_gateway", "resolve_bind_host", "print_bind_warning")


def require_gateway_deps() -> None:
    """Raise a clear ImportError when the gateway extra is missing."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise ImportError(GATEWAY_INSTALL_HINT) from exc


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        require_gateway_deps()
        from clawagents.gateway import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GATEWAY_INSTALL_HINT", "require_gateway_deps", *_EXPORTS]
