"""Lazy ATLAS import helpers (kept dependency-free until enabled)."""

from __future__ import annotations

_INSTALL_HINT = (
    "ATLAS support requires the optional extra. Install with: "
    "pip install 'clawagents[atlas]'"
)


def require_atlas_runtime() -> None:
    """Import-check ``atlas_runtime``; raise a clear error when missing."""
    try:
        import atlas_runtime  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
