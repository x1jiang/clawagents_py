"""Lazy ATLAS import helpers (kept dependency-free until enabled)."""

from __future__ import annotations

_INSTALL_HINT = (
    "ATLAS support requires the atlas-skill package. Install with: "
    "pip install 'atlas-skill @ "
    "git+https://github.com/multi-agent-systems-failure-taxonomy/"
    "ATLAS.git@3a917f3e0b993e3bfd77f652b013193aed167964'"
)


def require_atlas_runtime() -> None:
    """Import-check ``atlas_runtime``; raise a clear error when missing."""
    try:
        import atlas_runtime  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
