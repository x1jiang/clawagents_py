"""Profile-aware filesystem paths for ClawAgents.

ClawAgents uses two complementary on-disk locations:

- **Workspace** (per-project): ``<cwd>/.clawagents/...`` — trajectories,
  sessions, and lessons that are scoped to a specific repository or
  working tree. This is the legacy behavior; modules like
  :mod:`clawagents.trajectory.recorder` and
  :mod:`clawagents.session.persistence` still default to it.
- **Home** (per-user, profile-aware): ``~/.clawagents/<profile>/...`` —
  user-level state shared across projects (global lessons, identity
  caches, persistent memory, credential pools).

Pick the right one for your data:

==========================================  =========================
Data                                        Recommended scope
==========================================  =========================
Trajectory of a specific run                workspace
Lessons distilled from one repo             workspace
Cross-project lesson library                home
Per-user agent identity / preferences       home
Per-user MCP server credentials             home
==========================================  =========================

Environment overrides
---------------------
- ``CLAWAGENTS_HOME`` — absolute path overriding ``~/.clawagents``.
  Useful for sandboxed CI runs or testing.
- ``CLAWAGENTS_PROFILE`` — name of the active profile (default
  ``"default"``). Profiles let one user keep separate state for, say,
  personal vs. work agents.
- ``CLAWAGENTS_WORKSPACE`` — absolute path overriding ``<cwd>``.
  Mainly useful for tests.

This module is intentionally tiny and dependency-free; it just resolves
paths and creates dirs when asked. Mirrors
``clawagents/src/paths.ts`` (TypeScript).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PROFILE = "default"
WORKSPACE_DIRNAME = ".clawagents"
HOME_DIRNAME = ".clawagents"


def _profile_name(profile: str | None) -> str:
    if profile:
        return profile
    return os.environ.get("CLAWAGENTS_PROFILE", DEFAULT_PROFILE)


def get_clawagents_home(
    profile: str | None = None,
    *,
    create: bool = False,
) -> Path:
    """Return the active per-user, profile-scoped home directory.

    By default this is ``~/.clawagents/<profile>/``. Set
    ``CLAWAGENTS_HOME`` to override the parent (``~/.clawagents``); the
    profile suffix is always applied unless ``CLAWAGENTS_HOME`` already
    ends with the profile name.

    Args:
        profile: Profile name. Defaults to the ``CLAWAGENTS_PROFILE``
            env var, then ``"default"``.
        create: If True, create the directory (and parents) if missing.

    Returns:
        Absolute :class:`Path` to the home directory.
    """
    name = _profile_name(profile)
    override = os.environ.get("CLAWAGENTS_HOME")
    if override:
        base = Path(override).expanduser()
        # Treat the override as the *parent* unless it already ends with the
        # profile dir; this lets users set CLAWAGENTS_HOME to a sandbox
        # without losing profile separation.
        if base.name == name:
            home = base
        else:
            home = base / name
    else:
        home = Path.home() / HOME_DIRNAME / name
    if create:
        home.mkdir(parents=True, exist_ok=True)
    return home


def get_clawagents_workspace_dir(
    *,
    create: bool = False,
) -> Path:
    """Return the per-project workspace directory.

    By default this is ``<cwd>/.clawagents/``. Override the working
    directory via the ``CLAWAGENTS_WORKSPACE`` env var.

    Args:
        create: If True, create the directory (and parents) if missing.
    """
    override = os.environ.get("CLAWAGENTS_WORKSPACE")
    base = Path(override).expanduser() if override else Path.cwd()
    ws = base / WORKSPACE_DIRNAME
    if create:
        ws.mkdir(parents=True, exist_ok=True)
    return ws


def get_trajectories_dir(
    *,
    scope: str = "workspace",
    profile: str | None = None,
    create: bool = False,
) -> Path:
    """Return the trajectories directory under the chosen scope.

    Args:
        scope: ``"workspace"`` (default; per-project) or ``"home"``
            (per-user, profile-scoped).
        profile: Forwarded to :func:`get_clawagents_home` when
            ``scope == "home"``.
        create: If True, create the directory (and parents) if missing.
    """
    if scope == "home":
        out = get_clawagents_home(profile, create=False) / "trajectories"
    elif scope == "workspace":
        out = get_clawagents_workspace_dir(create=False) / "trajectories"
    else:
        raise ValueError(f"unknown scope {scope!r} (expected 'workspace' or 'home')")
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def get_sessions_dir(
    *,
    scope: str = "workspace",
    profile: str | None = None,
    create: bool = False,
) -> Path:
    """Return the sessions directory under the chosen scope.

    See :func:`get_trajectories_dir` for argument semantics.
    """
    if scope == "home":
        out = get_clawagents_home(profile, create=False) / "sessions"
    elif scope == "workspace":
        out = get_clawagents_workspace_dir(create=False) / "sessions"
    else:
        raise ValueError(f"unknown scope {scope!r} (expected 'workspace' or 'home')")
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def get_lessons_dir(
    *,
    scope: str = "home",
    profile: str | None = None,
    create: bool = False,
) -> Path:
    """Return the lessons directory under the chosen scope.

    Lessons default to ``home`` scope so they survive across projects.
    """
    if scope == "home":
        out = get_clawagents_home(profile, create=False) / "lessons"
    elif scope == "workspace":
        out = get_clawagents_workspace_dir(create=False) / "lessons"
    else:
        raise ValueError(f"unknown scope {scope!r} (expected 'workspace' or 'home')")
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def display_clawagents_home(profile: str | None = None) -> str:
    """Return a user-facing string for the active profile home directory.

    Resolves the same path as :func:`get_clawagents_home`, but renders
    paths under ``$HOME`` as ``~/...`` so tool descriptions, approval
    prompts, and trajectory messages show the user a familiar shorthand
    rather than a fully-resolved absolute path that may include the
    sandbox or profile chosen via ``CLAWAGENTS_HOME``/``CLAWAGENTS_PROFILE``.

    Use this whenever a path is shown to humans (tool schemas, log
    lines, README snippets); use :func:`get_clawagents_home` when the
    code itself needs to read or write a file.

    Mirrors ``displayClawagentsHome()`` in the TypeScript package and
    Hermes' ``display_hermes_home()``.
    """
    home = get_clawagents_home(profile, create=False)
    try:
        rel = home.relative_to(Path.home())
    except ValueError:
        return str(home)
    return f"~/{rel}" if str(rel) != "." else "~"


def display_clawagents_workspace_dir() -> str:
    """Return a user-facing string for the active workspace directory.

    Like :func:`display_clawagents_home` but for the per-project
    workspace dir. Renders paths under ``$HOME`` as ``~/...``.
    """
    ws = get_clawagents_workspace_dir(create=False)
    try:
        rel = ws.relative_to(Path.home())
    except ValueError:
        return str(ws)
    return f"~/{rel}" if str(rel) != "." else "~"


def list_profiles() -> list[str]:
    """Enumerate profile directories under ``~/.clawagents/`` (or the
    ``CLAWAGENTS_HOME`` override).

    Returns the profile names sorted alphabetically. Returns ``[]`` if
    the parent directory does not exist.
    """
    override = os.environ.get("CLAWAGENTS_HOME")
    parent = Path(override).expanduser() if override else Path.home() / HOME_DIRNAME
    if not parent.is_dir():
        return []
    out = []
    for child in parent.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            out.append(child.name)
    return sorted(out)


__all__ = [
    "DEFAULT_PROFILE",
    "WORKSPACE_DIRNAME",
    "HOME_DIRNAME",
    "get_clawagents_home",
    "get_clawagents_workspace_dir",
    "get_trajectories_dir",
    "get_sessions_dir",
    "get_lessons_dir",
    "display_clawagents_home",
    "display_clawagents_workspace_dir",
    "list_profiles",
]
