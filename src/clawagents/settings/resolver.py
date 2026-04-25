"""Settings hierarchy resolver — user / project / local / flag / policy.

Inspired by ``claude-code-main/src/utils/settings/``.

Five layers, from lowest to highest precedence (Policy ALWAYS wins):

    1. user     — ``~/.clawagents/settings.json``
    2. project  — ``<repo>/.clawagents/settings.json`` (committed)
    3. local    — ``<repo>/.clawagents/settings.local.json`` (gitignored)
    4. flag     — runtime flags passed to :func:`resolve_settings`
    5. policy   — ``/etc/clawagents/policy-settings.json`` (or env var
                  ``CLAWAGENTS_POLICY_SETTINGS_PATH``); applied last so even
                  runtime flags can't override it.

Each higher layer is deep-merged on top of the accumulated lower-precedence
result. File reads are graceful: a missing file is treated as an empty
layer; malformed JSON is logged at WARNING and skipped.

Public API:

    - :class:`SettingsLayer`         — enum of layer names
    - :func:`resolve_settings`       — collapse all layers into a single dict
    - :func:`get_setting`            — dotted-path lookup convenience wrapper
    - :func:`find_repo_root`         — exposed for tests / introspection

The module is pure stdlib JSON — no extra runtime dependencies.
"""

from __future__ import annotations

import enum
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "SettingsLayer",
    "resolve_settings",
    "get_setting",
    "find_repo_root",
    "POLICY_SETTINGS_PATH_ENV",
    "DEFAULT_POLICY_SETTINGS_PATH",
]

_log = logging.getLogger(__name__)

POLICY_SETTINGS_PATH_ENV = "CLAWAGENTS_POLICY_SETTINGS_PATH"
DEFAULT_POLICY_SETTINGS_PATH = "/etc/clawagents/policy-settings.json"


class SettingsLayer(str, enum.Enum):
    """Named layers, in order of increasing precedence (policy ALWAYS wins).

    Inheriting from ``str`` lets the enum serialise transparently.
    """

    USER = "user"
    PROJECT = "project"
    LOCAL = "local"
    FLAG = "flag"
    POLICY = "policy"


# ─── Internal helpers ──────────────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file. Missing → {}, malformed → warn & {}."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        _log.warning("clawagents.settings: cannot read %s: %s", path, exc)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log.warning("clawagents.settings: malformed JSON in %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        _log.warning(
            "clawagents.settings: %s does not contain a JSON object (got %s); skipping",
            path,
            type(data).__name__,
        )
        return {}
    return data


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` onto ``base``, returning a new dict.

    Dict values are merged recursively. All other types (lists, scalars)
    are replaced wholesale by the overlay's value.
    """
    out: dict[str, Any] = dict(base)
    for k, v in overlay.items():
        prev = out.get(k)
        if isinstance(prev, dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(prev, v)
        else:
            out[k] = v
    return out


def find_repo_root(start: Path | str | None = None) -> Path:
    """Walk up from ``start`` (default CWD) until a repo marker is found.

    Markers (any one is enough): ``.git``, ``pyproject.toml``,
    ``package.json``. If nothing matches, returns the original ``start``.
    """
    cur = Path(start) if start is not None else Path.cwd()
    cur = cur.resolve()
    markers = (".git", "pyproject.toml", "package.json")
    for candidate in (cur, *cur.parents):
        for m in markers:
            if (candidate / m).exists():
                return candidate
    return cur


def _user_settings_path() -> Path:
    return Path.home() / ".clawagents" / "settings.json"


def _project_settings_path(repo_root: Path) -> Path:
    return repo_root / ".clawagents" / "settings.json"


def _local_settings_path(repo_root: Path) -> Path:
    return repo_root / ".clawagents" / "settings.local.json"


def _policy_settings_path() -> Path:
    override = os.environ.get(POLICY_SETTINGS_PATH_ENV)
    if override:
        return Path(override)
    return Path(DEFAULT_POLICY_SETTINGS_PATH)


# ─── Public API ────────────────────────────────────────────────────────


def resolve_settings(
    *,
    flag_overrides: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Collapse all layers into a single settings dict.

    Args:
        flag_overrides: runtime-injected flags (the ``flag`` layer). Treated
            as already-loaded JSON; deep-merged like any other layer.
        repo_root: override the auto-detected repo root. Defaults to walking
            up from CWD.

    Precedence (highest wins, except policy which ALWAYS wins):

        user < project < local < flag < policy
    """
    root = find_repo_root(repo_root)

    user = _load_json(_user_settings_path())
    project = _load_json(_project_settings_path(root))
    local = _load_json(_local_settings_path(root))
    flag = dict(flag_overrides) if flag_overrides else {}
    policy = _load_json(_policy_settings_path())

    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, user)
    merged = _deep_merge(merged, project)
    merged = _deep_merge(merged, local)
    merged = _deep_merge(merged, flag)
    # Policy is applied LAST: it wins over everything, including flags.
    merged = _deep_merge(merged, policy)
    return merged


def get_setting(
    path: str,
    default: Any = None,
    *,
    flag_overrides: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
    settings: Mapping[str, Any] | None = None,
) -> Any:
    """Read a dotted path out of resolved settings.

    Example::

        get_setting("hooks.before_tool")   # → list[str] | None

    Args:
        path: dotted path, e.g. ``"hooks.before_tool"``.
        default: returned if any segment is missing or not a dict.
        flag_overrides: forwarded to :func:`resolve_settings` if ``settings``
            is not provided.
        repo_root: forwarded to :func:`resolve_settings` if ``settings``
            is not provided.
        settings: pre-resolved settings dict; skips re-resolution.
    """
    src: Mapping[str, Any]
    if settings is not None:
        src = settings
    else:
        src = resolve_settings(flag_overrides=flag_overrides, repo_root=repo_root)
    cur: Any = src
    for segment in path.split("."):
        if not isinstance(cur, Mapping) or segment not in cur:
            return default
        cur = cur[segment]
    return cur
