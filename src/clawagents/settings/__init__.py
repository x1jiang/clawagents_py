"""Settings hierarchy — user / project / local / flag / policy resolver.

See :mod:`clawagents.settings.resolver` for the full module doc.
"""

from clawagents.settings.resolver import (
    DEFAULT_POLICY_SETTINGS_PATH,
    POLICY_SETTINGS_PATH_ENV,
    SettingsLayer,
    find_repo_root,
    get_setting,
    resolve_settings,
)

__all__ = [
    "SettingsLayer",
    "resolve_settings",
    "get_setting",
    "find_repo_root",
    "POLICY_SETTINGS_PATH_ENV",
    "DEFAULT_POLICY_SETTINGS_PATH",
]
