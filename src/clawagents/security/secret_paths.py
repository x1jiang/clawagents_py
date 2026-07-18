"""Single source of truth for secret / credential path matching.

Used by:
  - sandbox seatbelt/bwrap deny globs (``default_secret_globs``)
  - permission engine default rules (``.env`` ask, credentials deny)
  - rewind / hunk watcher (never snapshot secrets)

Keep patterns aligned here — do not redefine secret names in callers.
"""

from __future__ import annotations

from pathlib import Path

# Basename / nested globs for OS sandbox overlays and SBPL regexes.
# ``**/`` forms match nested files; matchers also cover top-level names.
DEFAULT_SECRET_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "**/credentials*",
    "**/secrets*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/id_rsa",
    "**/id_ed25519",
)

# VCS / build noise that must never be snapshotted by the hunk watcher.
IGNORED_WATCH_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".clawagents",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        ".tox",
    }
)

_SECRET_EXACT_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
    }
)

_SECRET_SUFFIXES: tuple[str, ...] = (".pem", ".key", ".p12", ".pfx")


def default_secret_globs() -> tuple[str, ...]:
    """Globs for sandbox secret deny lists."""
    return DEFAULT_SECRET_GLOBS


def is_secret_basename(name: str) -> bool:
    """True if a file basename looks like a credential / secret file.

    Uses ``*credentials*`` / ``*secrets*`` style matches — not substring
    ``secret`` (avoids false positives like ``secretary.txt``).
    """
    import fnmatch

    n = (name or "").strip()
    if not n:
        return False
    if n in _SECRET_EXACT_NAMES or n.startswith(".env."):
        return True
    if n.endswith(_SECRET_SUFFIXES):
        return True
    lower = n.lower()
    if fnmatch.fnmatch(lower, "*credentials*") or fnmatch.fnmatch(
        lower, "*secrets*"
    ):
        return True
    return False


def is_secret_or_ignored_path(rel: str) -> bool:
    """True for VCS/build noise or secret-like paths (hunk watcher / rewind)."""
    parts = Path(rel).parts
    if any(p in IGNORED_WATCH_DIRS for p in parts):
        return True
    return is_secret_basename(Path(rel).name)


def path_matches_secret_globs(
    path: str,
    cwd: str,
    secret_globs: tuple[str, ...] | None = None,
) -> bool:
    """Match ``path`` against secret globs relative to ``cwd`` (sandbox helper).

    Basename component of ``**/*.pem`` is matched via ``os.path.basename(pat)``
    so top-level ``key.pem`` is caught (``pat.lstrip("*/")`` wrongly yielded
    ``.pem`` and matched nothing).
    """
    import fnmatch
    import os

    globs = secret_globs if secret_globs is not None else DEFAULT_SECRET_GLOBS
    if not globs:
        return False
    name = os.path.basename(path)
    try:
        rel = os.path.relpath(path, cwd)
    except ValueError:
        rel = name
    rel_posix = rel.replace(os.sep, "/")
    for pattern in globs:
        pat = (pattern or "").replace("\\", "/")
        if fnmatch.fnmatch(name, os.path.basename(pat)) or fnmatch.fnmatch(
            rel_posix, pat
        ):
            return True
        if pat in {".env", "credentials", "secrets"} and (
            name == pat or name.startswith(pat + ".")
        ):
            return True
    if globs and name == ".env":
        return True
    return is_secret_basename(name)


def default_secure_path_rules() -> list[tuple[str, str, str]]:
    """Return ``(path_pattern, decision, message)`` for the permission engine.

    Decisions:
      - ``ask`` for dotenv files (common legitimate edit, needs approval)
      - ``deny`` for credential / key material
    """
    ask = [
        ("*.env", "ask", "Writing .env requires approval"),
        ("**/.env", "ask", "Writing .env requires approval"),
        ("**/.env.*", "ask", "Writing dotenv files requires approval"),
        (".env.*", "ask", "Writing dotenv files requires approval"),
    ]
    deny = [
        ("**/credentials*", "deny", "Refused write to credentials path"),
        ("**/secrets*", "deny", "Refused write to secrets path"),
        ("**/*.pem", "deny", "Refused write to private key / cert"),
        ("**/*.key", "deny", "Refused write to private key"),
        ("**/*.p12", "deny", "Refused write to PKCS bundle"),
        ("**/*.pfx", "deny", "Refused write to PKCS bundle"),
        ("**/id_rsa", "deny", "Refused write to SSH private key"),
        ("**/id_ed25519", "deny", "Refused write to SSH private key"),
        ("credentials*", "deny", "Refused write to credentials path"),
        ("*.pem", "deny", "Refused write to private key / cert"),
        ("*.key", "deny", "Refused write to private key"),
    ]
    return ask + deny
