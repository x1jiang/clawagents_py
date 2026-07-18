"""Optional RTK wrapping for noisy ``execute`` commands (loop-side, not hooks).

When ``CLAW_FEATURE_RTK_WRAP=1`` and ``rtk`` is on PATH, rewrite common
high-volume shell commands so RTK filters/summarizes before output hits the
agent context. The model does not need to remember to call ``rtk``.
"""

from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache
from typing import Optional

# First token → rtk subcommand (pass remaining argv through).
_DIRECT_MAP: dict[str, str] = {
    "ls": "ls",
    "tree": "tree",
    "rg": "rg",
    "grep": "grep",
    "find": "find",
    "docker": "docker",
    "kubectl": "kubectl",
    "pnpm": "pnpm",
    "tsc": "tsc",
    "jest": "jest",
    "vitest": "vitest",
    "prisma": "prisma",
    "dotnet": "dotnet",
    "aws": "aws",
    "psql": "psql",
    "gh": "gh",
    "glab": "glab",
    "wc": "wc",
    "wget": "wget",
    "diff": "diff",
}

_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "branch", "stash", "fetch",
    "pull", "push", "add", "commit", "worktree",
})

# Run via ``rtk test -- <cmd>`` (failures-focused).
_TEST_HEAD_RE = re.compile(
    r"^(?:"
    r"pytest\b|python\s+-m\s+pytest\b|py\.test\b|"
    r"npm\s+test\b|npm\s+run\s+test\b|"
    r"yarn\s+test\b|pnpm\s+test\b|"
    r"cargo\s+test\b|go\s+test\b|"
    r"make\s+test\b"
    r")",
    re.IGNORECASE,
)

# Noisy builds/lints → ``rtk err -- <cmd>`` (errors/warnings only).
_ERR_HEAD_RE = re.compile(
    r"^(?:"
    r"npm\s+run\s+build\b|npm\s+run\s+lint\b|"
    r"yarn\s+build\b|yarn\s+lint\b|"
    r"pnpm\s+run\s+build\b|pnpm\s+run\s+lint\b|"
    r"cargo\s+build\b|cargo\s+check\b|cargo\s+clippy\b|"
    r"make\s+build\b|cmake\s+--build\b|"
    r"mvn\s+|gradle\w*\s+|"
    r"eslint\b|ruff\s+check\b|mypy\b|pyright\b|flake8\b"
    r")",
    re.IGNORECASE,
)

_SKIP_HEAD_RE = re.compile(
    r"^(?:"
    r"rtk\b|"
    r"cd\b|export\b|unset\b|source\b|\.|\[|test\s|"
    r"echo\b|printf\b|true\b|false\b|:"
    r")",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def rtk_binary() -> Optional[str]:
    override = (os.environ.get("CLAW_RTK_BIN") or "").strip()
    if override:
        return override if os.path.isfile(override) and os.access(override, os.X_OK) else None
    return shutil.which("rtk")


_rtk_floor_warned = False


def reset_rtk_cache() -> None:
    """Test helper — clear cached ``which(rtk)`` result."""
    global _rtk_floor_warned
    rtk_binary.cache_clear()
    _rtk_floor_warned = False


def _split_head(command: str) -> tuple[str, list[str]]:
    """Return (first_token, rest_tokens) with a light shell-aware split."""
    # Strip leading env assignments: FOO=1 BAR=2 cmd ...
    s = command.strip()
    while True:
        m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|\S+)\s+", s)
        if not m:
            break
        s = s[m.end() :]
    # Don't wrap pipelines / redirects / compound commands — too easy to break.
    if any(op in s for op in ("|", "||", "&&", ";", ">", "<", "`", "$(", "\n")):
        return "", []
    parts = s.split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _warn_rtk_below_floor(binary: str) -> None:
    """Emit a one-shot stderr note when rtk is present but below the floor."""
    global _rtk_floor_warned
    if _rtk_floor_warned:
        return
    _rtk_floor_warned = True
    try:
        from clawagents.companions import probe_rtk

        status = probe_rtk()
        if status.found and not status.ok_vs_floor:
            import sys

            sys.stderr.write(
                f"[clawagents] rtk below floor: {status.summary()} — {status.hint}\n"
            )
    except Exception:  # noqa: BLE001
        pass
    _ = binary


def maybe_wrap_with_rtk(command: str) -> tuple[str, Optional[str]]:
    """Return ``(command, reason|None)``. Unchanged when wrap does not apply."""
    from clawagents.config.features import is_enabled

    if not is_enabled("rtk_wrap"):
        return command, None
    if not command or not command.strip():
        return command, None

    binary = rtk_binary()
    if not binary:
        return command, None

    _warn_rtk_below_floor(binary)

    stripped = command.strip()
    if _SKIP_HEAD_RE.match(stripped):
        return command, None

    # Already rtk-prefixed (including via env-assignment strip miss).
    if stripped.startswith("rtk ") or stripped.startswith(f"{binary} "):
        return command, None

    if _TEST_HEAD_RE.match(stripped):
        return f"{binary} test {stripped}", "rtk test"

    if _ERR_HEAD_RE.match(stripped):
        return f"{binary} err {stripped}", "rtk err"

    head, rest = _split_head(stripped)
    if not head:
        return command, None

    # python -m pytest already caught; bare pytest too.
    if head == "git" and rest and rest[0] in _GIT_SUBCOMMANDS:
        return f"{binary} git {' '.join(rest)}", f"rtk git {rest[0]}"

    mapped = _DIRECT_MAP.get(head)
    if mapped:
        # Preserve flags/args after the head token (from original stripped, not
        # env-stripped head alone — use rest from _split_head).
        tail = " ".join(rest)
        wrapped = f"{binary} {mapped}" + (f" {tail}" if tail else "")
        return wrapped, f"rtk {mapped}"

    return command, None


__all__ = ["maybe_wrap_with_rtk", "rtk_binary", "reset_rtk_cache"]
