"""Grok-inspired shell session state for ``execute`` (cwd continuity).

Grok Build persists cwd/env via a dump/replay script after each command.
We take the high-ROI slice: track cwd across ``execute`` calls by wrapping
commands in ``cd <session_cwd>`` and scraping a trailing pwd marker.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PWD_MARKER = "__CLAW_PWD__"


@dataclass
class ShellSession:
    """Per-agent shell session (state emulation, not a persistent process)."""

    cwd: str = field(default_factory=lambda: str(Path.cwd().resolve()))

    def wrap(self, command: str) -> str:
        """Prefix command so it runs in ``self.cwd`` and emits a pwd trailer."""
        q = shlex.quote(self.cwd)
        # Trailer always runs so we learn about ``cd`` side effects.
        return (
            f"cd {q} || exit 121; "
            f"{command}; "
            f"__claw_ec=$?; "
            f"printf '%s%s\\n' '{PWD_MARKER}' \"$(pwd -P 2>/dev/null || pwd)\"; "
            f"exit $__claw_ec"
        )

    def consume_stdout(self, stdout: str) -> str:
        """Strip pwd marker line and update cwd when valid. Return clean stdout."""
        if not stdout:
            return stdout
        lines = stdout.splitlines(keepends=True)
        # Find last marker line (may lack trailing newline).
        idx = None
        for i in range(len(lines) - 1, -1, -1):
            raw = lines[i].rstrip("\n\r")
            if raw.startswith(PWD_MARKER):
                idx = i
                break
        if idx is None:
            return stdout
        marker_line = lines[idx].rstrip("\n\r")
        new_cwd = marker_line[len(PWD_MARKER) :]
        if new_cwd and os.path.isdir(new_cwd):
            try:
                self.cwd = str(Path(new_cwd).resolve())
            except OSError:
                pass
        cleaned = "".join(lines[:idx] + lines[idx + 1 :])
        # Drop a single trailing newline we introduced if stdout becomes empty-ish
        return cleaned


def session_for(
    run_context: object | None,
    sb: object | None = None,
    *,
    store: dict[int, ShellSession] | None = None,
) -> ShellSession:
    """Get or create a ShellSession bound to run_context or sandbox."""
    # Prefer attaching to run_context so subagents can inherit/override later.
    if run_context is not None:
        existing = getattr(run_context, "shell_session", None)
        if isinstance(existing, ShellSession):
            return existing
        initial = getattr(sb, "cwd", None) if sb is not None else None
        sess = ShellSession(cwd=str(Path(initial or Path.cwd()).resolve()))
        try:
            setattr(run_context, "shell_session", sess)
        except Exception:
            pass
        return sess
    initial = getattr(sb, "cwd", None) if sb is not None else None
    return ShellSession(cwd=str(Path(initial or Path.cwd()).resolve()))


__all__ = ["ShellSession", "PWD_MARKER", "session_for"]
