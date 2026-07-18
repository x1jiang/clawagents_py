"""Grok-inspired shell session state for ``execute`` (cwd + sticky env).

Grok Build persists cwd/env via dump/replay after each command.
We track cwd and a filtered env overlay via stdout markers (no eval of
``export -p`` dumps).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PWD_MARKER = "__CLAW_PWD__"
ENV_MARKER = "__CLAW_ENV__"

_EXTRA_DENY_PREFIXES = (
    "SSH_",
    "GPG_",
    "AWS_",
    "GOOGLE_",
    "AZURE_",
    "KUBE",
    "DOCKER_",
    "NPM_",
    "PIP_",
)
_EXTRA_DENY_SUBSTR = (
    "PROXY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "API_KEY",
    "AUTH",
)


def _sticky_env_allowed(name: str) -> bool:
    if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return False
    upper = name.upper()
    try:
        from clawagents.redact import is_secret_name

        if is_secret_name(name):
            return False
    except Exception:
        pass
    try:
        from clawagents.sandbox.local import LocalBackend

        if name in LocalBackend._SENSITIVE_ENV_KEYS:
            return False
    except Exception:
        pass
    if upper.startswith("CLAW_") and any(
        s in upper for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")
    ):
        return False
    for p in _EXTRA_DENY_PREFIXES:
        if upper.startswith(p):
            return False
    for s in _EXTRA_DENY_SUBSTR:
        if s in upper:
            return False
    return True


def filter_sticky_env(env: dict[str, str]) -> dict[str, str]:
    """Keep only safe key/value pairs for sticky replay."""
    out: dict[str, str] = {}
    for k, v in env.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if not _sticky_env_allowed(k):
            continue
        # Cap value size to avoid huge dumps
        if len(v) > 8_192:
            continue
        out[k] = v
    return out


@dataclass
class ShellSession:
    """Per-agent shell session (state emulation, not a persistent process)."""

    cwd: str = field(default_factory=lambda: str(Path.cwd().resolve()))
    env: dict[str, str] = field(default_factory=dict)
    # Baseline process env at session start — sticky overlay = diffs only.
    _baseline: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._baseline:
            self._baseline = {
                k: v
                for k, v in os.environ.items()
                if isinstance(v, str) and _sticky_env_allowed(k)
            }

    def wrap(self, command: str, *, sticky_env: bool = True) -> str:
        """Prefix command so it runs in ``self.cwd`` (+ sticky env) and emits trailers."""
        q = shlex.quote(self.cwd)
        exports = ""
        if sticky_env and self.env:
            bits = []
            for k, v in filter_sticky_env(self.env).items():
                bits.append(f"export {shlex.quote(k)}={shlex.quote(v)}")
            if bits:
                exports = "; ".join(bits) + "; "

        # Dump filtered env via python so we never eval shell export dumps.
        dump_cmd = (
            "python3 -c \"import json,os;"
            "D=('SSH_','GPG_','AWS_','GOOGLE_','AZURE_','DOCKER_');"
            "S=('PROXY','TOKEN','SECRET','PASSWORD','PASSWD','CREDENTIAL','PRIVATE_KEY','API_KEY','AUTH');"
            "o={};"
            "[o.__setitem__(k,v) for k,v in os.environ.items() "
            "if isinstance(v,str) and len(v)<=8192 and k.isidentifier() "
            "and not k.upper().startswith(D) and not any(x in k.upper() for x in S) "
            "and not (k.upper().startswith('CLAW_') and any(x in k.upper() for x in "
            "('KEY','TOKEN','SECRET','PASSWORD','AUTH')))];"
            f"print('{ENV_MARKER}'+json.dumps(o,separators=(',',':')))\""
        )

        if sticky_env:
            return (
                f"cd {q} || exit 121; "
                f"{exports}"
                f"{command}; "
                f"__claw_ec=$?; "
                f"printf '%s%s\\n' '{PWD_MARKER}' \"$(pwd -P 2>/dev/null || pwd)\"; "
                f"{dump_cmd}; "
                f"exit $__claw_ec"
            )
        return (
            f"cd {q} || exit 121; "
            f"{command}; "
            f"__claw_ec=$?; "
            f"printf '%s%s\\n' '{PWD_MARKER}' \"$(pwd -P 2>/dev/null || pwd)\"; "
            f"exit $__claw_ec"
        )

    def consume_stdout(self, stdout: str, *, sticky_env: bool = True) -> str:
        """Strip marker lines; update cwd/env. Return clean stdout."""
        if not stdout:
            return stdout
        lines = stdout.splitlines(keepends=True)
        keep: list[str] = []
        for line in lines:
            raw = line.rstrip("\n\r")
            if raw.startswith(PWD_MARKER):
                new_cwd = raw[len(PWD_MARKER) :]
                if new_cwd and os.path.isdir(new_cwd):
                    try:
                        self.cwd = str(Path(new_cwd).resolve())
                    except OSError:
                        pass
                continue
            if sticky_env and raw.startswith(ENV_MARKER):
                payload = raw[len(ENV_MARKER) :]
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        dumped = filter_sticky_env(
                            {str(k): str(v) for k, v in data.items()}
                        )
                        # Keep only keys that differ from session baseline
                        # (exports / overrides), not the entire process env.
                        sticky: dict[str, str] = {}
                        for k, v in dumped.items():
                            if self._baseline.get(k) != v:
                                sticky[k] = v
                        self.env = sticky
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                continue
            keep.append(line)
        return "".join(keep)


def session_for(
    run_context: object | None,
    sb: object | None = None,
    *,
    store: dict[int, ShellSession] | None = None,
) -> ShellSession:
    """Get or create a ShellSession bound to run_context or sandbox."""
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


__all__ = [
    "ShellSession",
    "PWD_MARKER",
    "ENV_MARKER",
    "session_for",
    "filter_sticky_env",
]
