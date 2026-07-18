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

# Hard caps — sticky overlay is for a handful of exports, not full env dumps.
MAX_STICKY_KEYS = 64
MAX_STICKY_VALUE_BYTES = 4_096
MAX_STICKY_EXPORT_CHARS = 24_000

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
    "HTTP_",
    "HTTPS_",
    "FTP_",
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

# Inherited process noise — never stick these even if they differ from baseline.
_COMMON_ENV_SKIP = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "TERMINFO",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "DISPLAY",
        "XPC_FLAGS",
        "XPC_SERVICE_NAME",
        "TERM_SESSION_ID",
        "COLORTERM",
        "COMMAND_MODE",
        "__CF_USER_TEXT_ENCODING",
        "SECURITYSESSIONID",
    }
)


def _sticky_env_allowed(name: str) -> bool:
    if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return False
    if name in _COMMON_ENV_SKIP or name.startswith("LC_"):
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
    """Keep only safe key/value pairs for sticky replay (capped)."""
    out: dict[str, str] = {}
    for k, v in env.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if not _sticky_env_allowed(k):
            continue
        if len(v) > MAX_STICKY_VALUE_BYTES:
            continue
        out[k] = v
        if len(out) >= MAX_STICKY_KEYS:
            break
    return out


def _apply_env_payload(session: "ShellSession", payload: str) -> None:
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return
    if not isinstance(data, dict):
        return
    coerced: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, bool):
            coerced[str(k)] = "1" if v else "0"
        elif isinstance(v, (str, int, float)):
            coerced[str(k)] = str(v)
        # skip null / objects / arrays
    dumped = filter_sticky_env(coerced)
    sticky: dict[str, str] = {}
    for k, v in dumped.items():
        if session._baseline.get(k) != v:
            sticky[k] = v
        if len(sticky) >= MAX_STICKY_KEYS:
            break
    session.env = sticky


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
            total = 0
            for k, v in filter_sticky_env(self.env).items():
                piece = f"export {shlex.quote(k)}={shlex.quote(v)}"
                if total + len(piece) > MAX_STICKY_EXPORT_CHARS:
                    break
                bits.append(piece)
                total += len(piece) + 2
            if bits:
                exports = "; ".join(bits) + "; "

        # Dump filtered env via python so we never eval shell export dumps.
        # Prefer python3, then python; never fail the user command if missing.
        # consume_stdout applies the full denylist + baseline diff.
        py_dump = f"""import json, os
D = ("SSH_", "GPG_", "AWS_", "GOOGLE_", "AZURE_", "DOCKER_", "NPM_", "PIP_", "HTTP_", "HTTPS_")
S = ("PROXY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE_KEY", "API_KEY", "AUTH")
SKIP = set("PATH HOME USER LOGNAME SHELL TERM TMPDIR TMP TEMP PWD OLDPWD SHLVL LANG LANGUAGE LC_ALL LC_CTYPE DISPLAY _".split())
o = {{}}
for k, v in sorted(os.environ.items()):
    if len(o) >= 64:
        break
    if not isinstance(v, str) or len(v) > 4096 or not k.isidentifier():
        continue
    if k in SKIP or k.startswith("LC_"):
        continue
    u = k.upper()
    if u.startswith(D) or any(x in u for x in S):
        continue
    if u.startswith("CLAW_") and any(x in u for x in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")):
        continue
    o[k] = v
print({ENV_MARKER!r} + json.dumps(o, separators=(",", ":")))
"""
        dump_cmd = (
            "__claw_py=$(command -v python3 2>/dev/null || command -v python 2>/dev/null); "
            'if [ -n "$__claw_py" ]; then '
            f'"$__claw_py" -c {shlex.quote(py_dump)}; '
            "fi"
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
        """Strip trailing marker trailers; update cwd/env. Return clean stdout.

        Only the trailing marker block is consumed (PWD then ENV). Mid-output
        lines that look like markers are left intact to avoid poisoning.
        """
        if not stdout:
            return stdout
        lines = stdout.splitlines(keepends=True)
        if not lines:
            return stdout

        pwd_raw: str | None = None
        env_raw: str | None = None
        idx = len(lines) - 1
        peeled = 0
        # Peel at most a few trailing non-empty lines for markers.
        while idx >= 0 and peeled < 6:
            raw = lines[idx].rstrip("\n\r")
            if not raw:
                idx -= 1
                continue
            if sticky_env and env_raw is None and raw.startswith(ENV_MARKER):
                env_raw = raw[len(ENV_MARKER) :]
                idx -= 1
                peeled += 1
                continue
            if pwd_raw is None and raw.startswith(PWD_MARKER):
                pwd_raw = raw[len(PWD_MARKER) :]
                idx -= 1
                peeled += 1
                continue
            break

        keep_end = idx + 1
        if pwd_raw is not None and pwd_raw and os.path.isdir(pwd_raw):
            try:
                self.cwd = str(Path(pwd_raw).resolve())
            except OSError:
                pass
        if sticky_env and env_raw is not None:
            _apply_env_payload(self, env_raw)
        return "".join(lines[:keep_end])


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
