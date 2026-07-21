"""Named OS sandbox profiles with real path/network/exec enforcement."""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from clawagents.security.secret_paths import (
    DEFAULT_SECRET_GLOBS as _DEFAULT_SECRET_GLOBS,
    default_secret_globs as _default_secret_globs_central,
    path_matches_secret_globs as _path_matches_secret_globs_central,
)

BackendName = Literal["local", "docker", "seatbelt", "bwrap"]


@dataclass(frozen=True)
class OSSandboxProfile:
    name: str
    backend: BackendName = "local"
    read_only: bool = False
    network: bool = True
    allow_paths: tuple[str, ...] = ()
    deny_paths: tuple[str, ...] = ()
    env_allow: tuple[str, ...] = ()
    description: str = ""
    # When True, missing seatbelt/bwrap raises instead of soft-fallback.
    require_binary: bool = False
    # Paths denied for read+write (secret files) when fail-closed sandbox is on.
    secret_deny_paths: tuple[str, ...] = _DEFAULT_SECRET_GLOBS
    auto_allow_bash: bool = False


_BUILTIN: dict[str, OSSandboxProfile] = {
    "off": OSSandboxProfile(
        name="off",
        backend="local",
        description="No extra isolation (plain LocalBackend).",
    ),
    "workspace": OSSandboxProfile(
        name="workspace",
        backend="local",
        allow_paths=(".",),
        description="Confine paths to the workspace root.",
    ),
    "read-only": OSSandboxProfile(
        name="read-only",
        backend="local",
        read_only=True,
        allow_paths=(".",),
        description="Filesystem writes blocked; exec still allowed.",
    ),
    "strict": OSSandboxProfile(
        name="strict",
        backend="local",
        read_only=True,
        network=False,
        allow_paths=(".",),
        description="Read-only FS + network denied for child exec.",
    ),
    "docker": OSSandboxProfile(
        name="docker",
        backend="docker",
        network=False,
        description="Ephemeral DockerBackend when docker is available.",
    ),
    "seatbelt": OSSandboxProfile(
        name="seatbelt",
        backend="seatbelt",
        allow_paths=(".",),
        network=False,
        require_binary=False,
        description="macOS sandbox-exec: workspace write, network deny when available.",
    ),
    "bwrap": OSSandboxProfile(
        name="bwrap",
        backend="bwrap",
        allow_paths=(".",),
        network=False,
        require_binary=False,
        description="Linux bubblewrap: bind workspace, optional --unshare-net.",
    ),
    "devbox": OSSandboxProfile(
        name="devbox",
        backend="local",
        network=True,
        allow_paths=(".",),
        description="Developer box — writable workspace, network on.",
    ),
}




def _default_secret_globs() -> tuple[str, ...]:
    return _default_secret_globs_central()


def _path_matches_secret_globs(
    resolved: str,
    cwd: str,
    secret_globs: tuple[str, ...],
) -> bool:
    """True when ``resolved`` is a secret path under ``cwd`` (or basename match)."""
    return _path_matches_secret_globs_central(resolved, cwd, secret_globs)


def _scratch_roots() -> list[str]:
    """OS temp roots writable by in-process tools (parity with seatbelt /tmp allow).

    Models often ``write_file`` under ``/tmp`` while ``execute`` already can; without
    this, ProfileBackend rejects absolute temp paths that LocalBackend.safe_path
    cannot resolve inside the workspace.
    """
    seen: list[str] = []

    def _add(p: str) -> None:
        for form in (os.path.realpath(p), os.path.abspath(p)):
            if form and form not in seen:
                seen.append(form)

    try:
        _add(tempfile.gettempdir())
    except OSError:
        pass
    for extra in ("/tmp", "/private/tmp"):
        try:
            _add(extra)
        except OSError:
            pass
    return seen


def _resolve_secret_overlay_paths(
    workspace: str,
    secret_globs: tuple[str, ...],
) -> list[str]:
    """Expand secret deny globs to absolute paths for bwrap --ro-bind /dev/null overlays."""
    if not secret_globs:
        return []
    ws = Path(workspace).resolve()
    found: set[str] = set()
    for pattern in secret_globs:
        pat = (pattern or "").strip()
        if not pat:
            continue
        if "*" in pat or "?" in pat or "**" in pat:
            for hit in ws.glob(pat):
                if hit.is_file():
                    found.add(str(hit.resolve()))
            # Fail-closed: bind missing .env even when absent
            if pat in {".env", ".env.*"} or pat.startswith(".env"):
                env_candidate = ws / ".env"
                found.add(str(env_candidate.resolve()))
            continue
        target = (ws / pat).resolve()
        if target.is_file():
            found.add(str(target))
        elif target.parent.exists():
            found.add(str(target))
    # Always overlay workspace .env when secret deny is enabled
    found.add(str((ws / ".env").resolve()))
    return sorted(found)


def load_project_sandbox_toml(workspace: str | Path | None = None) -> dict[str, OSSandboxProfile]:
    """Load add-only custom profiles from ``.clawagents/sandbox.toml`` (JSON-compatible).

    Project configs may *add* profiles but never redefine built-in names.
    Supports a minimal JSON/TOML-ish ``{"profiles": {"name": {...}}}`` file
    (JSON body is accepted; full TOML is optional if tomllib is present).
    """
    import json

    ws = Path(workspace or os.getcwd())
    candidates = [
        ws / ".clawagents" / "sandbox.toml",
        ws / ".clawagents" / "sandbox.json",
        Path.home() / ".clawagents" / "sandbox.toml",
    ]
    found: dict[str, OSSandboxProfile] = {}
    conflicts: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw_text = path.read_text(encoding="utf-8")
            data: Any
            if path.suffix == ".json":
                data = json.loads(raw_text)
            else:
                try:
                    import tomllib
                    data = tomllib.loads(raw_text)
                except Exception:
                    data = json.loads(raw_text)
        except Exception:
            continue
        rows = data.get("profiles") if isinstance(data, dict) else None
        if not isinstance(rows, dict):
            continue
        for name, cfg in rows.items():
            key = str(name).strip().lower()
            if key in _BUILTIN:
                conflicts.append(key)
                continue
            if key in found:
                continue  # add-only: first wins
            if not isinstance(cfg, dict):
                continue
            found[key] = OSSandboxProfile(
                name=key,
                backend=cfg.get("backend", "local"),  # type: ignore[arg-type]
                read_only=bool(cfg.get("read_only", False)),
                network=bool(cfg.get("network", True)),
                allow_paths=tuple(cfg.get("allow_paths") or (".",)),
                deny_paths=tuple(cfg.get("deny_paths") or ()),
                require_binary=bool(cfg.get("require_binary", False)),
                secret_deny_paths=tuple(cfg.get("secret_deny_paths") or _default_secret_globs()),
                description=str(cfg.get("description") or f"project profile {key}"),
                auto_allow_bash=bool(cfg.get("auto_allow_bash", False)),
            )
    if conflicts:
        # Stash for diagnostics
        os.environ.setdefault(
            "CLAW_SANDBOX_PROFILE_CONFLICTS",
            ",".join(sorted(set(conflicts))),
        )
    return found


def get_profile(name: str | OSSandboxProfile | None) -> OSSandboxProfile:
    if name is None:
        return _BUILTIN["off"]
    if isinstance(name, OSSandboxProfile):
        return name
    key = str(name).strip().lower()
    if key in _BUILTIN:
        return _BUILTIN[key]
    project = load_project_sandbox_toml()
    if key in project:
        return project[key]
    known = sorted(set(_BUILTIN) | set(project))
    raise ValueError(f"Unknown sandbox profile: {name!r}. Known: {known}")


def list_profiles() -> list[OSSandboxProfile]:
    return [_BUILTIN[k] for k in sorted(_BUILTIN)]


def _seatbelt_basename_regex(glob: str) -> str:
    """Convert a secret glob to an anchored SBPL path-regex fragment.

    Mirrors ``_path_matches_secret_globs``: match the glob's basename component
    against any file under the workspace subtree (``([^/]+/)*`` = zero or more
    intermediate dirs), so both top-level ``key.pem`` and nested
    ``sub/key.pem`` are covered. The caller prepends the (regex-escaped)
    workspace root and wraps the result for ``(regex #"…")``.
    """
    base = os.path.basename(glob.replace("\\", "/").rstrip("/")) or glob
    frag = []
    for ch in base:
        if ch == "*":
            frag.append("[^/]*")
        elif ch == "?":
            frag.append("[^/]")
        else:
            frag.append(re.escape(ch))
    return "([^/]+/)*" + "".join(frag)


def sandbox_profile_for_chat_mode(
    chat_mode: str | None,
    *,
    allow_full_access: bool = False,
    explicit: str | None = None,
    env_profile: str | None = None,
) -> str | None:
    """Map chat UI mode → OS sandbox profile name.

    Layers stay coupled: permission mode and seatbelt cannot drift.

    Precedence:
    1. Explicit ``sandbox_profile=`` constructor arg (power users / tests)
    2. ``full_access`` + Settings gate → ``off``
    3. ``read_only`` → ``read-only``
    4. ``CLAW_SANDBOX_PROFILE`` env
    5. ``None`` → resolve_sandbox default (``workspace``)
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    mode = str(chat_mode or "").strip().lower()
    if mode == "full_access" and allow_full_access:
        return "off"
    if mode in ("read_only", "readonly", "plan"):
        return "read-only"
    if env_profile is not None and str(env_profile).strip():
        return str(env_profile).strip()
    return None


def _seatbelt_profile_text(
    *,
    cwd: str,
    network: bool,
    read_only: bool,
    secret_deny_paths: tuple[str, ...] = (),
) -> str:
    """Seatbelt profile with write confinement (deny file-write*, then allow workspace).

    ``(allow default)`` alone would still permit arbitrary writes — we always
    deny ``file-write*`` first, then re-allow only workspace (+ temp), matching
    Grok-style path enforcement rather than soft allow-default writes.

    SBPL is **last-match-wins**, so the secret deny rules are emitted *after*
    the workspace write-allow — otherwise the trailing ``(allow file-write*
    (subpath cwd))`` would silently override every secret write-deny. Secrets
    are matched by regex against the resolved path (literals only ever caught
    one exact filename and the old glob→literal reduction produced names like
    ``.pem``/``credentials`` that match no real file).
    """
    # Seatbelt canonicalises paths (resolves symlinks) before matching, so the
    # profile must anchor on the *real* path the child sees — e.g. macOS maps
    # ``/var/…`` → ``/private/var/…``. Emit both the real and abspath forms of
    # each root for allows and denies so neither can be bypassed via the
    # alternate spelling.
    def _roots(p: str) -> list[str]:
        seen: list[str] = []
        for form in (os.path.realpath(p), os.path.abspath(p)):
            if form not in seen:
                seen.append(form)
        return seen

    def _sb_str(s: str) -> str:
        # SBPL string literal: only backslash and double-quote need escaping.
        return s.replace("\\", "\\\\").replace('"', '\\"')

    cwd_roots = _roots(cwd)
    tmp_roots = _roots(tempfile.gettempdir())
    # Models habitually write to /tmp; on macOS that is distinct from
    # tempfile.gettempdir() (/var/folders/…). Allow both spellings.
    for extra in ("/tmp", "/private/tmp"):
        try:
            for form in _roots(extra):
                if form not in tmp_roots:
                    tmp_roots.append(form)
        except OSError:
            pass
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
    ]
    if not network:
        lines.append("(deny network*)")
    # CLIs (gcloud, git, shells) redirect to /dev/null constantly. Allow it in
    # both read-only and writable profiles — otherwise every `> /dev/null`
    # fails with "Operation not permitted" even when the real work is fine.
    lines.append('(allow file-write-data (literal "/dev/null"))')
    lines.append('(allow file-write-data (literal "/private/dev/null"))')
    if not read_only:
        for root in cwd_roots + tmp_roots:
            lines.append(f'(allow file-write* (subpath "{_sb_str(root)}"))')
    # Secret deny rules LAST so they win over the workspace write-allow above
    # (SBPL is last-match-wins). Regexes keep single backslashes: inside a
    # ``#"…"`` literal only ``"`` needs escaping, so the regex ``\.env`` is
    # written verbatim (double-escaping produced ``\\.env`` which matched
    # nothing).
    if secret_deny_paths:
        emitted: set[str] = set()
        for root in cwd_roots:
            root_re = re.escape(root)
            for glob in secret_deny_paths:
                pattern = f'^{root_re}/{_seatbelt_basename_regex(glob)}$'
                if pattern in emitted:
                    continue
                emitted.add(pattern)
                sb = pattern.replace('"', '\\"')
                lines.append(f'(deny file-read* (regex #"{sb}"))')
                lines.append(f'(deny file-write* (regex #"{sb}"))')
    return "\n".join(lines) + "\n"


class ProfileBackend:
    """Wrap a SandboxBackend applying read_only / allow / deny / network policy."""

    def __init__(self, inner: Any, profile: OSSandboxProfile):
        self._inner = inner
        self._profile = profile
        self.kind = f"profile:{profile.name}:{getattr(inner, 'kind', 'unknown')}"
        self.profile_warnings: list[str] = []

    @property
    def cwd(self) -> str:
        return self._inner.cwd

    @property
    def sep(self) -> str:
        return self._inner.sep

    def resolve(self, *segments: str) -> str:
        return self._inner.resolve(*segments)

    def relative(self, base: str, target: str) -> str:
        return self._inner.relative(base, target)

    def dirname(self, path: str) -> str:
        return self._inner.dirname(path)

    def basename(self, path: str) -> str:
        return self._inner.basename(path)

    def join(self, *segments: str) -> str:
        return self._inner.join(*segments)

    def _path_allowed(self, resolved: str) -> bool:
        allows = self._profile.allow_paths
        if not allows:
            return True
        for allow in allows:
            if allow in (".", ""):
                root = os.path.abspath(self.cwd)
            else:
                root = os.path.abspath(os.path.join(self.cwd, allow))
            if resolved == root or resolved.startswith(root + os.sep):
                return True
        # Match seatbelt: writable profiles may use OS temp / /tmp scratch
        # (write_file previously blocked these while execute could use them).
        if not self._profile.read_only:
            for root in _scratch_roots():
                if resolved == root or resolved.startswith(root + os.sep):
                    return True
        return False

    def _apply_deny_and_secrets(self, resolved: str, user_path: str) -> None:
        for deny in self._profile.deny_paths:
            deny_abs = os.path.abspath(os.path.join(self.cwd, deny))
            if resolved == deny_abs or resolved.startswith(deny_abs + os.sep):
                raise ValueError(
                    f"Path denied by profile {self._profile.name}: {user_path}"
                )
        secret_globs = getattr(self._profile, "secret_deny_paths", ()) or ()
        if secret_globs and _path_matches_secret_globs(resolved, self.cwd, secret_globs):
            raise ValueError(
                f"Secret path denied by profile {self._profile.name}: {user_path}"
            )

    def safe_path(self, user_path: str) -> str:
        raw = str(user_path or "")
        # Absolute/tilde paths under allow_paths or scratch must not go through
        # LocalBackend.safe_path first — that helper rejects anything outside cwd.
        if raw.startswith("~") or os.path.isabs(raw):
            candidate = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
            self._apply_deny_and_secrets(candidate, user_path)
            if not self._path_allowed(candidate):
                raise ValueError(
                    f"Path outside allow_paths for profile {self._profile.name}: {user_path}"
                )
            return candidate

        resolved = self._inner.safe_path(user_path)
        self._apply_deny_and_secrets(resolved, user_path)
        if not self._path_allowed(resolved):
            raise ValueError(
                f"Path outside allow_paths for profile {self._profile.name}: {user_path}"
            )
        return resolved

    async def read_file(self, path: str) -> str:
        # Enforce secret deny even if callers bypass safe_path.
        self.safe_path(path)
        return await self._inner.read_file(path)

    async def read_file_bytes(self, path: str) -> bytes:
        self.safe_path(path)
        return await self._inner.read_file_bytes(path)

    async def write_file(self, path: str, content: str) -> None:
        if self._profile.read_only:
            raise PermissionError(
                f"Profile {self._profile.name} is read-only; write blocked: {path}"
            )
        safe = self.safe_path(path)
        await self._inner.write_file(safe, content)

    async def read_dir(self, path: str) -> list:
        return await self._inner.read_dir(path)

    async def mkdir(self, path: str, recursive: bool = False) -> None:
        if self._profile.read_only:
            raise PermissionError(
                f"Profile {self._profile.name} is read-only; mkdir blocked: {path}"
            )
        safe = self.safe_path(path)
        await self._inner.mkdir(safe, recursive=recursive)

    async def exists(self, path: str) -> bool:
        return await self._inner.exists(path)

    async def stat(self, path: str):
        return await self._inner.stat(path)

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        if env is None and self._profile.network:
            return None
        base = dict(env or {})
        if not self._profile.network:
            base["CLAW_SANDBOX_NETWORK"] = "0"
            # Soft hints for common HTTP libs
            base.setdefault("HTTP_PROXY", "")
            base.setdefault("HTTPS_PROXY", "")
            base.setdefault("ALL_PROXY", "")
            base.setdefault("NO_PROXY", "*")
        return base

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        *,
        max_output_chars: int | None = None,
        on_output: Any | None = None,
    ):
        merged_env = self._merge_env(env)
        # Never interpolate the user command with !r — a single quote in
        # ``command`` flips Python's repr to double quotes and lets
        # $()/backticks expand in the outer shell BEFORE sandbox-exec.
        wrapped = self.wrap_command(command, cwd=cwd)
        return await self._inner.exec(
            wrapped,
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
            max_output_chars=max_output_chars,
            on_output=on_output,
        )

    def wrap_command(self, command: str, *, cwd: str | None = None) -> str:
        """Wrap ``command`` for seatbelt/bwrap without executing it.

        Used by background ``execute`` so isolation matches foreground
        ``ProfileBackend.exec``. Appends to ``profile_warnings`` on soft fallback.
        """
        backend = self._profile.backend
        if backend == "seatbelt":
            binary = shutil.which("sandbox-exec")
            if not binary:
                msg = "sandbox-exec unavailable; falling back to local exec"
                self.profile_warnings.append(msg)
                if self._profile.require_binary:
                    raise RuntimeError(msg)
                return command
            profile_text = _seatbelt_profile_text(
                cwd=self.cwd,
                network=self._profile.network,
                read_only=self._profile.read_only,
                secret_deny_paths=getattr(self._profile, "secret_deny_paths", ()),
            )
            try:
                from clawagents import __version__ as _ca_ver
            except Exception:
                _ca_ver = "?"
            profile_text = (
                f"; generated by clawagents {_ca_ver} "
                f"profile={self._profile.name}\n"
                + profile_text
            )
            profile_path = Path(self.cwd) / ".clawagents" / "seatbelt.sb"
            try:
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profile_path.write_text(profile_text, encoding="utf-8")
            except OSError as exc:
                self.profile_warnings.append(f"seatbelt profile write failed: {exc}")
                if self._profile.require_binary:
                    raise
                return command
            # Use module-level ``shlex`` only — never ``import shlex`` in this
            # method (local import makes shlex unbound on other branches).
            return " ".join(
                shlex.quote(p)
                for p in [
                    binary,
                    "-f",
                    str(profile_path),
                    "/bin/sh",
                    "-c",
                    command,
                ]
            )
        if backend == "bwrap":
            binary = shutil.which("bwrap")
            if not binary:
                msg = "bwrap unavailable; falling back to local exec"
                self.profile_warnings.append(msg)
                if self._profile.require_binary:
                    raise RuntimeError(msg)
                return command
            net = [] if self._profile.network else ["--unshare-net"]
            ro = ["--ro-bind", "/", "/"]
            bind = ["--bind", self.cwd, self.cwd]
            if self._profile.read_only:
                bind = ["--ro-bind", self.cwd, self.cwd]
            secret_overlays: list[str] = []
            secret_globs = getattr(self._profile, "secret_deny_paths", ())
            if secret_globs:
                for secret_path in _resolve_secret_overlay_paths(
                    self.cwd, secret_globs
                ):
                    try:
                        sp = Path(secret_path)
                        if not sp.exists():
                            sp.parent.mkdir(parents=True, exist_ok=True)
                            sp.touch(exist_ok=True)
                        secret_overlays.extend(["--ro-bind", "/dev/null", str(sp)])
                    except OSError as exc:
                        self.profile_warnings.append(
                            f"bwrap secret overlay skipped {secret_path}: {exc}"
                        )
            parts = [
                binary,
                "--die-with-parent",
                *ro,
                *net,
                *bind,
                *secret_overlays,
                "--chdir",
                cwd or self.cwd,
                "/bin/sh",
                "-c",
                command,
            ]
            return " ".join(shlex.quote(p) for p in parts)
        return command


def resolve_sandbox(
    profile: str | OSSandboxProfile | None = None,
    *,
    workspace: str | None = None,
    default: str | None = None,
) -> Any:
    """Build a SandboxBackend for the named profile.

    ``default`` is used when ``profile`` is None (e.g. ``workspace`` for
    create_claw_agent). Feature flag ``os_sandbox_profiles`` forces ``off``
    when disabled.
    """
    from clawagents.config.features import is_enabled
    from clawagents.sandbox.local import LocalBackend

    if not is_enabled("os_sandbox_profiles"):
        chosen: str | OSSandboxProfile | None = "off"
    elif profile is not None:
        chosen = profile
    else:
        chosen = default or "off"

    prof = get_profile(chosen)
    if prof.backend == "docker":
        try:
            from clawagents.sandbox.docker import DockerBackend

            inner: Any = DockerBackend(root=workspace)
        except Exception:
            inner = LocalBackend(root=workspace)
            wrapped = ProfileBackend(inner, prof)
            wrapped.profile_warnings.append("DockerBackend unavailable; using local")
            return wrapped
    else:
        inner = LocalBackend(root=workspace)

    # Fail-closed: when feature on, require real OS sandbox binaries.
    try:
        from clawagents.config.features import is_enabled as _feat_sb
        if _feat_sb("sandbox_fail_closed") and prof.name != "off":
            prof = OSSandboxProfile(
                name=prof.name,
                backend=prof.backend if prof.backend in ("seatbelt", "bwrap", "docker") else (
                    "seatbelt" if os.uname().sysname == "Darwin" else "bwrap"
                ),
                read_only=prof.read_only,
                network=prof.network,
                allow_paths=prof.allow_paths or (".",),
                deny_paths=prof.deny_paths,
                env_allow=prof.env_allow,
                require_binary=True,
                secret_deny_paths=getattr(prof, "secret_deny_paths", _default_secret_globs()),
                description=prof.description,
                auto_allow_bash=getattr(prof, "auto_allow_bash", False),
            )
    except Exception:
        pass

    # Auto-upgrade path-confined / network-deny local profiles onto real OS
    # sandboxes when binaries exist (workspace writes stay confined).
    _wants_os = (
        prof.backend == "local"
        and prof.name != "off"
        and (bool(prof.allow_paths) or not prof.network or prof.read_only)
    )
    if (
        _wants_os
        and shutil.which("sandbox-exec")
        and os.uname().sysname == "Darwin"
    ):
        prof = OSSandboxProfile(
            name=prof.name,
            backend="seatbelt",
            read_only=prof.read_only,
            network=prof.network,
            allow_paths=prof.allow_paths or (".",),
            deny_paths=prof.deny_paths,
            env_allow=prof.env_allow,
            require_binary=prof.require_binary,
            secret_deny_paths=getattr(prof, "secret_deny_paths", _default_secret_globs()),
            description=prof.description,
            auto_allow_bash=getattr(prof, "auto_allow_bash", False),
        )
    elif (
        _wants_os
        and shutil.which("bwrap")
        and os.uname().sysname == "Linux"
    ):
        prof = OSSandboxProfile(
            name=prof.name,
            backend="bwrap",
            read_only=prof.read_only,
            network=prof.network,
            allow_paths=prof.allow_paths or (".",),
            deny_paths=prof.deny_paths,
            env_allow=prof.env_allow,
            require_binary=prof.require_binary,
            secret_deny_paths=getattr(prof, "secret_deny_paths", _default_secret_globs()),
            description=prof.description,
            auto_allow_bash=getattr(prof, "auto_allow_bash", False),
        )

    if prof.name == "off" and not prof.read_only and not prof.deny_paths and not prof.allow_paths:
        return inner
    return ProfileBackend(inner, prof)


__all__ = [
    "OSSandboxProfile",
    "ProfileBackend",
    "get_profile",
    "list_profiles",
    "resolve_sandbox",
]
