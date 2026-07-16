"""Named OS sandbox profiles with real path/network/exec enforcement."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    secret_deny_paths: tuple[str, ...] = (
        ".env", ".env.*", "**/credentials*", "**/secrets*", "**/*.pem",
    )
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
    return (".env", ".env.*", "**/credentials*", "**/secrets*", "**/*.pem")


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
    """
    safe = cwd.replace("\\", "\\\\").replace('"', '\\"')
    tmp = tempfile.gettempdir().replace("\\", "\\\\").replace('"', '\\"')
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
    ]
    # Fail-closed secret reads: deny literal .env-class files under workspace.
    for glob in secret_deny_paths or ():
        base = glob.replace("**/", "").replace("*", "")
        if not base or "/" in base.strip("."):
            # Only emit simple basename literals for seatbelt (full glob
            # expansion is Linux/bwrap's job).
            if glob in {".env", "credentials", "secrets"} or glob.startswith(".env"):
                lit = f"{safe}/{glob}".replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'(deny file-read* (literal "{lit}"))')
                lines.append(f'(deny file-write* (literal "{lit}"))')
            continue
        lit = f'{safe}/{base}'.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'(deny file-read* (literal "{lit}"))')
        lines.append(f'(deny file-write* (literal "{lit}"))')
    # Always deny workspace/.env when secrets enabled
    if secret_deny_paths:
        env_lit = f'{safe}/.env'.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'(deny file-read* (literal "{env_lit}"))')
        lines.append(f'(deny file-write* (literal "{env_lit}"))')
    if not network:
        lines.append("(deny network*)")
    if read_only:
        lines.append('(allow file-write-data (literal "/dev/null"))')
    else:
        lines.append(f'(allow file-write* (subpath "{safe}"))')
        lines.append(f'(allow file-write* (subpath "{tmp}"))')
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
        return False

    def safe_path(self, user_path: str) -> str:
        resolved = self._inner.safe_path(user_path)
        for deny in self._profile.deny_paths:
            deny_abs = os.path.abspath(os.path.join(self.cwd, deny))
            if resolved == deny_abs or resolved.startswith(deny_abs + os.sep):
                raise ValueError(
                    f"Path denied by profile {self._profile.name}: {user_path}"
                )
        if not self._path_allowed(resolved):
            raise ValueError(
                f"Path outside allow_paths for profile {self._profile.name}: {user_path}"
            )
        return resolved

    async def read_file(self, path: str) -> str:
        return await self._inner.read_file(path)

    async def read_file_bytes(self, path: str) -> bytes:
        return await self._inner.read_file_bytes(path)

    async def write_file(self, path: str, content: str) -> None:
        if self._profile.read_only:
            raise PermissionError(
                f"Profile {self._profile.name} is read-only; write blocked: {path}"
            )
        await self._inner.write_file(path, content)

    async def read_dir(self, path: str) -> list:
        return await self._inner.read_dir(path)

    async def mkdir(self, path: str, recursive: bool = False) -> None:
        if self._profile.read_only:
            raise PermissionError(
                f"Profile {self._profile.name} is read-only; mkdir blocked: {path}"
            )
        await self._inner.mkdir(path, recursive=recursive)

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
    ):
        merged_env = self._merge_env(env)
        wrapped = command
        backend = self._profile.backend

        if backend == "seatbelt":
            binary = shutil.which("sandbox-exec")
            if binary:
                profile_text = _seatbelt_profile_text(
                    cwd=self.cwd,
                    network=self._profile.network,
                    read_only=self._profile.read_only,
                    secret_deny_paths=getattr(
                        self._profile, "secret_deny_paths", ()
                    ),
                )
                profile_path = Path(self.cwd) / ".clawagents" / "seatbelt.sb"
                try:
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    profile_path.write_text(profile_text, encoding="utf-8")
                    wrapped = (
                        f"{binary} -f {profile_path!s} /bin/sh -c {command!r}"
                    )
                except OSError as exc:
                    self.profile_warnings.append(f"seatbelt profile write failed: {exc}")
                    if self._profile.require_binary:
                        raise
            else:
                msg = "sandbox-exec unavailable; falling back to local exec"
                self.profile_warnings.append(msg)
                if self._profile.require_binary:
                    raise RuntimeError(msg)

        elif backend == "bwrap":
            binary = shutil.which("bwrap")
            if binary:
                net = [] if self._profile.network else ["--unshare-net"]
                ro = ["--ro-bind", "/", "/"]
                # Remount workspace writable unless read_only
                bind = ["--bind", self.cwd, self.cwd]
                if self._profile.read_only:
                    bind = ["--ro-bind", self.cwd, self.cwd]
                parts = [
                    binary,
                    "--die-with-parent",
                    *ro,
                    *net,
                    *bind,
                    "--chdir",
                    cwd or self.cwd,
                    "/bin/sh",
                    "-c",
                    command,
                ]
                # Pass as a shell-escaped single command for LocalBackend.exec
                import shlex

                wrapped = " ".join(shlex.quote(p) for p in parts)
            else:
                msg = "bwrap unavailable; falling back to local exec"
                self.profile_warnings.append(msg)
                if self._profile.require_binary:
                    raise RuntimeError(msg)

        return await self._inner.exec(
            wrapped, timeout=timeout, cwd=cwd, env=merged_env
        )


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
