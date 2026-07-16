"""Named OS sandbox profiles (Grok Build-inspired abstraction).

Profiles describe intended isolation. Enforcement today:
  - ``local`` / ``read-only`` / ``workspace`` / ``strict`` → LocalBackend with
    optional write denial via ProfileBackend wrapper
  - ``docker`` → DockerBackend when available
  - ``seatbelt`` / ``bwrap`` → LocalBackend + exec wrapper when the binary exists,
    otherwise soft-fallback to local with a warning flag
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
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
        description="Filesystem writes blocked; exec still allowed.",
    ),
    "strict": OSSandboxProfile(
        name="strict",
        backend="local",
        read_only=True,
        network=False,
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
        description="macOS sandbox-exec wrapper when available.",
    ),
    "bwrap": OSSandboxProfile(
        name="bwrap",
        backend="bwrap",
        description="Linux bubblewrap wrapper when available.",
    ),
    "devbox": OSSandboxProfile(
        name="devbox",
        backend="local",
        network=True,
        description="Developer box — writable workspace, network on.",
    ),
}


def get_profile(name: str | OSSandboxProfile | None) -> OSSandboxProfile:
    if name is None:
        return _BUILTIN["off"]
    if isinstance(name, OSSandboxProfile):
        return name
    key = str(name).strip().lower()
    if key in _BUILTIN:
        return _BUILTIN[key]
    raise ValueError(f"Unknown sandbox profile: {name!r}. Known: {sorted(_BUILTIN)}")


def list_profiles() -> list[OSSandboxProfile]:
    return [_BUILTIN[k] for k in sorted(_BUILTIN)]


class ProfileBackend:
    """Wrap a SandboxBackend applying read_only / path policy."""

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

    def safe_path(self, user_path: str) -> str:
        resolved = self._inner.safe_path(user_path)
        for deny in self._profile.deny_paths:
            deny_abs = os.path.abspath(os.path.join(self.cwd, deny))
            if resolved == deny_abs or resolved.startswith(deny_abs + os.sep):
                raise ValueError(f"Path denied by profile {self._profile.name}: {user_path}")
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

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        wrapped = command
        if self._profile.backend == "seatbelt" and shutil.which("sandbox-exec"):
            # Minimal allow-default profile inline — hosts can replace later.
            wrapped = f"sandbox-exec -p '(version 1)(allow default)' /bin/sh -c {command!r}"
        elif self._profile.backend == "bwrap" and shutil.which("bwrap"):
            net = [] if self._profile.network else ["--unshare-net"]
            wrapped = (
                "bwrap --die-with-parent --ro-bind / / "
                + " ".join(net)
                + f" --bind {self.cwd} {self.cwd} --chdir {cwd or self.cwd} "
                + f"/bin/sh -c {command!r}"
            )
        elif self._profile.backend in ("seatbelt", "bwrap"):
            self.profile_warnings.append(
                f"{self._profile.backend} binary unavailable; falling back to local exec"
            )
        if not self._profile.network and env is not None:
            env = {**env, "CLAW_SANDBOX_NETWORK": "0"}
        return await self._inner.exec(wrapped, timeout=timeout, cwd=cwd, env=env)


def resolve_sandbox(
    profile: str | OSSandboxProfile | None = None,
    *,
    workspace: str | None = None,
) -> Any:
    """Build a SandboxBackend for the named profile."""
    from clawagents.config.features import is_enabled
    from clawagents.sandbox.local import LocalBackend

    prof = get_profile(profile if is_enabled("os_sandbox_profiles") else "off")
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

    if prof.name == "off" and not prof.read_only and not prof.deny_paths:
        return inner
    return ProfileBackend(inner, prof)


__all__ = [
    "OSSandboxProfile",
    "ProfileBackend",
    "get_profile",
    "list_profiles",
    "resolve_sandbox",
]
