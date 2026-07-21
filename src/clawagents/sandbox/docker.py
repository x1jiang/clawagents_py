from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from clawagents.redact import is_secret_name
from clawagents.sandbox.backend import DirEntry, ExecResult, FileStat, SandboxBackend

# Vendor-prefixed shapes (``GITHUB_PAT``, ``STRIPE_SK_LIVE``) and infra
# names (``DATABASE_URL``, ``DSN``) that don't contain any of redact's
# generic hint words.
_SANDBOX_EXTRA_ENV_RE = re.compile(
    r"(?:SK[_-]?LIVE|SK[_-]?TEST|GITHUB[_-]?PAT|"
    r"DATABASE[_-]?URL|CONNECTION[_-]?STRING|DSN)",
    re.I,
)


def _is_sensitive_env(name: str) -> bool:
    return is_secret_name(name) or bool(_SANDBOX_EXTRA_ENV_RE.search(name))


class DockerBackend:
    """Sandbox backend that runs commands in ephemeral Docker containers.

    File operations use the mounted host workspace directly; shell execution
    happens through ``docker run --rm`` with explicit mounts and scrubbed env.
    """

    kind = "docker"
    sep = os.sep

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        image: str = "python:3.12-alpine",
        docker_bin: str = "docker",
        container_workdir: str = "/workspace",
        read_only_root: bool = False,
    ):
        self.cwd = str(Path(root or Path.cwd()).resolve())
        self.image = image
        self.docker_bin = docker_bin
        self.container_workdir = container_workdir
        self.read_only_root = read_only_root

    def resolve(self, *segments: str) -> str:
        return str(Path(self.cwd, *segments).resolve())

    def relative(self, from_path: str, to_path: str) -> str:
        return os.path.relpath(to_path, from_path)

    def dirname(self, path: str) -> str:
        return os.path.dirname(path)

    def basename(self, path: str) -> str:
        return os.path.basename(path)

    def join(self, *segments: str) -> str:
        return os.path.join(*segments)

    def safe_path(self, user_path: str) -> str:
        root = Path(self.cwd)
        p = Path(user_path)
        if not p.is_absolute():
            p = root / p
        resolved = p.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Path traversal blocked: {user_path}")
        return str(resolved)

    async def read_file(self, path: str, encoding: str = "utf-8") -> str:
        return await asyncio.to_thread(Path(path).read_text, encoding=encoding)

    async def read_file_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(Path(path).read_bytes)

    async def write_file(self, path: str, content: str, encoding: str = "utf-8") -> None:
        await asyncio.to_thread(Path(path).write_text, content, encoding=encoding)

    async def read_dir(self, path: str) -> list[DirEntry]:
        def _read() -> list[DirEntry]:
            return [
                DirEntry(name=p.name, is_directory=p.is_dir(), is_file=p.is_file())
                for p in Path(path).iterdir()
            ]
        return await asyncio.to_thread(_read)

    async def mkdir(self, path: str, recursive: bool = False) -> None:
        await asyncio.to_thread(Path(path).mkdir, parents=recursive, exist_ok=recursive)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(path).exists)

    async def stat(self, path: str) -> FileStat:
        def _stat() -> FileStat:
            s = Path(path).stat()
            p = Path(path)
            return FileStat(is_file=p.is_file(), is_directory=p.is_dir(), size=s.st_size, mtime_ms=s.st_mtime * 1000)
        return await asyncio.to_thread(_stat)

    def build_docker_args(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Optional[Dict[str, str]] = None,
    ) -> list[str]:
        host_cwd = Path(self.safe_path(cwd)) if cwd else Path(self.cwd)
        rel = os.path.relpath(host_cwd, self.cwd)
        container_cwd = self.container_workdir if rel == "." else f"{self.container_workdir}/{rel}"
        mount_mode = "ro" if self.read_only_root else "rw"
        args = [
            "run", "--rm",
            "-v", f"{self.cwd}:{self.container_workdir}:{mount_mode}",
            "-w", container_cwd,
        ]
        for key, value in (env or {}).items():
            if not _is_sensitive_env(key):
                args.extend(["-e", f"{key}={value}"])
        args.extend([self.image, "sh", "-lc", command])
        return args

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **_kwargs: Any,
    ) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin,
            *self.build_docker_args(
                command,
                cwd=cwd,
                env=env,
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout if timeout is not None else 30_000) / 1000.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecResult(stdout="", stderr="", exit_code=1, killed=True)
        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            exit_code=int(proc.returncode or 0),
        )
