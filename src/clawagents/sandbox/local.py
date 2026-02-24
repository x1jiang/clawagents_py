"""LocalBackend — SandboxBackend backed by the real filesystem."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from clawagents.sandbox.backend import DirEntry, ExecResult, FileStat


class LocalBackend:
    kind = "local"

    def __init__(self, root: str | None = None):
        self._cwd = root or os.getcwd()

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def sep(self) -> str:
        return os.sep

    # ── Path helpers ────────────────────────────────────────────────

    def resolve(self, *segments: str) -> str:
        return str(Path(self._cwd, *segments).resolve())

    def relative(self, base: str, target: str) -> str:
        return os.path.relpath(target, base)

    def dirname(self, path: str) -> str:
        return str(Path(path).parent)

    def basename(self, path: str) -> str:
        return Path(path).name

    def join(self, *segments: str) -> str:
        return os.path.join(*segments)

    def safe_path(self, user_path: str) -> str:
        resolved = str(Path(self._cwd, user_path).resolve())
        if resolved != self._cwd and not resolved.startswith(self._cwd + os.sep):
            raise ValueError(f"Path traversal blocked: {user_path}")
        return resolved

    # ── File I/O ────────────────────────────────────────────────────

    async def read_file(self, path: str) -> str:
        return Path(path).read_text("utf-8")

    async def read_file_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_file(self, path: str, content: str) -> None:
        Path(path).write_text(content, "utf-8")

    # ── Directory operations ────────────────────────────────────────

    async def read_dir(self, path: str) -> list[DirEntry]:
        p = Path(path)
        return [
            DirEntry(name=e.name, is_directory=e.is_dir(), is_file=e.is_file())
            for e in sorted(p.iterdir(), key=lambda x: x.name)
        ]

    async def mkdir(self, path: str, recursive: bool = False) -> None:
        Path(path).mkdir(parents=recursive, exist_ok=True)

    # ── Metadata ────────────────────────────────────────────────────

    async def exists(self, path: str) -> bool:
        return Path(path).exists()

    async def stat(self, path: str) -> FileStat:
        s = Path(path).stat()
        p = Path(path)
        return FileStat(
            is_file=p.is_file(),
            is_directory=p.is_dir(),
            size=s.st_size,
            mtime_ms=s.st_mtime * 1000,
        )

    # ── Command execution ───────────────────────────────────────────

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        merged_env = {**os.environ, "PAGER": "cat"}
        if env:
            merged_env.update(env)

        timeout_s = (timeout or 30_000) / 1000.0

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self._cwd,
            env=merged_env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(stdout="", stderr="", exit_code=1, killed=True)

        return ExecResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
        )
