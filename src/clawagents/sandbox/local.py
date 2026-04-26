"""LocalBackend — SandboxBackend backed by the real filesystem."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from clawagents.sandbox.backend import DirEntry, ExecResult, FileStat


class LocalBackend:
    kind = "local"

    def __init__(self, root: str | None = None):
        self._cwd = str(Path(root or os.getcwd()).resolve())

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
        return await asyncio.to_thread(Path(path).read_text, encoding="utf-8")

    async def read_file_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(Path(path).read_bytes)

    async def write_file(self, path: str, content: str) -> None:
        await asyncio.to_thread(Path(path).write_text, content, encoding="utf-8")

    # ── Directory operations ────────────────────────────────────────

    async def read_dir(self, path: str) -> list[DirEntry]:
        def _read_dir() -> list[DirEntry]:
            p = Path(path)
            return [
                DirEntry(name=e.name, is_directory=e.is_dir(), is_file=e.is_file())
                for e in sorted(p.iterdir(), key=lambda x: x.name)
            ]

        return await asyncio.to_thread(_read_dir)

    async def mkdir(self, path: str, recursive: bool = False) -> None:
        await asyncio.to_thread(Path(path).mkdir, parents=recursive, exist_ok=True)

    # ── Metadata ────────────────────────────────────────────────────

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(path).exists)

    async def stat(self, path: str) -> FileStat:
        def _stat() -> FileStat:
            p = Path(path)
            s = p.stat()
            return FileStat(
                is_file=p.is_file(),
                is_directory=p.is_dir(),
                size=s.st_size,
                mtime_ms=s.st_mtime * 1000,
            )

        return await asyncio.to_thread(_stat)

    # ── Credential isolation ────────────────────────────────────────
    # Keys stripped from subprocess env to prevent credential leakage.
    # Claude-generated code running in execute() should never see API keys.
    _SENSITIVE_ENV_KEYS = frozenset({
        "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
        "ADVISOR_API_KEY", "ADVISOR_MODEL",
        "GATEWAY_API_KEY", "TAVILY_API_KEY",
        "TELEGRAM_BOT_TOKEN", "WHATSAPP_API_TOKEN",
        "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AZURE_API_KEY", "GOOGLE_API_KEY",
    })

    def _sanitized_env(self) -> dict[str, str]:
        return {k: v for k, v in os.environ.items()
                if k not in self._SENSITIVE_ENV_KEYS}

    # ── Command execution ───────────────────────────────────────────

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        merged_env = {**self._sanitized_env(), "PAGER": "cat"}
        if env:
            merged_env.update(env)

        timeout_s = (timeout or 30_000) / 1000.0

        # ``start_new_session`` puts the shell + every child it spawns
        # into a new process group. On timeout we send SIGKILL to the
        # whole group so long-running grandchildren (e.g. ``python
        # script.py`` started by ``sh -c``) don't orphan and keep
        # writing to a closed pipe.
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or self._cwd,
            env=merged_env,
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # Either the process group is already gone, or we don't
                # own it — fall back to killing just the parent.
                proc.kill()
            await proc.wait()
            return ExecResult(stdout="", stderr="", exit_code=1, killed=True)

        return ExecResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
        )
