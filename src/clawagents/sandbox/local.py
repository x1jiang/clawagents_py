"""LocalBackend — SandboxBackend backed by the real filesystem."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

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
        # The explicit denylist above is a floor; the broad name-based matcher
        # (``*_TOKEN``/``*_API_KEY``/``*_SECRET``/``*PASSWORD*`` …) catches the
        # long tail (GITHUB_TOKEN, AWS_ACCESS_KEY_ID, DB_PASSWORD, …) that the
        # static set would otherwise leak into LLM-generated shell commands.
        from clawagents.redact import is_secret_name

        return {k: v for k, v in os.environ.items()
                if k not in self._SENSITIVE_ENV_KEYS and not is_secret_name(k)}

    # ── Command execution ───────────────────────────────────────────

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        *,
        max_output_chars: int | None = None,
        on_output: Any | None = None,
    ) -> ExecResult:
        from clawagents.utils.bounded_output import SpoolingTextAccumulator

        merged_env = {**self._sanitized_env(), "PAGER": "cat"}
        if env:
            merged_env.update(env)

        timeout_s = (timeout or 30_000) / 1000.0
        # Default matches TS bounded-process / tool exec retain budget.
        retain = int(max_output_chars) if max_output_chars and max_output_chars > 0 else 40_000
        stdout_acc = SpoolingTextAccumulator(retain)
        stderr_acc = SpoolingTextAccumulator(retain)

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

        def _feed(stream: str, data: bytes) -> None:
            text = data.decode("utf-8", errors="replace")
            if not text:
                return
            (stdout_acc if stream == "stdout" else stderr_acc).append(text)
            if on_output is not None:
                try:
                    on_output(stream, text)
                except Exception:
                    pass

        async def _pump(stream_name: str, reader: asyncio.StreamReader | None) -> None:
            if reader is None:
                return
            while True:
                chunk = await reader.read(65_536)
                if not chunk:
                    break
                _feed(stream_name, chunk)

        async def _run() -> int:
            await asyncio.gather(
                _pump("stdout", proc.stdout),
                _pump("stderr", proc.stderr),
                proc.wait(),
            )
            return proc.returncode or 0

        killed = False
        try:
            exit_code = await asyncio.wait_for(_run(), timeout=timeout_s)
        except asyncio.TimeoutError:
            killed = True
            exit_code = 1
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await proc.wait()
            except ProcessLookupError:
                pass
        stdout_path = stdout_acc.close()
        stderr_path = stderr_acc.close()
        return ExecResult(
            stdout=str(stdout_acc),
            stderr=str(stderr_acc),
            exit_code=exit_code,
            killed=killed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
