"""SandboxBackend — unified abstraction over filesystem + command execution.

Implementations:
    LocalBackend    — thin wrapper over pathlib/os    (production)
    InMemoryBackend — pure in-process VFS             (testing)
    DockerBackend   — ephemeral containers            (future)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DirEntry:
    name: str
    is_directory: bool
    is_file: bool


@dataclass(frozen=True)
class FileStat:
    is_file: bool
    is_directory: bool
    size: int
    mtime_ms: float


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    killed: bool = False


@runtime_checkable
class SandboxBackend(Protocol):
    """Minimal contract that all sandbox backends must implement."""

    @property
    def kind(self) -> str: ...

    @property
    def cwd(self) -> str: ...

    @property
    def sep(self) -> str: ...

    # ── Path helpers (pure, no I/O) ─────────────────────────────────

    def resolve(self, *segments: str) -> str: ...
    def relative(self, base: str, target: str) -> str: ...
    def dirname(self, path: str) -> str: ...
    def basename(self, path: str) -> str: ...
    def join(self, *segments: str) -> str: ...
    def safe_path(self, user_path: str) -> str: ...

    # ── File I/O ────────────────────────────────────────────────────

    async def read_file(self, path: str) -> str: ...
    async def read_file_bytes(self, path: str) -> bytes: ...
    async def write_file(self, path: str, content: str) -> None: ...

    # ── Directory operations ────────────────────────────────────────

    async def read_dir(self, path: str) -> list[DirEntry]: ...
    async def mkdir(self, path: str, recursive: bool = False) -> None: ...

    # ── Metadata ────────────────────────────────────────────────────

    async def exists(self, path: str) -> bool: ...
    async def stat(self, path: str) -> FileStat: ...

    # ── Command execution ───────────────────────────────────────────

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult: ...
