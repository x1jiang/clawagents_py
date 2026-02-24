"""InMemoryBackend — pure in-process virtual filesystem + exec stub.

Designed for fast, deterministic testing of the full agent loop
without touching the real filesystem. No temp directories, no cleanup.

Usage:
    mem = InMemoryBackend("/project")
    mem.seed({"src/main.py": "print('hi')"})
    tools = create_filesystem_tools(mem)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from clawagents.sandbox.backend import DirEntry, ExecResult, FileStat

ExecStub = Callable[..., Awaitable[ExecResult] | ExecResult]


class _VFSNode:
    __slots__ = ("kind", "content", "raw_bytes", "mtime_ms")

    def __init__(
        self,
        kind: str,
        content: str | None = None,
        raw_bytes: bytes | None = None,
        mtime_ms: float | None = None,
    ):
        self.kind = kind
        self.content = content
        self.raw_bytes = raw_bytes
        self.mtime_ms = mtime_ms or _now_ms()


def _now_ms() -> float:
    import time
    return time.time() * 1000


def _default_exec_stub(*_a: Any, **_kw: Any) -> ExecResult:
    return ExecResult(stdout="", stderr="exec not available in memory backend", exit_code=1)


class InMemoryBackend:
    kind = "memory"
    sep = "/"

    def __init__(self, root: str = "/project", exec_stub: ExecStub | None = None):
        self._cwd = root
        self._nodes: dict[str, _VFSNode] = {}
        self._exec_stub = exec_stub or _default_exec_stub
        self._nodes[self._normalize(root)] = _VFSNode(kind="dir")

    @property
    def cwd(self) -> str:
        return self._cwd

    def seed(self, files: dict[str, str | bytes]) -> None:
        """Pre-populate the VFS. Keys are paths relative to cwd."""
        for rel_path, content in files.items():
            abs_path = self.resolve(rel_path)
            self._ensure_parent_dirs(abs_path)
            if isinstance(content, bytes):
                self._nodes[abs_path] = _VFSNode(kind="file", raw_bytes=content)
            else:
                self._nodes[abs_path] = _VFSNode(kind="file", content=content)

    def snapshot(self) -> dict[str, str]:
        """Return a snapshot of all files (relative paths -> contents)."""
        result: dict[str, str] = {}
        for abs_path, node in self._nodes.items():
            if node.kind == "file":
                rel = self.relative(self._cwd, abs_path)
                result[rel] = node.content or (node.raw_bytes.decode("utf-8") if node.raw_bytes else "")
        return result

    # ── Path helpers ────────────────────────────────────────────────

    def resolve(self, *segments: str) -> str:
        path = self._cwd
        for seg in segments:
            if seg.startswith("/"):
                path = seg
            else:
                path = path + "/" + seg
        return self._normalize(path)

    def relative(self, base: str, target: str) -> str:
        b = self._normalize(base)
        t = self._normalize(target)
        if t.startswith(b + "/"):
            return t[len(b) + 1:]
        if t == b:
            return "."
        return t

    def dirname(self, path: str) -> str:
        n = self._normalize(path)
        idx = n.rfind("/")
        return n[:idx] if idx > 0 else "/"

    def basename(self, path: str) -> str:
        n = self._normalize(path)
        idx = n.rfind("/")
        return n[idx + 1:] if idx >= 0 else n

    def join(self, *segments: str) -> str:
        return self._normalize("/".join(segments))

    def safe_path(self, user_path: str) -> str:
        resolved = self.resolve(user_path)
        if resolved != self._cwd and not resolved.startswith(self._cwd + "/"):
            raise ValueError(f"Path traversal blocked: {user_path}")
        return resolved

    # ── File I/O ────────────────────────────────────────────────────

    async def read_file(self, path: str) -> str:
        n = self._normalize(path)
        node = self._nodes.get(n)
        if not node or node.kind != "file":
            raise FileNotFoundError(f"No such file: {path}")
        return node.content or (node.raw_bytes.decode("utf-8") if node.raw_bytes else "")

    async def read_file_bytes(self, path: str) -> bytes:
        n = self._normalize(path)
        node = self._nodes.get(n)
        if not node or node.kind != "file":
            raise FileNotFoundError(f"No such file: {path}")
        if node.raw_bytes is not None:
            return node.raw_bytes
        return (node.content or "").encode("utf-8")

    async def write_file(self, path: str, content: str) -> None:
        n = self._normalize(path)
        self._ensure_parent_dirs(n)
        self._nodes[n] = _VFSNode(kind="file", content=content)

    # ── Directory operations ────────────────────────────────────────

    async def read_dir(self, path: str) -> list[DirEntry]:
        n = self._normalize(path)
        node = self._nodes.get(n)
        if not node or node.kind != "dir":
            raise NotADirectoryError(f"Not a directory: {path}")

        prefix = n + "/"
        seen: set[str] = set()
        entries: list[DirEntry] = []

        for key in self._nodes:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            name = rest.split("/")[0]
            if name in seen:
                continue
            seen.add(name)

            child_path = prefix + name
            child = self._nodes.get(child_path)
            entries.append(DirEntry(
                name=name,
                is_directory=(child is not None and child.kind == "dir") or "/" in rest,
                is_file=child is not None and child.kind == "file" and "/" not in rest,
            ))

        return sorted(entries, key=lambda e: e.name)

    async def mkdir(self, path: str, recursive: bool = False) -> None:
        n = self._normalize(path)
        if n in self._nodes:
            return
        if recursive:
            self._ensure_parent_dirs(n + "/dummy")
            self._nodes[n] = _VFSNode(kind="dir")
        else:
            parent = self.dirname(n)
            pnode = self._nodes.get(parent)
            if not pnode or pnode.kind != "dir":
                raise FileNotFoundError(f"Parent not found: {parent}")
            self._nodes[n] = _VFSNode(kind="dir")

    # ── Metadata ────────────────────────────────────────────────────

    async def exists(self, path: str) -> bool:
        n = self._normalize(path)
        if n in self._nodes:
            return True
        prefix = n + "/"
        return any(k.startswith(prefix) for k in self._nodes)

    async def stat(self, path: str) -> FileStat:
        n = self._normalize(path)
        node = self._nodes.get(n)
        if node:
            size = len(node.content or "") if node.content else (len(node.raw_bytes) if node.raw_bytes else 0)
            return FileStat(
                is_file=node.kind == "file",
                is_directory=node.kind == "dir",
                size=size,
                mtime_ms=node.mtime_ms,
            )
        prefix = n + "/"
        if any(k.startswith(prefix) for k in self._nodes):
            return FileStat(is_file=False, is_directory=True, size=0, mtime_ms=_now_ms())
        raise FileNotFoundError(f"No such file or directory: {path}")

    # ── Command execution ───────────────────────────────────────────

    async def exec(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        result = self._exec_stub(command, timeout=timeout, cwd=cwd, env=env)
        if hasattr(result, "__await__"):
            return await result
        return result  # type: ignore[return-value]

    # ── Internal ────────────────────────────────────────────────────

    def _normalize(self, p: str) -> str:
        parts = p.replace("\\", "/").split("/")
        stack: list[str] = []
        for part in parts:
            if part == "..":
                if stack:
                    stack.pop()
            elif part and part != ".":
                stack.append(part)
        return "/" + "/".join(stack)

    def _ensure_parent_dirs(self, abs_path: str) -> None:
        parts = abs_path.split("/")
        current = ""
        for part in parts[:-1]:
            if not part:
                continue
            current += "/" + part
            if current not in self._nodes:
                self._nodes[current] = _VFSNode(kind="dir")
