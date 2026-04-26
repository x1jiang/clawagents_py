"""Cross-provider conformance test suite.

Shared tests that all SandboxBackend implementations must pass.
"""
import pytest
import threading
from pathlib import Path
from clawagents.sandbox.backend import SandboxBackend
from clawagents.sandbox.local import LocalBackend
from clawagents.sandbox.memory import InMemoryBackend


class BackendConformanceSuite:
    """Base test class — subclass per backend."""

    def get_backend(self) -> SandboxBackend:
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_write_and_read(self):
        backend = self.get_backend()
        await backend.write_file(backend.resolve("test.txt"), "hello")
        content = await backend.read_file(backend.resolve("test.txt"))
        assert content == "hello"

    @pytest.mark.asyncio
    async def test_mkdir_and_ls(self):
        backend = self.get_backend()
        subdir_path = backend.resolve("subdir")
        await backend.mkdir(subdir_path)
        entries = await backend.read_dir(backend.cwd)
        assert any(e.name == "subdir" for e in entries)

    @pytest.mark.asyncio
    async def test_overwrite_file(self):
        backend = self.get_backend()
        path = backend.resolve("overwrite.txt")
        await backend.write_file(path, "first")
        await backend.write_file(path, "second")
        content = await backend.read_file(path)
        assert content == "second"

    @pytest.mark.asyncio
    async def test_file_exists(self):
        backend = self.get_backend()
        path = backend.resolve("exists.txt")
        assert not await backend.exists(path)
        await backend.write_file(path, "data")
        assert await backend.exists(path)

    @pytest.mark.asyncio
    async def test_stat_file(self):
        backend = self.get_backend()
        path = backend.resolve("stat.txt")
        await backend.write_file(path, "content")
        info = await backend.stat(path)
        assert info.is_file
        assert not info.is_directory
        assert info.size > 0

    @pytest.mark.asyncio
    async def test_read_missing_file_raises(self):
        backend = self.get_backend()
        path = backend.resolve("no_such_file.txt")
        with pytest.raises((FileNotFoundError, OSError)):
            await backend.read_file(path)

    @pytest.mark.asyncio
    async def test_read_bytes(self):
        backend = self.get_backend()
        path = backend.resolve("bytes.bin")
        await backend.write_file(path, "binary-ish")
        data = await backend.read_file_bytes(path)
        assert isinstance(data, bytes)
        assert len(data) > 0


class TestLocalBackendConformance(BackendConformanceSuite):
    def get_backend(self) -> SandboxBackend:
        import tempfile
        return LocalBackend(root=tempfile.mkdtemp())

    @pytest.mark.asyncio
    async def test_file_reads_run_off_event_loop_thread(self, tmp_path, monkeypatch):
        file_path = tmp_path / "threaded.txt"
        file_path.write_text("hello", encoding="utf-8")
        main_thread = threading.get_ident()
        seen_threads: list[int] = []
        original_read_text = Path.read_text

        def wrapped_read_text(self, *args, **kwargs):
            seen_threads.append(threading.get_ident())
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", wrapped_read_text)
        backend = LocalBackend(root=str(tmp_path))

        assert await backend.read_file(str(file_path)) == "hello"
        assert seen_threads
        assert all(thread_id != main_thread for thread_id in seen_threads)


class TestInMemoryBackendConformance(BackendConformanceSuite):
    def get_backend(self) -> SandboxBackend:
        return InMemoryBackend()
