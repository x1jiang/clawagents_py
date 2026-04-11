"""Cross-provider conformance test suite.

Shared tests that all SandboxBackend implementations must pass.
"""
import pytest
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


class TestInMemoryBackendConformance(BackendConformanceSuite):
    def get_backend(self) -> SandboxBackend:
        return InMemoryBackend()
