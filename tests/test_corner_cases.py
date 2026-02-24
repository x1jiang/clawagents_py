"""
Corner case and edge case tests for ClawAgents.

Pushes the boundaries: unicode, special chars, large files, empty inputs,
boundary conditions, whitespace handling, error recovery, concurrent hooks,
symlinks, binary content, etc.

Run: python -m pytest tests/test_corner_cases.py -v
"""

import asyncio
import json
import os
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock

from clawagents.tools.registry import ToolResult


# ─── Filesystem: Unicode / Special Characters ─────────────────────────────

class TestUnicodeContent:
    """Ensure tools handle unicode correctly."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_write_and_read_unicode(self):
        from clawagents.tools.filesystem import WriteFileTool, ReadFileTool
        path = os.path.join(self.tmpdir, "unicode.txt")

        content = "Hello 世界! 🎉 Ñoño café résumé über naïve"
        w = await WriteFileTool().execute({"path": path, "content": content})
        assert w.success is True

        r = await ReadFileTool().execute({"path": path})
        assert r.success is True
        assert "世界" in r.output
        assert "🎉" in r.output
        assert "résumé" in r.output

    @pytest.mark.asyncio
    async def test_edit_unicode_target(self):
        from clawagents.tools.filesystem import WriteFileTool, EditFileTool
        path = os.path.join(self.tmpdir, "edit_unicode.txt")

        await WriteFileTool().execute({"path": path, "content": "Hello 世界"})
        result = await EditFileTool().execute({
            "path": path,
            "target": "世界",
            "replacement": "World 🌍"
        })
        assert result.success is True

        with open(path, encoding="utf-8") as f:
            assert "World 🌍" in f.read()

    @pytest.mark.asyncio
    async def test_grep_unicode_pattern(self):
        from clawagents.tools.filesystem import WriteFileTool, GrepTool
        path = os.path.join(self.tmpdir, "grep_unicode.txt")

        await WriteFileTool().execute({"path": path, "content": "Line 1\n函数定义\nLine 3\n"})
        result = await GrepTool().execute({"path": path, "pattern": "函数"})

        assert result.success is True
        assert "函数" in result.output


# ─── Filesystem: Special Filenames ────────────────────────────────────────

class TestSpecialFilenames:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_file_with_spaces(self):
        from clawagents.tools.filesystem import WriteFileTool, ReadFileTool
        path = os.path.join(self.tmpdir, "file with spaces.txt")

        w = await WriteFileTool().execute({"path": path, "content": "spaced"})
        assert w.success is True

        r = await ReadFileTool().execute({"path": path})
        assert r.success is True
        assert "spaced" in r.output

    @pytest.mark.asyncio
    async def test_dotfile(self):
        from clawagents.tools.filesystem import WriteFileTool, LsTool
        path = os.path.join(self.tmpdir, ".hidden")

        await WriteFileTool().execute({"path": path, "content": "secret"})
        result = await LsTool().execute({"path": self.tmpdir})

        assert result.success is True
        assert ".hidden" in result.output

    @pytest.mark.asyncio
    async def test_deeply_nested_path(self):
        from clawagents.tools.filesystem import WriteFileTool, ReadFileTool
        path = os.path.join(self.tmpdir, "a", "b", "c", "d", "e", "f", "deep.txt")

        w = await WriteFileTool().execute({"path": path, "content": "deep!"})
        assert w.success is True

        r = await ReadFileTool().execute({"path": path})
        assert r.success is True
        assert "deep!" in r.output


# ─── Filesystem: Large Files / Boundary Conditions ────────────────────────

class TestBoundaryConditions:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_empty_file_read(self):
        from clawagents.tools.filesystem import ReadFileTool
        path = os.path.join(self.tmpdir, "empty.txt")
        with open(path, "w") as f:
            pass  # empty file

        result = await ReadFileTool().execute({"path": path})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_single_line_file(self):
        from clawagents.tools.filesystem import ReadFileTool
        path = os.path.join(self.tmpdir, "one.txt")
        with open(path, "w") as f:
            f.write("only line")

        result = await ReadFileTool().execute({"path": path})
        assert result.success is True
        assert "only line" in result.output

    @pytest.mark.asyncio
    async def test_large_file(self):
        from clawagents.tools.filesystem import ReadFileTool
        path = os.path.join(self.tmpdir, "large.txt")
        with open(path, "w") as f:
            for i in range(10000):
                f.write(f"Line {i}: {'x' * 80}\n")

        result = await ReadFileTool().execute({"path": path, "limit": 10})
        assert result.success is True
        assert "10000 lines total" in result.output

    @pytest.mark.asyncio
    async def test_write_empty_content(self):
        from clawagents.tools.filesystem import WriteFileTool
        path = os.path.join(self.tmpdir, "empty_write.txt")

        result = await WriteFileTool().execute({"path": path, "content": ""})
        assert result.success is True
        assert os.path.getsize(path) == 0

    @pytest.mark.asyncio
    async def test_edit_whitespace_only_target(self):
        from clawagents.tools.filesystem import EditFileTool
        path = os.path.join(self.tmpdir, "ws.txt")
        with open(path, "w") as f:
            f.write("before\n    \nafter\n")

        result = await EditFileTool().execute({
            "path": path,
            "target": "    \n",
            "replacement": "REPLACED\n"
        })
        assert result.success is True

    @pytest.mark.asyncio
    async def test_edit_multiline_target(self):
        from clawagents.tools.filesystem import EditFileTool
        path = os.path.join(self.tmpdir, "multi.txt")
        with open(path, "w") as f:
            f.write("line1\nline2\nline3\n")

        result = await EditFileTool().execute({
            "path": path,
            "target": "line1\nline2",
            "replacement": "REPLACED"
        })
        assert result.success is True
        with open(path) as f:
            content = f.read()
        assert "REPLACED\nline3" in content

    @pytest.mark.asyncio
    async def test_read_offset_beyond_file(self):
        from clawagents.tools.filesystem import ReadFileTool
        path = os.path.join(self.tmpdir, "short.txt")
        with open(path, "w") as f:
            f.write("only 2 lines\nhere\n")

        result = await ReadFileTool().execute({"path": path, "offset": 100, "limit": 10})
        assert result.success is True
        # Should return empty slice gracefully

    @pytest.mark.asyncio
    async def test_grep_empty_pattern(self):
        from clawagents.tools.filesystem import GrepTool
        result = await GrepTool().execute({"path": self.tmpdir, "pattern": ""})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_glob_empty_pattern(self):
        from clawagents.tools.filesystem import GlobTool
        result = await GlobTool().execute({"pattern": "", "path": self.tmpdir})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_ls_file_not_dir(self):
        from clawagents.tools.filesystem import LsTool
        path = os.path.join(self.tmpdir, "notadir.txt")
        with open(path, "w") as f:
            f.write("x")

        result = await LsTool().execute({"path": path})
        assert result.success is False


# ─── Grep: Pattern Edge Cases ────────────────────────────────────────────

class TestGrepEdgeCases:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_grep_special_regex_chars(self):
        """Grep uses literal matching, not regex — special chars should work."""
        from clawagents.tools.filesystem import GrepTool
        path = os.path.join(self.tmpdir, "special.txt")
        with open(path, "w") as f:
            f.write("price is $100.00\nfoo(bar)\n[brackets]\n")

        for pattern in ["$100.00", "foo(bar)", "[brackets]"]:
            result = await GrepTool().execute({"path": path, "pattern": pattern})
            assert result.success is True
            assert pattern in result.output, f"Failed for pattern: {pattern}"

    @pytest.mark.asyncio
    async def test_grep_case_sensitive(self):
        from clawagents.tools.filesystem import GrepTool
        path = os.path.join(self.tmpdir, "case.txt")
        with open(path, "w") as f:
            f.write("Hello World\nhello world\nHELLO WORLD\n")

        result = await GrepTool().execute({"path": path, "pattern": "Hello"})
        assert result.success is True
        assert "1 match" in result.output  # Only first line


# ─── Edit: Replace All Corner Cases ──────────────────────────────────────

class TestEditReplaceAllCornerCases:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_replace_all_consecutive(self):
        from clawagents.tools.filesystem import EditFileTool
        path = os.path.join(self.tmpdir, "consec.txt")
        with open(path, "w") as f:
            f.write("aaa")  # 3 consecutive 'a's — "a" appears 3 times

        result = await EditFileTool().execute({
            "path": path,
            "target": "a",
            "replacement": "bb",
            "replace_all": True
        })
        assert result.success is True
        with open(path) as f:
            assert f.read() == "bbbbbb"

    @pytest.mark.asyncio
    async def test_replace_with_empty(self):
        from clawagents.tools.filesystem import EditFileTool
        path = os.path.join(self.tmpdir, "delete.txt")
        with open(path, "w") as f:
            f.write("keep DELETE keep")

        result = await EditFileTool().execute({
            "path": path,
            "target": " DELETE ",
            "replacement": " "
        })
        assert result.success is True
        with open(path) as f:
            assert f.read() == "keep keep"

    @pytest.mark.asyncio
    async def test_replace_target_is_replacement(self):
        """Replace X with X — no-op but should succeed."""
        from clawagents.tools.filesystem import EditFileTool
        path = os.path.join(self.tmpdir, "noop.txt")
        with open(path, "w") as f:
            f.write("same same same")

        result = await EditFileTool().execute({
            "path": path,
            "target": "same",
            "replacement": "same",
            "replace_all": True
        })
        assert result.success is True
        with open(path) as f:
            assert f.read() == "same same same"


# ─── TodoList: Corner Cases ──────────────────────────────────────────────

class TestTodoListCornerCases:

    def setup_method(self):
        from clawagents.tools.todolist import reset_todos
        reset_todos()

    @pytest.mark.asyncio
    async def test_large_todo_list(self):
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        items = [f"Step {i}" for i in range(100)]

        result = await WriteTodosTool().execute({"todos": items})
        assert result.success is True
        assert "0/100" in result.output

        # Complete last one
        result = await UpdateTodoTool().execute({"index": 99})
        assert result.success is True
        assert "1/100" in result.output

    @pytest.mark.asyncio
    async def test_double_complete(self):
        """Completing an already-done item should still succeed."""
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        await WriteTodosTool().execute({"todos": ["Only task"]})
        await UpdateTodoTool().execute({"index": 0})
        result = await UpdateTodoTool().execute({"index": 0})

        assert result.success is True
        assert "[x]" in result.output

    @pytest.mark.asyncio
    async def test_empty_todo_list(self):
        from clawagents.tools.todolist import WriteTodosTool
        result = await WriteTodosTool().execute({"todos": []})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_todos_with_special_chars(self):
        from clawagents.tools.todolist import WriteTodosTool
        result = await WriteTodosTool().execute({
            "todos": ["Fix bug #123", "Handle edge case: 'quotes'", "Parse <xml> & entities"]
        })
        assert result.success is True
        assert "#123" in result.output
        assert "quotes" in result.output
        assert "<xml>" in result.output

    @pytest.mark.asyncio
    async def test_update_negative_index(self):
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        await WriteTodosTool().execute({"todos": ["A"]})
        result = await UpdateTodoTool().execute({"index": -1})
        assert result.success is False


# ─── Hook: Chaining and Override Edge Cases ───────────────────────────────

class TestHookChaining:

    def test_inject_context_then_block_tools(self):
        """Both hooks should work independently."""
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.inject_context("Be brief")
        agent.block_tools("execute")

        # Context should be injected
        msgs = [{"role": "system", "content": "base"}]
        result = agent.before_llm(msgs)
        assert "Be brief" in result[-1]["content"]

        # Execute should be blocked
        assert agent.before_tool("execute", {}) is False
        assert agent.before_tool("ls", {}) is True

    def test_truncate_preserves_error(self):
        """Truncation should preserve the error field."""
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.truncate_output(10)

        result = agent.after_tool(
            "read_file", {},
            ToolResult(success=False, output="x" * 100, error="some error")
        )
        assert result.error == "some error"
        assert result.success is False
        assert len(result.output) < 100

    def test_allow_only_tools_overrides_block(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.block_tools("execute")
        agent.allow_only_tools("execute")  # Override: now only execute allowed

        assert agent.before_tool("execute", {}) is True
        assert agent.before_tool("ls", {}) is False

    def test_multiple_inject_context_preserves_order(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.inject_context("First")
        agent.inject_context("Second")
        agent.inject_context("Third")

        msgs = [{"role": "user", "content": "hi"}]
        result = agent.before_llm(msgs)

        # All 3 injected in order
        assert len(result) == 4
        assert "First" in result[1]["content"]
        assert "Second" in result[2]["content"]
        assert "Third" in result[3]["content"]


# ─── Memory: Corner Cases ────────────────────────────────────────────────

class TestMemoryCornerCases:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_binary_content_skipped(self):
        from clawagents.memory.loader import load_memory_files
        path = os.path.join(self.tmpdir, "binary.md")
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02\x03\xff\xfe")

        # Should not crash
        result = load_memory_files([path])
        # May return content or None — just should not crash

    def test_very_large_memory_file(self):
        from clawagents.memory.loader import load_memory_files
        path = os.path.join(self.tmpdir, "big.md")
        with open(path, "w") as f:
            f.write("x" * 100_000)

        result = load_memory_files([path])
        assert result is not None
        assert len(result) > 100_000

    def test_whitespace_only_file(self):
        from clawagents.memory.loader import load_memory_files
        path = os.path.join(self.tmpdir, "ws.md")
        with open(path, "w") as f:
            f.write("   \n\n   \t\t\n  ")

        result = load_memory_files([path])
        # Whitespace-only should be treated as empty
        assert result is None

    def test_mixed_existing_and_missing(self):
        from clawagents.memory.loader import load_memory_files
        path = os.path.join(self.tmpdir, "exists.md")
        with open(path, "w") as f:
            f.write("Real content")

        result = load_memory_files([
            "/nonexistent/file.md",
            path,
            "/also/missing.md",
        ])
        assert result is not None
        assert "Real content" in result


# ─── Auto-Discovery: Corner Cases ────────────────────────────────────────

class TestAutoDiscoveryCornerCases:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmpdir)

    def test_agents_md_is_directory_not_file(self):
        """AGENTS.md as a directory should NOT be discovered as memory."""
        from clawagents.agent import _auto_discover_memory
        os.makedirs("AGENTS.md")  # It's a dir, not a file!
        found = _auto_discover_memory()
        assert len(found) == 0

    def test_skills_is_file_not_directory(self):
        """'skills' as a file should NOT be discovered as skill dir."""
        from clawagents.agent import _auto_discover_skills
        with open("skills", "w") as f:
            f.write("I'm a file, not a dir")
        found = _auto_discover_skills()
        assert len(found) == 0


# ─── Tool Registry: Corner Cases ─────────────────────────────────────────

class TestToolRegistryCornerCases:

    def test_register_duplicate_name(self):
        from clawagents.tools.registry import ToolRegistry

        class DummyTool:
            name = "test"
            description = "v1"
            parameters = {}
            async def execute(self, args):
                return ToolResult(success=True, output="v1")

        class DummyTool2:
            name = "test"
            description = "v2"
            parameters = {}
            async def execute(self, args):
                return ToolResult(success=True, output="v2")

        registry = ToolRegistry()
        registry.register(DummyTool())
        registry.register(DummyTool2())

        # Later registration should win
        tools = registry.list()
        test_tools = [t for t in tools if t.name == "test"]
        assert len(test_tools) >= 1

    @pytest.mark.asyncio
    async def test_tool_returns_none(self):
        """Tool that returns None instead of ToolResult."""
        from clawagents.tools.registry import ToolRegistry

        class BadTool:
            name = "bad"
            description = "returns None"
            parameters = {}
            async def execute(self, args):
                return None

        registry = ToolRegistry()
        registry.register(BadTool())
        result = await registry.execute_tool("bad", {})
        # Should handle gracefully (may wrap in error or succeed)

    @pytest.mark.asyncio
    async def test_tool_with_huge_output(self):
        """Tool that returns very large output."""
        from clawagents.tools.registry import ToolRegistry

        class BigTool:
            name = "big"
            description = "big output"
            parameters = {}
            async def execute(self, args):
                return ToolResult(success=True, output="x" * 1_000_000)

        registry = ToolRegistry()
        registry.register(BigTool())
        result = await registry.execute_tool("big", {})
        assert result.success is True
        # Registry may truncate very large outputs — just verify it succeeded
        assert len(result.output) > 0
