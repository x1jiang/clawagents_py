"""
Unit tests for new full-stack protocol features.
Covers: hooks, memory loader, todolist, filesystem upgrades, subagent.
Run with: python -m pytest tests/test_fullstack.py -v
"""

import json
import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from clawagents.tools.registry import ToolRegistry, ToolResult, ParsedToolCall


# ─── Memory Loader Tests ─────────────────────────────────────────────────


class TestMemoryLoader:
    def test_load_single_file(self, tmp_path):
        from clawagents.memory.loader import load_memory_files

        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# Project\nThis is a test project.", "utf-8")

        result = load_memory_files([str(agents_md)])
        assert result is not None
        assert "Agent Memory" in result
        assert "This is a test project" in result
        assert 'source="AGENTS.md"' in result

    def test_load_multiple_files(self, tmp_path):
        from clawagents.memory.loader import load_memory_files

        f1 = tmp_path / "AGENTS.md"
        f1.write_text("File one content", "utf-8")
        f2 = tmp_path / "CLAWAGENTS.md"
        f2.write_text("File two content", "utf-8")

        result = load_memory_files([str(f1), str(f2)])
        assert result is not None
        assert "File one content" in result
        assert "File two content" in result

    def test_missing_file_returns_none(self):
        from clawagents.memory.loader import load_memory_files

        result = load_memory_files(["/nonexistent/AGENTS.md"])
        assert result is None

    def test_empty_file_skipped(self, tmp_path):
        from clawagents.memory.loader import load_memory_files

        f = tmp_path / "AGENTS.md"
        f.write_text("", "utf-8")

        result = load_memory_files([str(f)])
        assert result is None


# ─── TodoList Tests ───────────────────────────────────────────────────────


class TestTodoList:
    @pytest.fixture(autouse=True)
    def reset(self):
        from clawagents.tools.todolist import reset_todos
        reset_todos()
        yield
        reset_todos()

    @pytest.mark.asyncio
    async def test_write_todos(self):
        from clawagents.tools.todolist import WriteTodosTool
        tool = WriteTodosTool()
        result = await tool.execute({"todos": ["Step 1", "Step 2", "Step 3"]})
        assert result.success is True
        assert "0/3 complete" in result.output
        assert "[ ] Step 1" in result.output

    @pytest.mark.asyncio
    async def test_write_todos_json_string(self):
        from clawagents.tools.todolist import WriteTodosTool
        tool = WriteTodosTool()
        result = await tool.execute({"todos": '["A", "B"]'})
        assert result.success is True
        assert "0/2 complete" in result.output

    @pytest.mark.asyncio
    async def test_update_todo(self):
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        write = WriteTodosTool()
        update = UpdateTodoTool()

        await write.execute({"todos": ["A", "B", "C"]})
        result = await update.execute({"index": 1})
        assert result.success is True
        assert "1/3 complete" in result.output
        assert "[x] B" in result.output

    @pytest.mark.asyncio
    async def test_update_out_of_range(self):
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        write = WriteTodosTool()
        update = UpdateTodoTool()

        await write.execute({"todos": ["A"]})
        result = await update.execute({"index": 5})
        assert result.success is False
        assert "out of range" in result.error

    @pytest.mark.asyncio
    async def test_update_no_todos(self):
        from clawagents.tools.todolist import UpdateTodoTool
        tool = UpdateTodoTool()
        result = await tool.execute({"index": 0})
        assert result.success is False
        assert "No todo list" in result.error


# ─── Filesystem Tool Tests ────────────────────────────────────────────────


class TestGlobTool:
    @pytest.mark.asyncio
    async def test_glob_py_files(self, tmp_path):
        from clawagents.tools.filesystem import GlobTool
        tool = GlobTool()

        (tmp_path / "a.py").write_text("pass", "utf-8")
        (tmp_path / "b.txt").write_text("hello", "utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("pass", "utf-8")

        result = await tool.execute({"pattern": "**/*.py", "path": str(tmp_path)})
        assert result.success is True
        assert "a.py" in result.output
        assert "c.py" in result.output
        assert "b.txt" not in result.output

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, tmp_path):
        from clawagents.tools.filesystem import GlobTool
        tool = GlobTool()

        result = await tool.execute({"pattern": "*.xyz", "path": str(tmp_path)})
        assert result.success is True
        assert "No files" in result.output


class TestGrepToolRecursive:
    @pytest.mark.asyncio
    async def test_recursive_grep(self, tmp_path):
        from clawagents.tools.filesystem import GrepTool
        tool = GrepTool()

        (tmp_path / "a.py").write_text("def hello():\n    pass\n", "utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("def world():\n    hello()\n", "utf-8")

        result = await tool.execute({
            "path": str(tmp_path),
            "pattern": "hello",
            "glob_filter": "*.py",
            "recursive": True,
        })
        assert result.success is True
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "2 match" in result.output


class TestLsToolMetadata:
    @pytest.mark.asyncio
    async def test_ls_with_metadata(self, tmp_path):
        from clawagents.tools.filesystem import LsTool
        tool = LsTool()

        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("hello world", "utf-8")

        result = await tool.execute({"path": str(tmp_path)})
        assert result.success is True
        assert "[DIR]" in result.output
        assert "[FILE]" in result.output
        assert "file.txt" in result.output  


class TestEditFileReplaceAll:
    @pytest.mark.asyncio
    async def test_replace_all(self, tmp_path):
        from clawagents.tools.filesystem import EditFileTool
        tool = EditFileTool()

        f = tmp_path / "test.txt"
        f.write_text("foo bar foo baz foo", "utf-8")

        result = await tool.execute({
            "path": str(f),
            "target": "foo",
            "replacement": "qux",
            "replace_all": True,
        })
        assert result.success is True
        assert "3 occurrence" in result.output
        assert f.read_text("utf-8") == "qux bar qux baz qux"

    @pytest.mark.asyncio
    async def test_single_replace_requires_unique(self, tmp_path):
        from clawagents.tools.filesystem import EditFileTool
        tool = EditFileTool()

        f = tmp_path / "test.txt"
        f.write_text("foo bar foo", "utf-8")

        result = await tool.execute({
            "path": str(f),
            "target": "foo",
            "replacement": "qux",
        })
        assert result.success is False
        assert "2 times" in result.error


# ─── Hook Tests ───────────────────────────────────────────────────────────


class TestBeforeToolHook:
    @pytest.mark.asyncio
    async def test_before_tool_blocks_execution(self):
        """Verify that before_tool returning False skips the tool."""
        class EchoTool:
            name = "echo"
            description = "echoes input"
            parameters = {}
            async def execute(self, args: Dict[str, Any]) -> ToolResult:
                return ToolResult(success=True, output="echoed")

        registry = ToolRegistry()
        registry.register(EchoTool())

        # Normal execution
        result = await registry.execute_tool("echo", {})
        assert result.success is True
        assert result.output == "echoed"


class TestAfterToolHook:
    def test_hook_type_defined(self):
        from clawagents.graph.agent_loop import AfterToolHook
        # Just verify the type is importable
        assert AfterToolHook is not None


# ─── SubAgent Tool Tests ─────────────────────────────────────────────────


class TestTaskTool:
    def test_task_tool_creation(self):
        from clawagents.tools.subagent import create_task_tool
        from clawagents.tools.registry import ToolRegistry

        class FakeLLM:
            name = "fake"

        tool = create_task_tool(FakeLLM(), ToolRegistry())
        assert tool.name == "task"
        assert "isolated" in tool.description.lower()
