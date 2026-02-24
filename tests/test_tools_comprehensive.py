"""
Comprehensive tests for ClawAgents tool implementations.

Exercises actual tool execution on real filesystem with temp directories:
  - LsTool: metadata, sorting, empty dirs, nonexistent paths
  - ReadFileTool: pagination, offset/limit, missing files
  - WriteFileTool: create, overwrite, nested dirs
  - EditFileTool: replace, replace_all, missing target, non-unique target
  - GrepTool: single file, recursive, glob filter, no matches
  - GlobTool: patterns, recursive, empty results
  - TodoList: write, update, edge cases, reset
  - Memory loader: loading, missing files, empty files, tags

Run: python -m pytest tests/test_tools_comprehensive.py -v
"""

import asyncio
import os
import pytest
import tempfile
import shutil

# ─── Filesystem Tools ─────────────────────────────────────────────────────

class TestLsTool:
    """Test ls tool with real filesystem."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create some files and dirs
        os.makedirs(os.path.join(self.tmpdir, "subdir"))
        with open(os.path.join(self.tmpdir, "file_a.txt"), "w") as f:
            f.write("hello")
        with open(os.path.join(self.tmpdir, "file_b.py"), "w") as f:
            f.write("print('hi')" * 100)  # larger file

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_ls_with_metadata(self):
        from clawagents.tools.filesystem import LsTool
        tool = LsTool()
        result = await tool.execute({"path": self.tmpdir})

        assert result.success is True
        assert "[DIR]" in result.output
        assert "subdir/" in result.output
        assert "[FILE]" in result.output
        assert "file_a.txt" in result.output

    @pytest.mark.asyncio
    async def test_ls_dirs_first(self):
        from clawagents.tools.filesystem import LsTool
        tool = LsTool()
        result = await tool.execute({"path": self.tmpdir})
        lines = result.output.strip().split("\n")

        # First line should be the [DIR], files after
        assert "[DIR]" in lines[0]

    @pytest.mark.asyncio
    async def test_ls_empty_dir(self):
        from clawagents.tools.filesystem import LsTool
        tool = LsTool()
        empty = os.path.join(self.tmpdir, "subdir")
        result = await tool.execute({"path": empty})

        assert result.success is True
        assert "(empty directory)" in result.output

    @pytest.mark.asyncio
    async def test_ls_nonexistent(self):
        from clawagents.tools.filesystem import LsTool
        tool = LsTool()
        result = await tool.execute({"path": "/nonexistent/path"})

        assert result.success is False
        assert "failed" in result.error.lower() or "not a directory" in result.error.lower()


class TestReadFileTool:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test.txt")
        with open(self.filepath, "w") as f:
            for i in range(50):
                f.write(f"Line {i+1}: content here\n")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_read_file_basic(self):
        from clawagents.tools.filesystem import ReadFileTool
        tool = ReadFileTool()
        result = await tool.execute({"path": self.filepath})

        assert result.success is True
        assert "Line 1" in result.output
        assert "50 lines total" in result.output

    @pytest.mark.asyncio
    async def test_read_file_pagination(self):
        from clawagents.tools.filesystem import ReadFileTool
        tool = ReadFileTool()
        result = await tool.execute({"path": self.filepath, "offset": 10, "limit": 5})

        assert result.success is True
        assert "Line 11" in result.output
        # Should only show 5 lines
        assert "showing 11-15" in result.output

    @pytest.mark.asyncio
    async def test_read_missing_file(self):
        from clawagents.tools.filesystem import ReadFileTool
        tool = ReadFileTool()
        result = await tool.execute({"path": "/nonexistent/file.txt"})

        assert result.success is False


class TestWriteFileTool:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_write_creates_file(self):
        from clawagents.tools.filesystem import WriteFileTool
        tool = WriteFileTool()
        path = os.path.join(self.tmpdir, "new.txt")
        result = await tool.execute({"path": path, "content": "Hello world!"})

        assert result.success is True
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "Hello world!"

    @pytest.mark.asyncio
    async def test_write_creates_nested_dirs(self):
        from clawagents.tools.filesystem import WriteFileTool
        tool = WriteFileTool()
        path = os.path.join(self.tmpdir, "a", "b", "c", "deep.txt")
        result = await tool.execute({"path": path, "content": "nested!"})

        assert result.success is True
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_write_overwrites(self):
        from clawagents.tools.filesystem import WriteFileTool
        tool = WriteFileTool()
        path = os.path.join(self.tmpdir, "overwrite.txt")

        await tool.execute({"path": path, "content": "first"})
        await tool.execute({"path": path, "content": "second"})

        with open(path) as f:
            assert f.read() == "second"


class TestEditFileTool:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "edit_me.txt")
        with open(self.filepath, "w") as f:
            f.write("Hello World\nFoo Bar\nHello World\nBaz Qux\n")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_edit_single_replace(self):
        from clawagents.tools.filesystem import EditFileTool
        tool = EditFileTool()
        result = await tool.execute({
            "path": self.filepath,
            "target": "Foo Bar",
            "replacement": "FOO BAR REPLACED"
        })

        assert result.success is True
        with open(self.filepath) as f:
            content = f.read()
        assert "FOO BAR REPLACED" in content
        assert "Foo Bar" not in content

    @pytest.mark.asyncio
    async def test_edit_fails_on_non_unique(self):
        from clawagents.tools.filesystem import EditFileTool
        tool = EditFileTool()
        result = await tool.execute({
            "path": self.filepath,
            "target": "Hello World",
            "replacement": "Hi"
        })

        # Should fail because "Hello World" appears twice
        assert result.success is False
        assert "2 times" in result.error

    @pytest.mark.asyncio
    async def test_edit_replace_all(self):
        from clawagents.tools.filesystem import EditFileTool
        tool = EditFileTool()
        result = await tool.execute({
            "path": self.filepath,
            "target": "Hello World",
            "replacement": "Replaced!",
            "replace_all": True
        })

        assert result.success is True
        with open(self.filepath) as f:
            content = f.read()
        assert content.count("Replaced!") == 2
        assert "Hello World" not in content

    @pytest.mark.asyncio
    async def test_edit_missing_target(self):
        from clawagents.tools.filesystem import EditFileTool
        tool = EditFileTool()
        result = await tool.execute({
            "path": self.filepath,
            "target": "NONEXISTENT TEXT",
            "replacement": "Hi"
        })

        assert result.success is False
        assert "not find" in result.error.lower() or "could not" in result.error.lower()

    @pytest.mark.asyncio
    async def test_edit_missing_file(self):
        from clawagents.tools.filesystem import EditFileTool
        tool = EditFileTool()
        result = await tool.execute({
            "path": "/nonexistent/file.txt",
            "target": "x",
            "replacement": "y"
        })

        assert result.success is False


class TestGrepTool:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create file structure for grep
        os.makedirs(os.path.join(self.tmpdir, "src"))
        with open(os.path.join(self.tmpdir, "README.md"), "w") as f:
            f.write("# Project\nThis is a TODO marker\nAnother line\n")
        with open(os.path.join(self.tmpdir, "src", "main.py"), "w") as f:
            f.write("def main():\n    # TODO: implement\n    pass\n")
        with open(os.path.join(self.tmpdir, "src", "utils.py"), "w") as f:
            f.write("def helper():\n    return 42\n")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_grep_single_file(self):
        from clawagents.tools.filesystem import GrepTool
        tool = GrepTool()
        result = await tool.execute({
            "path": os.path.join(self.tmpdir, "README.md"),
            "pattern": "TODO"
        })

        assert result.success is True
        assert "TODO" in result.output
        assert "1 match" in result.output

    @pytest.mark.asyncio
    async def test_grep_recursive(self):
        from clawagents.tools.filesystem import GrepTool
        tool = GrepTool()
        result = await tool.execute({
            "path": self.tmpdir,
            "pattern": "TODO",
            "recursive": True
        })

        assert result.success is True
        # Should find in both README.md and src/main.py
        assert "2 match" in result.output

    @pytest.mark.asyncio
    async def test_grep_with_glob_filter(self):
        from clawagents.tools.filesystem import GrepTool
        tool = GrepTool()
        result = await tool.execute({
            "path": self.tmpdir,
            "pattern": "TODO",
            "glob_filter": "*.py",
            "recursive": True
        })

        assert result.success is True
        assert "main.py" in result.output
        # Should NOT include README.md
        assert "README" not in result.output

    @pytest.mark.asyncio
    async def test_grep_no_matches(self):
        from clawagents.tools.filesystem import GrepTool
        tool = GrepTool()
        result = await tool.execute({
            "path": self.tmpdir,
            "pattern": "ZZZZNONEXISTENT",
            "recursive": True
        })

        assert result.success is True
        assert "No matches" in result.output


class TestGlobTool:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, "src", "deep"))
        with open(os.path.join(self.tmpdir, "README.md"), "w") as f:
            f.write("readme")
        with open(os.path.join(self.tmpdir, "src", "main.py"), "w") as f:
            f.write("main")
        with open(os.path.join(self.tmpdir, "src", "deep", "nested.py"), "w") as f:
            f.write("nested")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_glob_py_files(self):
        from clawagents.tools.filesystem import GlobTool
        tool = GlobTool()
        result = await tool.execute({
            "pattern": "**/*.py",
            "path": self.tmpdir
        })

        assert result.success is True
        assert "main.py" in result.output
        assert "nested.py" in result.output

    @pytest.mark.asyncio
    async def test_glob_md_files(self):
        from clawagents.tools.filesystem import GlobTool
        tool = GlobTool()
        result = await tool.execute({
            "pattern": "*.md",
            "path": self.tmpdir
        })

        assert result.success is True
        assert "README.md" in result.output

    @pytest.mark.asyncio
    async def test_glob_no_matches(self):
        from clawagents.tools.filesystem import GlobTool
        tool = GlobTool()
        result = await tool.execute({
            "pattern": "**/*.xyz",
            "path": self.tmpdir
        })

        assert result.success is True
        assert "No files" in result.output


# ─── TodoList Tools ────────────────────────────────────────────────────────

class TestTodoList:

    def setup_method(self):
        from clawagents.tools.todolist import reset_todos
        reset_todos()

    @pytest.mark.asyncio
    async def test_write_todos(self):
        from clawagents.tools.todolist import WriteTodosTool
        tool = WriteTodosTool()
        result = await tool.execute({
            "todos": ["Read the code", "Find the bug", "Fix it"]
        })

        assert result.success is True
        assert "[ ] Read the code" in result.output
        assert "[ ] Find the bug" in result.output
        assert "[ ] Fix it" in result.output
        assert "0/3" in result.output

    @pytest.mark.asyncio
    async def test_write_todos_from_json_string(self):
        from clawagents.tools.todolist import WriteTodosTool
        tool = WriteTodosTool()
        result = await tool.execute({
            "todos": '["Step 1", "Step 2"]'
        })

        assert result.success is True
        assert "Step 1" in result.output
        assert "Step 2" in result.output

    @pytest.mark.asyncio
    async def test_update_todo(self):
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        write = WriteTodosTool()
        update = UpdateTodoTool()

        await write.execute({"todos": ["A", "B", "C"]})
        result = await update.execute({"index": 1})

        assert result.success is True
        assert "[x] B" in result.output
        assert "1/3" in result.output

    @pytest.mark.asyncio
    async def test_update_all_todos(self):
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        write = WriteTodosTool()
        update = UpdateTodoTool()

        await write.execute({"todos": ["A", "B"]})
        await update.execute({"index": 0})
        result = await update.execute({"index": 1})

        assert result.success is True
        assert "2/2" in result.output
        assert "[x] A" in result.output
        assert "[x] B" in result.output

    @pytest.mark.asyncio
    async def test_update_out_of_range(self):
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool
        write = WriteTodosTool()
        update = UpdateTodoTool()

        await write.execute({"todos": ["A"]})
        result = await update.execute({"index": 99})

        assert result.success is False
        assert "out of range" in result.error.lower()

    @pytest.mark.asyncio
    async def test_update_no_todos(self):
        from clawagents.tools.todolist import UpdateTodoTool
        tool = UpdateTodoTool()
        result = await tool.execute({"index": 0})

        assert result.success is False
        assert "No todo list" in result.error

    @pytest.mark.asyncio
    async def test_write_invalid_json(self):
        from clawagents.tools.todolist import WriteTodosTool
        tool = WriteTodosTool()
        result = await tool.execute({"todos": "not valid json"})

        assert result.success is False

    @pytest.mark.asyncio
    async def test_write_replaces_existing(self):
        from clawagents.tools.todolist import WriteTodosTool
        tool = WriteTodosTool()

        await tool.execute({"todos": ["Old 1", "Old 2"]})
        result = await tool.execute({"todos": ["New 1"]})

        assert result.success is True
        assert "New 1" in result.output
        assert "Old 1" not in result.output


# ─── Memory Loader ────────────────────────────────────────────────────────

class TestMemoryLoader:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_loads_single_file(self):
        from clawagents.memory.loader import load_memory_files

        path = os.path.join(self.tmpdir, "AGENTS.md")
        with open(path, "w") as f:
            f.write("# Project Rules\n- Use async/await\n- Type all functions")

        result = load_memory_files([path])

        assert result is not None
        assert "Agent Memory" in result
        assert "agent_memory" in result
        assert "Use async/await" in result
        assert 'source="AGENTS.md"' in result

    def test_loads_multiple_files(self):
        from clawagents.memory.loader import load_memory_files

        path1 = os.path.join(self.tmpdir, "AGENTS.md")
        path2 = os.path.join(self.tmpdir, "CLAWAGENTS.md")
        with open(path1, "w") as f:
            f.write("Rule 1")
        with open(path2, "w") as f:
            f.write("Rule 2")

        result = load_memory_files([path1, path2])

        assert result is not None
        assert "Rule 1" in result
        assert "Rule 2" in result

    def test_returns_none_for_no_files(self):
        from clawagents.memory.loader import load_memory_files
        result = load_memory_files([])
        assert result is None

    def test_returns_none_for_missing_files(self):
        from clawagents.memory.loader import load_memory_files
        result = load_memory_files(["/nonexistent/AGENTS.md"])
        assert result is None

    def test_skips_empty_files(self):
        from clawagents.memory.loader import load_memory_files

        path = os.path.join(self.tmpdir, "AGENTS.md")
        with open(path, "w") as f:
            f.write("")  # empty

        result = load_memory_files([path])
        assert result is None

    def test_tags_include_filename(self):
        from clawagents.memory.loader import load_memory_files

        path = os.path.join(self.tmpdir, "MY_MEMORY.md")
        with open(path, "w") as f:
            f.write("custom memory content")

        result = load_memory_files([path])

        assert result is not None
        assert 'source="MY_MEMORY.md"' in result


# ─── Hook Composition (_compose_before_llm) ───────────────────────────────

class TestComposeBeforeLLM:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_with_memory_only(self):
        from clawagents.agent import _compose_before_llm

        path = os.path.join(self.tmpdir, "AGENTS.md")
        with open(path, "w") as f:
            f.write("Always use TypeScript")

        hook = _compose_before_llm(memory_paths=[path], skill_summaries=None)

        assert hook is not None
        messages = [{"role": "system", "content": "You are helpful."}]
        result = hook(messages)

        assert "Always use TypeScript" in result[0]["content"]

    def test_with_skills_only(self):
        from clawagents.agent import _compose_before_llm

        hook = _compose_before_llm(
            memory_paths=[],
            skill_summaries="## Available Skills\n- **code_review**: Reviews code"
        )

        assert hook is not None
        messages = [{"role": "system", "content": "Base prompt."}]
        result = hook(messages)

        assert "code_review" in result[0]["content"]

    def test_with_both(self):
        from clawagents.agent import _compose_before_llm

        path = os.path.join(self.tmpdir, "AGENTS.md")
        with open(path, "w") as f:
            f.write("Project rules here")

        hook = _compose_before_llm(
            memory_paths=[path],
            skill_summaries="## Skills\n- **test**: Test skill"
        )

        messages = [{"role": "system", "content": "Base."}]
        result = hook(messages)

        assert "Project rules" in result[0]["content"]
        assert "test" in result[0]["content"]

    def test_with_neither(self):
        from clawagents.agent import _compose_before_llm

        hook = _compose_before_llm(memory_paths=[], skill_summaries=None)
        assert hook is None


# ─── Edge Cases ────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_hook_override(self):
        """Setting a convenience hook then overriding with raw hook."""
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.block_tools("execute")
        assert agent.before_tool("execute", {}) is False

        # Override with raw hook
        agent.before_tool = lambda name, args: True
        assert agent.before_tool("execute", {}) is True

    def test_multiple_hook_changes(self):
        """Changing hooks multiple times should use the latest."""
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())

        agent.block_tools("execute")
        assert agent.before_tool("execute", {}) is False

        agent.allow_only_tools("execute")
        assert agent.before_tool("execute", {}) is True
        assert agent.before_tool("ls", {}) is False

    def test_empty_instruction(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry(), system_prompt="")
        assert agent.system_prompt == ""

    def test_none_instruction(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        assert agent.system_prompt is None


from unittest.mock import MagicMock
