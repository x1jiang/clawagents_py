"""
Sophisticated end-to-end integration tests.

Exercises the FULL agent pipeline without hitting real APIs:
  - Custom tools (calculator, database mock, API client mock)
  - Skill directories with SKILL.md files (frontmatter parsing, sub-directories)
  - AGENTS.md / CLAWAGENTS.md memory injection
  - Multiple hooks wired together (block + inject + truncate)
  - Full factory pipeline (create_claw_agent with all options)
  - Multi-tool execution chains
  - Sub-agent tool creation
  - LangChain-style tool adaptation
  - Compose before_llm with memory + skills
  - Tool registry: parallel execution, describe_for_llm

Run: python -m pytest tests/test_integration_sophisticated.py -v
"""

import asyncio
import json
import os
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List

from clawagents.tools.registry import Tool, ToolResult, ToolRegistry


# ─── Custom Tools ─────────────────────────────────────────────────────────

class CalculatorTool:
    """A custom calculator tool."""
    name = "calculate"
    description = "Evaluate a mathematical expression."
    parameters = {
        "expression": {"type": "string", "description": "Math expression to evaluate", "required": True}
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        expr = str(args.get("expression", ""))
        try:
            # Safe eval of simple math
            result = eval(expr, {"__builtins__": {}}, {"abs": abs, "min": min, "max": max, "round": round})
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Calculator error: {e}")


class DatabaseQueryTool:
    """Mock database query tool."""
    name = "query_db"
    description = "Run a SQL query against the mock database."
    parameters = {
        "sql": {"type": "string", "description": "SQL query to execute", "required": True}
    }

    def __init__(self):
        self.queries_executed: List[str] = []
        self.mock_data = {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"},
                {"id": 3, "name": "Charlie", "role": "user"},
            ],
            "projects": [
                {"id": 1, "name": "ClawAgents", "owner_id": 1},
                {"id": 2, "name": "DeepAgents", "owner_id": 2},
            ],
        }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        sql = str(args.get("sql", "")).lower().strip()
        self.queries_executed.append(sql)

        if "count" in sql:
            return ToolResult(success=True, output="3")
        elif "from users" in sql:
            return ToolResult(success=True, output=json.dumps(self.mock_data["users"], indent=2))
        elif "from projects" in sql:
            return ToolResult(success=True, output=json.dumps(self.mock_data["projects"], indent=2))
        else:
            return ToolResult(success=False, output="", error=f"Unknown table in query: {sql}")


class HttpClientTool:
    """Mock HTTP client tool."""
    name = "http_request"
    description = "Make an HTTP request to an API endpoint."
    parameters = {
        "url": {"type": "string", "description": "URL to request", "required": True},
        "method": {"type": "string", "description": "HTTP method (GET, POST)"},
    }

    def __init__(self):
        self.requests_made: List[dict] = []

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        url = str(args.get("url", ""))
        method = str(args.get("method", "GET")).upper()
        self.requests_made.append({"url": url, "method": method})

        if "api.weather.com" in url:
            return ToolResult(success=True, output=json.dumps({"temp": 72, "condition": "sunny"}))
        elif "api.github.com" in url:
            return ToolResult(success=True, output=json.dumps({"stars": 1500, "forks": 200}))
        else:
            return ToolResult(success=False, output="", error=f"Mock does not support URL: {url}")


class LangChainStyleTool:
    """Simulates a LangChain-compatible tool (has ainvoke, no execute)."""
    name = "langchain_search"
    description = "Search using LangChain-style interface."

    def __init__(self):
        self.input_schema = type("Schema", (), {
            "schema": lambda self: {
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        })()

    async def ainvoke(self, args):
        return f"LangChain result for: {args.get('query', args.get('input', ''))}"


# ─── Skill Directory Setup ────────────────────────────────────────────────

def create_skill_dir(base_dir: str) -> str:
    """Create a realistic skills directory structure."""
    skills_dir = os.path.join(base_dir, "skills")
    os.makedirs(skills_dir)

    # Skill 1: code_review (sub-directory with SKILL.md)
    review_dir = os.path.join(skills_dir, "code_review")
    os.makedirs(review_dir)
    with open(os.path.join(review_dir, "SKILL.md"), "w") as f:
        f.write("""---
name: code_review
description: "Structured code review with severity levels"
---

# Code Review Skill

## Instructions
1. Read the file with `read_file`
2. Analyze for: bugs, style issues, security vulnerabilities
3. Rate each issue: LOW / MEDIUM / HIGH / CRITICAL
4. Suggest fixes with code snippets
5. Summarize findings
""")

    # Skill 2: sql_expert (flat .md file)
    with open(os.path.join(skills_dir, "sql_expert.md"), "w") as f:
        f.write("""---
name: sql_expert
description: "Expert SQL query builder and optimizer"
---

# SQL Expert Skill

## Rules
- Always use parameterized queries
- Prefer JOINs over subqueries
- Add indexes for frequently queried columns
- Validate input before building queries
""")

    # Skill 3: api_tester (sub-directory with SKILL.md + helper files)
    api_dir = os.path.join(skills_dir, "api_tester")
    os.makedirs(api_dir)
    with open(os.path.join(api_dir, "SKILL.md"), "w") as f:
        f.write("""---
name: api_tester
description: "Automated API endpoint testing"
---

# API Tester Skill

## Steps
1. Use `http_request` to call endpoints
2. Verify response status and body
3. Test error cases (4xx, 5xx)
4. Measure response time
""")
    # Helper file (should be ignored by skill parser)
    with open(os.path.join(api_dir, "helpers.py"), "w") as f:
        f.write("# Helper module — not a SKILL.md\n")

    # Dotfile directory (should be ignored)
    hidden_dir = os.path.join(skills_dir, ".hidden_skill")
    os.makedirs(hidden_dir)
    with open(os.path.join(hidden_dir, "SKILL.md"), "w") as f:
        f.write("---\nname: hidden\n---\nShould not load!")

    return skills_dir


def create_memory_files(base_dir: str) -> list:
    """Create realistic AGENTS.md and CLAWAGENTS.md."""
    agents_md = os.path.join(base_dir, "AGENTS.md")
    with open(agents_md, "w") as f:
        f.write("""# Agent Memory

## Project: ClawAgents Framework
- Language: Python + TypeScript
- Architecture: ReAct loop with tool registry
- Coding style: async/await, type hints everywhere
- Test with pytest (Python) and node:test (TypeScript)
- Max line length: 100 characters
- Always run `npx tsc --noEmit` before committing TypeScript

## Key Decisions
- Skills auto-discovered from ./skills
- Memory auto-discovered from ./AGENTS.md
- Hooks are convenience methods on ClawAgent
""")

    clawagents_md = os.path.join(base_dir, "CLAWAGENTS.md")
    with open(clawagents_md, "w") as f:
        f.write("""# ClawAgents Configuration

## Security Rules
- NEVER execute shell commands without approval
- Block `execute` tool by default in production
- Truncate all tool outputs to 5000 chars
- Redact API keys from output
""")

    return [agents_md, clawagents_md]


# ─── Project Files for Tool Testing ──────────────────────────────────────

def create_project_files(base_dir: str):
    """Create a realistic project structure for tools to operate on."""
    src = os.path.join(base_dir, "src")
    os.makedirs(src)

    with open(os.path.join(src, "main.py"), "w") as f:
        f.write("""import os
import sys

# TODO: Add error handling
def process_data(input_path: str) -> dict:
    with open(input_path) as f:
        data = json.loads(f.read())
    return data

# TODO: Implement caching
def get_user(user_id: int):
    # BUG: No input validation
    return db.query(f"SELECT * FROM users WHERE id = {user_id}")

def main():
    result = process_data(sys.argv[1])
    print(result)
""")

    with open(os.path.join(src, "utils.py"), "w") as f:
        f.write("""def format_output(data: dict) -> str:
    return str(data)

def validate_input(s: str) -> bool:
    # TODO: Add proper validation
    return len(s) > 0

API_KEY = "sk-secret-123456789"  # BUG: Hardcoded secret
""")

    tests = os.path.join(base_dir, "tests")
    os.makedirs(tests)
    with open(os.path.join(tests, "test_main.py"), "w") as f:
        f.write("""import pytest

def test_process_data():
    assert True  # TODO: real test

def test_get_user():
    assert True  # TODO: real test
""")


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class TestCustomToolRegistration:
    """Test registering and executing multiple custom tools."""

    @pytest.mark.asyncio
    async def test_multiple_custom_tools(self):
        registry = ToolRegistry()
        calc = CalculatorTool()
        db = DatabaseQueryTool()
        http = HttpClientTool()

        registry.register(calc)
        registry.register(db)
        registry.register(http)

        # All registered
        names = [t.name for t in registry.list()]
        assert "calculate" in names
        assert "query_db" in names
        assert "http_request" in names

        # Execute each
        r1 = await registry.execute_tool("calculate", {"expression": "2 ** 10"})
        assert r1.success and r1.output == "1024"

        r2 = await registry.execute_tool("query_db", {"sql": "SELECT * FROM users"})
        assert r2.success and "Alice" in r2.output

        r3 = await registry.execute_tool("http_request", {"url": "https://api.weather.com/forecast"})
        assert r3.success and "sunny" in r3.output

    @pytest.mark.asyncio
    async def test_custom_tools_with_builtins(self):
        """Custom tools alongside built-in filesystem/exec tools."""
        from clawagents.tools.filesystem import filesystem_tools
        from clawagents.tools.exec import exec_tools
        from clawagents.tools.todolist import todolist_tools

        registry = ToolRegistry()
        for t in filesystem_tools + exec_tools + todolist_tools:
            registry.register(t)

        registry.register(CalculatorTool())
        registry.register(DatabaseQueryTool())

        names = [t.name for t in registry.list()]
        # Built-ins
        assert "ls" in names
        assert "read_file" in names
        assert "execute" in names
        assert "write_todos" in names
        # Custom
        assert "calculate" in names
        assert "query_db" in names

    @pytest.mark.asyncio
    async def test_describe_for_llm_with_custom_tools(self):
        """Verify describe_for_llm includes custom tool descriptions."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(DatabaseQueryTool())

        desc = registry.describe_for_llm()
        assert "calculate" in desc
        assert "mathematical expression" in desc.lower()
        assert "query_db" in desc
        assert "SQL" in desc


class TestSkillDiscoveryAndLoading:
    """Test full skill loading pipeline."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.skills_dir = create_skill_dir(self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_skill_store_loads_all(self):
        from clawagents.tools.skills import SkillStore

        store = SkillStore()
        store.add_directory(self.skills_dir)
        await store.load_all()

        skills = store.list()
        names = [s.name for s in skills]

        assert "code_review" in names
        assert "sql_expert" in names
        assert "api_tester" in names
        assert "hidden" not in names  # .hidden_skill should be ignored
        assert len(skills) == 3

    @pytest.mark.asyncio
    async def test_skill_frontmatter_parsing(self):
        from clawagents.tools.skills import SkillStore

        store = SkillStore()
        store.add_directory(self.skills_dir)
        await store.load_all()

        review = store.get("code_review")
        assert review is not None
        assert review.description == "Structured code review with severity levels"
        assert "Rate each issue" in review.content

        sql = store.get("sql_expert")
        assert sql is not None
        assert sql.description == "Expert SQL query builder and optimizer"

    @pytest.mark.asyncio
    async def test_skill_tools_creation(self):
        from clawagents.tools.skills import SkillStore, create_skill_tools

        store = SkillStore()
        store.add_directory(self.skills_dir)
        await store.load_all()

        tools = create_skill_tools(store)
        tool_names = [t.name for t in tools]
        assert "list_skills" in tool_names
        assert "use_skill" in tool_names

    @pytest.mark.asyncio
    async def test_list_skills_tool(self):
        from clawagents.tools.skills import SkillStore, create_skill_tools

        store = SkillStore()
        store.add_directory(self.skills_dir)
        await store.load_all()

        tools = create_skill_tools(store)
        list_tool = [t for t in tools if t.name == "list_skills"][0]
        result = await list_tool.execute({})

        assert result.success
        assert "code_review" in result.output
        assert "sql_expert" in result.output
        assert "api_tester" in result.output
        assert "3" in result.output  # count

    @pytest.mark.asyncio
    async def test_use_skill_tool(self):
        from clawagents.tools.skills import SkillStore, create_skill_tools

        store = SkillStore()
        store.add_directory(self.skills_dir)
        await store.load_all()

        tools = create_skill_tools(store)
        use_tool = [t for t in tools if t.name == "use_skill"][0]

        result = await use_tool.execute({"name": "code_review"})
        assert result.success
        assert "Rate each issue" in result.output
        assert "CRITICAL" in result.output

    @pytest.mark.asyncio
    async def test_use_skill_not_found(self):
        from clawagents.tools.skills import SkillStore, create_skill_tools

        store = SkillStore()
        store.add_directory(self.skills_dir)
        await store.load_all()

        tools = create_skill_tools(store)
        use_tool = [t for t in tools if t.name == "use_skill"][0]

        result = await use_tool.execute({"name": "nonexistent_skill"})
        assert result.success is False
        assert "not found" in result.error.lower()


class TestMemoryInjection:
    """Test memory loading and injection into LLM messages."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory_files = create_memory_files(self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_memory_loaded_with_correct_tags(self):
        from clawagents.memory.loader import load_memory_files

        result = load_memory_files(self.memory_files)
        assert result is not None
        assert 'source="AGENTS.md"' in result
        assert 'source="CLAWAGENTS.md"' in result
        assert "agent_memory" in result

    def test_memory_contains_project_rules(self):
        from clawagents.memory.loader import load_memory_files

        result = load_memory_files(self.memory_files)
        assert "async/await" in result
        assert "type hints" in result
        assert "npx tsc --noEmit" in result

    def test_memory_contains_security_rules(self):
        from clawagents.memory.loader import load_memory_files

        result = load_memory_files(self.memory_files)
        assert "NEVER execute shell commands" in result
        assert "Redact API keys" in result

    def test_compose_before_llm_injects_memory(self):
        from clawagents.agent import _compose_before_llm

        hook = _compose_before_llm(self.memory_files, None)
        messages = [{"role": "system", "content": "You are helpful."}]
        result = hook(messages)

        # Memory should be appended to system message
        assert "async/await" in result[0]["content"]
        assert "type hints" in result[0]["content"]

    def test_compose_before_llm_injects_skills_summary(self):
        from clawagents.agent import _compose_before_llm

        skill_summary = "## Available Skills\n- **code_review**: Structured review\n- **sql_expert**: SQL queries"
        hook = _compose_before_llm([], skill_summary)
        messages = [{"role": "system", "content": "Base prompt."}]
        result = hook(messages)

        assert "code_review" in result[0]["content"]
        assert "sql_expert" in result[0]["content"]

    def test_compose_with_both_memory_and_skills(self):
        from clawagents.agent import _compose_before_llm

        skill_summary = "## Available Skills\n- **api_tester**: Test APIs"
        hook = _compose_before_llm(self.memory_files, skill_summary)
        messages = [{"role": "system", "content": "Agent base."}]
        result = hook(messages)

        # Both present
        assert "async/await" in result[0]["content"]  # memory
        assert "api_tester" in result[0]["content"]    # skills


class TestFullPipelineWithHooks:
    """Test the complete pipeline: custom tools + skills + memory + hooks."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.skills_dir = create_skill_dir(self.tmpdir)
        self.memory_files = create_memory_files(self.tmpdir)
        create_project_files(self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    @pytest.mark.asyncio
    async def test_full_pipeline_wiring(self):
        """Verify all components wire together correctly."""
        from clawagents.agent import ClawAgent, _compose_before_llm
        from clawagents.tools.skills import SkillStore, create_skill_tools

        # 1. Build registry with built-in + custom tools
        from clawagents.tools.filesystem import filesystem_tools
        from clawagents.tools.exec import exec_tools
        from clawagents.tools.todolist import todolist_tools

        registry = ToolRegistry()
        for t in filesystem_tools + exec_tools + todolist_tools:
            registry.register(t)

        db_tool = DatabaseQueryTool()
        calc_tool = CalculatorTool()
        http_tool = HttpClientTool()
        registry.register(db_tool)
        registry.register(calc_tool)
        registry.register(http_tool)

        # 2. Load skills
        store = SkillStore()
        store.add_directory(self.skills_dir)
        await store.load_all()

        for tool in create_skill_tools(store):
            registry.register(tool)

        # 3. Build memory + skills hook
        skills = store.list()
        skill_summary = "## Available Skills\n" + "\n".join(
            f"- **{s.name}**: {s.description}" for s in skills
        )
        before_llm = _compose_before_llm(self.memory_files, skill_summary)

        # 4. Create agent
        agent = ClawAgent(
            llm=MagicMock(),
            tools=registry,
            system_prompt="You are a senior developer.",
            before_llm=before_llm,
        )

        # 5. Apply convenience hooks
        agent.block_tools("execute")  # Security: no shell
        agent.truncate_output(3000)   # Performance: cap output

        # ── Verify everything is wired ────────────────────────────────

        # Tools are registered
        tool_names = [t.name for t in registry.list()]
        assert "calculate" in tool_names
        assert "query_db" in tool_names
        assert "http_request" in tool_names
        assert "ls" in tool_names
        assert "read_file" in tool_names
        assert "use_skill" in tool_names
        assert "list_skills" in tool_names
        assert "write_todos" in tool_names

        # Execute blocked
        assert agent.before_tool("execute", {"command": "rm -rf /"}) is False
        assert agent.before_tool("read_file", {"path": "."}) is True
        assert agent.before_tool("calculate", {"expression": "1+1"}) is True

        # Truncation works
        big_result = ToolResult(success=True, output="x" * 10000)
        truncated = agent.after_tool("read_file", {}, big_result)
        assert len(truncated.output) < 10000
        assert "truncated" in truncated.output

        # Memory + skills injected
        msgs = [{"role": "system", "content": "Base."}]
        injected = agent.before_llm(msgs)
        # This won't work because block_tools overwrites before_tool, not before_llm
        # before_llm was set in constructor; inject_context stacks on it
        # But we used _compose_before_llm directly, so it should still work
        assert "async/await" in injected[0]["content"]

    @pytest.mark.asyncio
    async def test_multi_tool_execution_chain(self):
        """Simulate a realistic multi-step tool chain."""
        from clawagents.tools.filesystem import (
            LsTool, ReadFileTool, GrepTool, EditFileTool, WriteFileTool
        )
        from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool, reset_todos

        reset_todos()

        # Step 1: Plan the work
        plan = WriteTodosTool()
        r = await plan.execute({"todos": [
            "List project files",
            "Find all TODOs",
            "Find security issues",
            "Fix SQL injection",
            "Add input validation",
        ]})
        assert r.success and "0/5" in r.output

        # Step 2: List files
        ls = LsTool()
        r = await ls.execute({"path": os.path.join(self.tmpdir, "src")})
        assert r.success
        assert "main.py" in r.output

        # Step 3: Mark step 1 done, find TODOs
        update = UpdateTodoTool()
        await update.execute({"index": 0})

        grep = GrepTool()
        r = await grep.execute({
            "path": self.tmpdir,
            "pattern": "TODO",
            "recursive": True
        })
        assert r.success
        assert "TODO" in r.output
        assert int(r.output.split(" ")[0]) >= 4  # At least 4 TODOs

        await update.execute({"index": 1})

        # Step 4: Find security issues  
        r = await grep.execute({
            "path": os.path.join(self.tmpdir, "src"),
            "pattern": "sk-secret",
            "recursive": True
        })
        assert r.success
        assert "utils.py" in r.output

        await update.execute({"index": 2})

        # Step 5: Fix SQL injection
        edit = EditFileTool()
        r = await edit.execute({
            "path": os.path.join(self.tmpdir, "src", "main.py"),
            "target": 'f"SELECT * FROM users WHERE id = {user_id}"',
            "replacement": '"SELECT * FROM users WHERE id = ?", (user_id,)'
        })
        assert r.success

        await update.execute({"index": 3})

        # Step 6: Verify fix
        read = ReadFileTool()
        r = await read.execute({"path": os.path.join(self.tmpdir, "src", "main.py")})
        assert "user_id," in r.output  # parameterized
        assert "f\"SELECT" not in r.output  # f-string removed

        await update.execute({"index": 4})

        # Verify all done
        r = await update.execute({"index": 4})  # re-mark (idempotent)
        assert "5/5" in r.output

    @pytest.mark.asyncio
    async def test_custom_tool_parallel_execution(self):
        """Execute multiple custom tools in parallel."""
        registry = ToolRegistry()
        db = DatabaseQueryTool()
        http = HttpClientTool()
        calc = CalculatorTool()

        registry.register(db)
        registry.register(http)
        registry.register(calc)

        # Simulate parallel tool calls (like the agent would do)
        results = await asyncio.gather(
            registry.execute_tool("calculate", {"expression": "100 * 0.15"}),
            registry.execute_tool("query_db", {"sql": "SELECT COUNT(*) FROM users"}),
            registry.execute_tool("http_request", {"url": "https://api.weather.com/now"}),
        )

        assert results[0].success and results[0].output == "15.0"
        assert results[1].success and results[1].output == "3"
        assert results[2].success and "sunny" in results[2].output

        # Verify all tools executed
        assert len(db.queries_executed) == 1
        assert len(http.requests_made) == 1


class TestAutoDiscoveryWithFactory:
    """Test auto-discovery through the factory (mocked LLM)."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

        # Create auto-discoverable files
        create_skill_dir(self.tmpdir)
        create_memory_files(self.tmpdir)

    def teardown_method(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmpdir)

    def test_auto_discovers_memory(self):
        from clawagents.agent import _auto_discover_memory
        found = _auto_discover_memory()
        assert len(found) == 2
        names = [os.path.basename(f) for f in found]
        assert "AGENTS.md" in names
        assert "CLAWAGENTS.md" in names

    def test_auto_discovers_skills(self):
        from clawagents.agent import _auto_discover_skills
        found = _auto_discover_skills()
        assert len(found) >= 1
        assert any("skills" in f for f in found)

    @pytest.mark.asyncio
    async def test_full_factory_with_auto_discovery(self):
        """Test create_claw_agent with real auto-discovery (mocked LLM)."""
        from clawagents.agent import create_claw_agent

        with patch('clawagents.config.config.load_config') as mock_config, \
             patch('clawagents.providers.llm.create_provider') as mock_provider:
            mock_cfg = MagicMock()
            mock_cfg.streaming = True
            mock_cfg.provider = "gemini"
            mock_config.return_value = mock_cfg
            mock_provider.return_value = MagicMock()

            agent = create_claw_agent(
                model="gemini-3-flash",
                instruction="You are a code reviewer.",
                tools=[CalculatorTool(), DatabaseQueryTool()],
                # memory and skills should be auto-discovered from ./AGENTS.md and ./skills
            )

            # Verify instruction
            assert agent.system_prompt == "You are a code reviewer."

            # Verify tools registered
            tool_names = [t.name for t in agent.tools.list()]
            assert "calculate" in tool_names  # custom
            assert "query_db" in tool_names   # custom
            assert "ls" in tool_names         # built-in
            assert "read_file" in tool_names  # built-in
            assert "use_skill" in tool_names  # auto-discovered skills

            # Verify hooks can be applied
            agent.block_tools("execute")
            agent.inject_context("Review code carefully")
            agent.truncate_output(2000)

            assert agent.before_tool("execute", {}) is False
            assert agent.before_tool("calculate", {}) is True
