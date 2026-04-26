"""
Simulated test cases for the new ClawAgents interface.

Tests the full surface area WITHOUT hitting real APIs:
  - create_claw_agent with model string / None
  - instruction parameter
  - Convenience hooks: block_tools, allow_only_tools, inject_context, truncate_output
  - Auto-discovery of memory/skills
  - Built-in tool registration
  - Raw hooks
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock


# ─── Test: Factory creates agent with model string ────────────────────────

class TestCreateClawAgent:
    """Test the simplified create_claw_agent factory."""

    def test_factory_with_model_string(self):
        """create_claw_agent('gemini-3-flash') should resolve the model."""
        from clawagents.agent import _resolve_model
        from clawagents.providers.llm import LLMProvider

        with patch('clawagents.config.config.load_config') as mock_config, \
             patch('clawagents.providers.llm.create_provider') as mock_create:
            mock_cfg = MagicMock()
            mock_config.return_value = mock_cfg
            mock_provider = MagicMock(spec=LLMProvider)
            mock_create.return_value = mock_provider

            result = _resolve_model("gemini-3-flash", True)

            assert result is mock_provider
            mock_create.assert_called_once_with("gemini-3-flash", mock_cfg)

    def test_factory_with_openai_string(self):
        """create_claw_agent('gpt-5') should pass model string to create_provider."""
        from clawagents.agent import _resolve_model

        with patch('clawagents.config.config.load_config') as mock_config, \
             patch('clawagents.providers.llm.create_provider') as mock_create:
            mock_cfg = MagicMock()
            mock_config.return_value = mock_cfg
            mock_create.return_value = MagicMock()

            _resolve_model("gpt-5", True)

            mock_create.assert_called_once_with("gpt-5", mock_cfg)

    def test_factory_with_none_auto_detects(self):
        """create_claw_agent() should auto-detect from env."""
        from clawagents.agent import _resolve_model

        with patch('clawagents.config.config.load_config') as mock_config, \
             patch('clawagents.providers.llm.create_provider') as mock_create:
            mock_cfg = MagicMock()
            mock_config.return_value = mock_cfg
            mock_create.return_value = MagicMock()

            _resolve_model(None, True)

            # Should not modify model names — uses whatever config has
            mock_create.assert_called_once()


# ─── Test: Instruction parameter ──────────────────────────────────────────

class TestInstruction:
    """Test that instruction is passed as system_prompt."""

    def test_instruction_stored(self):
        """ClawAgent should store instruction as system_prompt."""
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(
            llm=MagicMock(),
            tools=ToolRegistry(),
            system_prompt="You are a code reviewer.",
        )
        assert agent.system_prompt == "You are a code reviewer."


# ─── Test: Convenience Hook - block_tools ─────────────────────────────────

class TestBlockTools:
    """Test agent.block_tools()."""

    def test_blocks_specified_tools(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.block_tools("execute", "write_file")

        assert agent.before_tool is not None
        # execute should be blocked
        assert agent.before_tool("execute", {}) is False
        # write_file should be blocked
        assert agent.before_tool("write_file", {"path": "x"}) is False
        # read_file should be allowed
        assert agent.before_tool("read_file", {"path": "x"}) is True
        # ls should be allowed
        assert agent.before_tool("ls", {}) is True

    def test_blocks_empty_allows_all(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.block_tools()  # block nothing

        assert agent.before_tool("execute", {}) is True
        assert agent.before_tool("anything", {}) is True


# ─── Test: Convenience Hook - allow_only_tools ────────────────────────────

class TestAllowOnlyTools:
    """Test agent.allow_only_tools()."""

    def test_allows_only_specified(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.allow_only_tools("read_file", "ls", "grep")

        assert agent.before_tool("read_file", {}) is True
        assert agent.before_tool("ls", {}) is True
        assert agent.before_tool("grep", {"pattern": "x"}) is True
        # Not in allowlist
        assert agent.before_tool("execute", {}) is False
        assert agent.before_tool("write_file", {}) is False
        assert agent.before_tool("edit_file", {}) is False


# ─── Test: Convenience Hook - inject_context ──────────────────────────────

class TestInjectContext:
    """Test agent.inject_context()."""

    def test_injects_message(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.inject_context("Always respond in Spanish")

        messages = [{"role": "system", "content": "You are helpful."}]
        result = agent.before_llm(messages)

        assert len(result) == 2
        last = result[-1]
        content = last.content if hasattr(last, "content") else last["content"]
        assert "[Context] Always respond in Spanish" in content

    def test_stacks_multiple_contexts(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.inject_context("Rule 1: Be brief")
        agent.inject_context("Rule 2: Use bullet points")

        messages = [{"role": "user", "content": "hello"}]
        result = agent.before_llm(messages)

        # Original + 2 injected
        assert len(result) == 3
        c1 = result[1].content if hasattr(result[1], "content") else result[1]["content"]
        c2 = result[2].content if hasattr(result[2], "content") else result[2]["content"]
        assert "Rule 1" in c1
        assert "Rule 2" in c2


# ─── Test: Convenience Hook - truncate_output ─────────────────────────────

class TestTruncateOutput:
    """Test agent.truncate_output()."""

    def test_truncates_long_output(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry, ToolResult

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.truncate_output(100)

        long_result = ToolResult(success=True, output="x" * 500)
        result = agent.after_tool("read_file", {}, long_result)

        assert len(result.output) < 500
        assert "truncated" in result.output
        assert result.success is True

    def test_passes_short_output_unchanged(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry, ToolResult

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.truncate_output(5000)

        short_result = ToolResult(success=True, output="short output")
        result = agent.after_tool("ls", {}, short_result)

        assert result.output == "short output"
        assert result.success is True


# ─── Test: Auto-Discovery ─────────────────────────────────────────────────

class TestAutoDiscovery:
    """Test auto-discovery of memory files and skill directories."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self.tmpdir)

    def test_discovers_agents_md(self):
        from clawagents.agent import _auto_discover_memory

        # Create AGENTS.md
        with open("AGENTS.md", "w") as f:
            f.write("# Project Memory\nUse async/await.")

        found = _auto_discover_memory()
        assert len(found) == 1
        assert found[0].endswith("AGENTS.md")

    def test_discovers_clawagents_md(self):
        from clawagents.agent import _auto_discover_memory

        with open("CLAWAGENTS.md", "w") as f:
            f.write("# ClawAgents Config")

        found = _auto_discover_memory()
        assert len(found) == 1
        assert found[0].endswith("CLAWAGENTS.md")

    def test_discovers_both_memory_files(self):
        from clawagents.agent import _auto_discover_memory

        with open("AGENTS.md", "w") as f:
            f.write("agents")
        with open("CLAWAGENTS.md", "w") as f:
            f.write("clawagents")

        found = _auto_discover_memory()
        assert len(found) == 2

    def test_no_memory_files(self):
        from clawagents.agent import _auto_discover_memory

        found = _auto_discover_memory()
        assert len(found) == 0

    def test_discovers_skills_dir(self):
        from clawagents.agent import _auto_discover_skills

        os.makedirs("skills")
        found = _auto_discover_skills()
        assert len(found) >= 1
        assert any(f.endswith("skills") or f.endswith("Skills") for f in found)

    def test_discovers_dotskills_dir(self):
        from clawagents.agent import _auto_discover_skills

        os.makedirs(".skills")
        found = _auto_discover_skills()
        assert len(found) == 1
        assert found[0].endswith(".skills")

    def test_discovers_multiple_skill_dirs(self):
        from clawagents.agent import _auto_discover_skills

        os.makedirs("skills", exist_ok=True)
        os.makedirs(".skills", exist_ok=True)
        found = _auto_discover_skills()
        assert len(found) >= 2

    def test_no_skill_dirs(self):
        from clawagents.agent import _auto_discover_skills

        found = _auto_discover_skills()
        assert len(found) == 0


# ─── Test: Built-in Tool Registration ─────────────────────────────────────

class TestBuiltinToolRegistration:
    """Verify that all expected built-in tools are available."""

    def test_filesystem_tools_exist(self):
        from clawagents.tools.filesystem import filesystem_tools

        names = [t.name for t in filesystem_tools]
        assert "ls" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "grep" in names
        assert "glob" in names

    def test_exec_tools_exist(self):
        from clawagents.tools.exec import exec_tools

        names = [t.name for t in exec_tools]
        assert "execute" in names

    def test_todolist_tools_exist(self):
        from clawagents.tools.todolist import todolist_tools

        names = [t.name for t in todolist_tools]
        assert "write_todos" in names
        assert "update_todo" in names


# ─── Test: _to_list helper ────────────────────────────────────────────────

class TestToList:
    """Test the _to_list helper handles all input types."""

    def test_none(self):
        from clawagents.agent import _to_list
        assert _to_list(None) == []

    def test_string(self):
        from clawagents.agent import _to_list
        assert _to_list("./AGENTS.md") == ["./AGENTS.md"]

    def test_list(self):
        from clawagents.agent import _to_list
        assert _to_list(["a", "b"]) == ["a", "b"]

    def test_pathlike(self):
        from clawagents.agent import _to_list
        from pathlib import Path
        result = _to_list(Path("./skills"))
        assert len(result) == 1


# ─── Test: Raw hooks ─────────────────────────────────────────────────────

class TestRawHooks:
    """Test raw hook assignment."""

    def test_before_tool_raw(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.before_tool = lambda name, args: name != "execute"

        assert agent.before_tool("read_file", {}) is True
        assert agent.before_tool("execute", {}) is False

    def test_after_tool_raw(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry, ToolResult

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.after_tool = lambda name, args, result: ToolResult(
            success=result.success, output="REDACTED"
        )

        result = agent.after_tool("ls", {}, ToolResult(success=True, output="secret data"))
        assert result.output == "REDACTED"

    def test_before_llm_raw(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())
        agent.before_llm = lambda msgs: msgs + [{"role": "user", "content": "extra"}]

        result = agent.before_llm([{"role": "system", "content": "hi"}])
        assert len(result) == 2
        assert result[-1]["content"] == "extra"


# ─── Test: Advisor Model ────────────────────────────────────────────────

class TestAdvisorModel:
    """Test advisor model integration."""

    def test_clawagent_stores_advisor_fields(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        mock_llm = MagicMock()
        mock_advisor = MagicMock()
        mock_advisor.name = "advisor-mock"

        agent = ClawAgent(
            llm=mock_llm, tools=ToolRegistry(),
            advisor_llm=mock_advisor, advisor_max_calls=5,
        )

        assert agent.advisor_llm is mock_advisor
        assert agent.advisor_max_calls == 5

    def test_clawagent_defaults_advisor_none(self):
        from clawagents.agent import ClawAgent
        from clawagents.tools.registry import ToolRegistry

        agent = ClawAgent(llm=MagicMock(), tools=ToolRegistry())

        assert agent.advisor_llm is None
        assert agent.advisor_max_calls == 3

    def test_factory_resolves_advisor_model_string(self):
        from clawagents.agent import create_claw_agent

        agent = create_claw_agent(
            "gpt-5-nano",
            advisor_model="gpt-5-nano",
        )

        assert agent.advisor_llm is not None
        assert agent.advisor_max_calls == 3

    def test_factory_works_without_advisor(self):
        from clawagents.agent import create_claw_agent

        agent = create_claw_agent("gpt-5-nano")

        assert agent.advisor_llm is None

    def test_advisor_max_calls_from_env(self):
        from clawagents.agent import create_claw_agent

        with patch.dict(os.environ, {"ADVISOR_MAX_CALLS": "7"}):
            agent = create_claw_agent(
                "gpt-5-nano",
                advisor_model="gpt-5-nano",
            )
            assert agent.advisor_max_calls == 7

    def test_advisor_max_calls_param_overrides_env(self):
        from clawagents.agent import create_claw_agent

        with patch.dict(os.environ, {"ADVISOR_MAX_CALLS": "99"}):
            agent = create_claw_agent(
                "gpt-5-nano",
                advisor_model="gpt-5-nano",
                advisor_max_calls=2,
            )
            assert agent.advisor_max_calls == 2

    def test_advisor_model_from_env(self):
        from clawagents.agent import create_claw_agent

        with patch.dict(os.environ, {"ADVISOR_MODEL": "gpt-5-nano"}):
            agent = create_claw_agent("gpt-5-nano")
            assert agent.advisor_llm is not None


# ─── Test: Credential Isolation ─────────────────────────────────────────

class TestCredentialIsolation:
    """Test that sensitive env vars are stripped from subprocess."""

    def test_sanitized_env_strips_keys(self):
        from clawagents.sandbox.local import LocalBackend

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret", "SAFE_VAR": "safe"}):
            backend = LocalBackend()
            sanitized = backend._sanitized_env()
            assert "OPENAI_API_KEY" not in sanitized
            assert sanitized["SAFE_VAR"] == "safe"

    def test_sanitized_env_strips_all_sensitive(self):
        from clawagents.sandbox.local import LocalBackend

        sensitive = {
            "OPENAI_API_KEY": "x", "GEMINI_API_KEY": "x",
            "ANTHROPIC_API_KEY": "x", "ADVISOR_API_KEY": "x",
            "GATEWAY_API_KEY": "x", "TAVILY_API_KEY": "x",
        }
        with patch.dict(os.environ, sensitive):
            backend = LocalBackend()
            sanitized = backend._sanitized_env()
            for key in sensitive:
                assert key not in sanitized

    @pytest.mark.asyncio
    async def test_execute_does_not_leak_keys(self):
        from clawagents.sandbox.local import LocalBackend

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-leaked-key"}):
            backend = LocalBackend()
            result = await backend.exec("env")
            assert "sk-leaked-key" not in result.stdout, "API key leaked into subprocess"


# ─── Test: Lazy Tool Provisioning ───────────────────────────────────────

class TestLazyToolProvisioning:
    """Test that lazy tools defer initialization."""

    def test_lazy_tool_defers_import(self):
        from clawagents.tools.registry import LazyTool

        lazy = LazyTool(
            name="mock_tool",
            description="A mock tool",
            parameters={"x": {"type": "string", "description": "input"}},
            module_path="clawagents.tools.think",
            class_name="ThinkTool",
        )

        # Schema available immediately
        assert lazy.name == "mock_tool"
        assert lazy.description == "A mock tool"
        # Not resolved yet
        assert lazy._resolved is None

    def test_factory_registers_lazy_tools(self):
        from clawagents.agent import create_claw_agent

        agent = create_claw_agent("gpt-5-nano")
        tool_names = [t.name for t in agent.tools.list()]

        assert "read_file" in tool_names
        assert "execute" in tool_names
        assert "ls" in tool_names
        assert "grep" in tool_names
        assert "web_fetch" in tool_names

    def test_factory_lazy_tool_schemas_match_implementations(self):
        from clawagents.agent import create_claw_agent
        from clawagents.sandbox.local import LocalBackend
        from clawagents.tools.advanced_fs import create_advanced_fs_tools
        from clawagents.tools.filesystem import create_filesystem_tools

        agent = create_claw_agent("gpt-5-nano")
        by_name = {t.name: t for t in agent.tools.list()}
        real_tools = {
            t.name: t for t in [
                *create_filesystem_tools(LocalBackend()),
                *create_advanced_fs_tools(LocalBackend()),
            ]
        }

        for name in ("edit_file", "grep", "tree"):
            assert by_name[name].parameters == real_tools[name].parameters
