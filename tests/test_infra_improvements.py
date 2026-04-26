from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawagents.eval import run_agent_environment
from clawagents.explorer import create_explorer_tools
from clawagents.agent import create_claw_agent
from clawagents.graph.agent_loop import AgentState, run_agent_graph
from clawagents.providers.llm import LLMMessage, LLMResponse, NativeToolCall
from clawagents.rl import Trajectory, to_next_state_transitions
from clawagents.run_result import RunResult
from clawagents.sandbox.backend import ExecResult
from clawagents.sandbox.docker import DockerBackend
from clawagents.session import InMemorySession
from clawagents.tools.cache import SqliteResultCacheManager
from clawagents.tools.catalog import create_tool_discovery_tools, names_for_tool_profile
from clawagents.tools.exec import create_exec_tools
from clawagents.tools.registry import ToolRegistry, ToolResult


class EchoTool:
    name = "read_file"
    description = "Read a file"
    parameters = {"value": {"type": "string", "description": "value"}}

    async def execute(self, args):
        return ToolResult(True, f"echo:{args.get('value', '')}")


class WriteTool:
    name = "write_file"
    description = "Write a file"
    parameters = {"path": {"type": "string", "description": "path"}}

    async def execute(self, args):
        return ToolResult(True, "wrote")


class BadlyNamedSearchTool:
    name = "scan_x7"
    description = "Process text units"
    keywords = ["search", "find text", "file contents"]
    parameters = {"value": {"type": "string", "description": "value"}}

    async def execute(self, args):
        return ToolResult(True, "ok")


class FakeLLM:
    name = "fake"

    async def chat(self, *args, **kwargs):
        raise AssertionError("chat should not be called")


@pytest.mark.asyncio
async def test_compact_tool_discovery_exposes_searchable_catalog_and_profiles():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(WriteTool())
    for tool in create_tool_discovery_tools(registry):
        registry.register(tool)

    result = await registry.execute_tool("tool_discover", {"query": "read"})
    assert result.success is True
    found = json.loads(str(result.output))
    assert [item["name"] for item in found] == ["read_file"]

    registry.register(BadlyNamedSearchTool())
    keyword_result = await registry.execute_tool("tool_discover", {"query": "find text"})
    assert keyword_result.success is True
    keyword_found = json.loads(str(keyword_result.output))
    assert [item["name"] for item in keyword_found] == ["scan_x7"]
    assert keyword_found[0]["keywords"] == ["search", "find text", "file contents"]

    token_result = await registry.execute_tool("tool_discover", {"query": "find units"})
    assert token_result.success is True
    token_found = json.loads(str(token_result.output))
    assert [item["name"] for item in token_found] == ["scan_x7"]

    described = await registry.execute_tool("tool_describe", {"name": "scan_x7"})
    assert described.success is True
    assert json.loads(str(described.output))["keywords"] == ["search", "find text", "file contents"]

    names = names_for_tool_profile(registry, "read-only")
    assert "read_file" in names
    assert "write_file" not in names

    bounded = ToolRegistry()
    bounded.register(EchoTool())
    bounded.register(WriteTool())
    for tool in create_tool_discovery_tools(bounded, max_profile="read-only"):
        bounded.register(tool)
    denied = await bounded.execute_tool("tool_describe", {"name": "write_file"})
    assert denied.success is False


@pytest.mark.asyncio
async def test_agent_factory_lazy_tools_preserve_discovery_keywords():
    agent = create_claw_agent(FakeLLM(), memory=[], skills=[])
    assert agent.tools.get("tool_discover") is not None

    result = await agent.tools.execute_tool(
        "tool_discover",
        {"query": "find text", "profile": "read-only"},
    )

    assert result.success is True
    found = json.loads(str(result.output))
    assert found[0]["name"] == "grep"
    assert "find text" in found[0]["keywords"]

    list_result = await agent.tools.execute_tool(
        "tool_discover",
        {"query": "list folder", "profile": "read-only"},
    )
    list_found = json.loads(str(list_result.output))
    assert any(item["name"] == "ls" for item in list_found)

    edit_result = await agent.tools.execute_tool(
        "tool_discover",
        {"query": "edit text", "profile": "full"},
    )
    edit_found = json.loads(str(edit_result.output))
    assert any(item["name"] == "edit_file" for item in edit_found)


@pytest.mark.asyncio
async def test_execute_returns_structured_context_for_nonzero_command_exits():
    class Backend:
        async def exec(self, command, timeout=None, cwd=None, env=None):
            return ExecResult(
                stdout="F\nFAILED tests/test_sample.py::test_demo",
                stderr="assertion failed",
                exit_code=1,
            )

    tool = create_exec_tools(Backend())[0]
    result = await tool.execute({"command": "pytest"})

    assert result.success is False
    payload = json.loads(str(result.output))
    assert payload["command_executed"] is True
    assert payload["exit_code"] == 1
    assert payload["command"] == "pytest"
    assert "FAILED" in payload["stdout"]
    assert "assertion failed" in payload["stderr"]
    assert "nonzero" in payload["interpretation"].lower()


@pytest.mark.asyncio
async def test_repeated_execute_calls_get_command_specific_recovery_hint():
    class RepeatingExecuteLLM:
        name = "repeat"

        def __init__(self):
            self.calls = 0
            self.seen = []

        async def chat(self, messages, **kwargs):
            self.calls += 1
            self.seen.append(list(messages))
            if self.calls <= 4:
                return LLMResponse(
                    content="",
                    model="fake",
                    tokens_used=1,
                    tool_calls=[
                        NativeToolCall(
                            "execute",
                            {"command": "pytest"},
                            tool_call_id=f"call_{self.calls}",
                        )
                    ],
                )
            return LLMResponse(content="done", model="fake", tokens_used=1)

    class ExecuteTool:
        name = "execute"
        description = "Execute a command"
        parameters = {"command": {"type": "string", "description": "command", "required": True}}

        async def execute(self, args):
            return ToolResult(
                False,
                '{"command_executed":true,"exit_code":1,"stdout":"FAILED","stderr":""}',
                "Command exited with code 1: pytest",
            )

    llm = RepeatingExecuteLLM()
    registry = ToolRegistry()
    registry.register(ExecuteTool())

    await run_agent_graph(
        "run tests",
        llm,
        tools=registry,
        max_iterations=8,
        streaming=False,
        use_native_tools=True,
    )

    hints = [
        str(message.content)
        for batch in llm.seen
        for message in batch
        if message.role == "user"
    ]
    transcript = "\n".join(str(message.content) for batch in llm.seen for message in batch)
    assert "command_executed" in transcript
    assert "exit_code" in transcript
    assert any(
        "execute command" in hint and "nonzero" in hint and "Do not rerun" in hint
        for hint in hints
    )


def test_sqlite_result_cache_persists_successful_tool_results(tmp_path: Path):
    db_path = tmp_path / "cache.sqlite"
    first = SqliteResultCacheManager(db_path=db_path, default_ttl_s=60)
    first.set("expensive_lookup", {"key": "a"}, ToolResult(True, "hello"))

    second = SqliteResultCacheManager(db_path=db_path, default_ttl_s=60)
    cached = second.get("expensive_lookup", {"key": "a"})
    assert cached is not None
    assert cached.success is True
    assert cached.output == "hello"
    second.set("read_file", {"path": ".env"}, ToolResult(True, "secret"))
    assert second.get("read_file", {"path": ".env"}) is None


def test_docker_backend_builds_safe_docker_run_arguments(tmp_path: Path):
    backend = DockerBackend(root=tmp_path, image="python:3.12-alpine")
    args = backend.build_docker_args("echo hi", env={"OPENAI_API_KEY": "secret", "SAFE": "1"})

    assert args[0] == "run"
    assert "--rm" in args
    assert any(f"{tmp_path}:/workspace" in arg for arg in args)
    assert "SAFE=1" in args
    assert "OPENAI_API_KEY=secret" not in args


@pytest.mark.asyncio
async def test_docker_backend_timeout_uses_milliseconds(tmp_path: Path, monkeypatch):
    seen = {}

    class Proc:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_create(*args, **kwargs):
        return Proc()

    async def fake_wait_for(awaitable, timeout):
        seen["timeout"] = timeout
        return await awaitable

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create)
    monkeypatch.setattr("asyncio.wait_for", fake_wait_for)

    backend = DockerBackend(root=tmp_path)
    result = await backend.exec("echo hi", timeout=1000)
    assert result.exit_code == 0
    assert seen["timeout"] == 1.0


@pytest.mark.asyncio
async def test_run_result_serializes_agent_state_and_can_resume_session_messages():
    state = AgentState(
        messages=[LLMMessage(role="user", content="hello"), LLMMessage(role="assistant", content="hi")],
        current_task="hello",
        status="done",
        result="hi",
        iterations=1,
        max_iterations=3,
        tool_calls=0,
    )
    result = RunResult.from_agent_state(state)
    restored = RunResult.from_state(result.to_state())
    assert restored.final_output == "hi"

    session = InMemorySession("resume")
    await restored.resume_into(session)
    assert len(await session.get_items()) == 2


@pytest.mark.asyncio
async def test_explorer_tools_list_tools_and_read_files_inside_root(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "demo.py").write_text("answer = 42\n", encoding="utf-8")
    subject = ToolRegistry()
    subject.register(EchoTool())

    explorer = ToolRegistry()
    for tool in create_explorer_tools(root=tmp_path, tools=subject):
        explorer.register(tool)

    catalog = await explorer.execute_tool("explorer_list_tools", {})
    assert "read_file" in str(catalog.output)

    file_result = await explorer.execute_tool("explorer_read_source", {"path": "src/demo.py"})
    assert file_result.success is True
    assert "answer" in str(file_result.output)


@pytest.mark.asyncio
async def test_run_agent_environment_is_gym_style_alias():
    async def responder(messages):
        return f"reply:{messages[-1]['content']}"

    class Env:
        async def init(self):
            return {"observations": [{"role": "user", "content": "start"}]}

        async def step(self, action):
            return {"observations": [], "reward": 1, "done": True}

    result = await run_agent_environment(responder, Env())
    assert result.total_reward == 1
    assert len(result.steps) == 1


def test_next_state_trajectory_export_links_actions_to_feedback():
    t = Trajectory(task="demo", model="mock")
    t.add_user("write code")
    t.add_assistant("done", trainable=True)
    t.add_user("tests failed", feedback=True)
    t.add_assistant("fixed", trainable=True)
    t.add_user("tests passed", feedback=True)

    transitions = to_next_state_transitions(t)
    assert len(transitions) == 2
    assert transitions[0]["action"]["content"] == "done"
    assert transitions[0]["next_state"]["content"] == "tests failed"
    assert transitions[1]["done"] is True
