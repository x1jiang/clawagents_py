from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawagents.eval import run_agent_environment
from clawagents.explorer import create_explorer_tools
from clawagents.agent import create_claw_agent
from clawagents.graph.agent_loop import AgentState, run_agent_graph
from clawagents.providers.llm import LLMMessage, LLMResponse, LLMProvider, NativeToolCall
from clawagents.rl import Trajectory, to_next_state_transitions
from clawagents.run_result import RunResult
from clawagents.sandbox.backend import ExecResult
from clawagents.sandbox.docker import DockerBackend
from clawagents.session import InMemorySession
from clawagents.tools.cache import SqliteResultCacheManager
from clawagents.tools.catalog import create_tool_discovery_tools, names_for_tool_profile
from clawagents.tools.exec import _format_nonzero_command_output, create_exec_tools
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


class FakeLLM(LLMProvider):
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


def test_execute_classifies_external_authentication_failure():
    payload = json.loads(
        _format_nonzero_command_output(
            "smbclient //server/share",
            1,
            "session setup failed: NT_STATUS_LOGON_FAILURE",
            "",
            "",
        )
    )
    interpretation = payload["interpretation"]
    assert "authentication" in interpretation.lower()
    assert "stop changing" in interpretation.lower()
    assert "user" in interpretation.lower()


def test_execute_classifies_npm_audit_findings_without_retrying():
    payload = json.loads(
        _format_nonzero_command_output(
            "bash -n deploy.sh; npm audit --omit=dev --audit-level=high",
            1,
            "# npm audit report\n10 vulnerabilities (4 high, 2 critical)",
            "",
            "",
        )
    )

    assert payload["success"] is False
    interpretation = payload["interpretation"].lower()
    assert "completed" in interpretation
    assert "failed security check" in interpretation
    assert "do not retry" in interpretation
    assert "lockfile" in interpretation
    assert "no fix available" in interpretation
    assert "earlier checks may have succeeded" in interpretation


@pytest.mark.parametrize(
    "command",
    [
        "npm --prefix web audit --omit=dev",
        "npm --workspace app audit --json",
        "cd frontend && npm -w app audit",
        "env CI=1 /usr/local/bin/npm --prefix=web audit",
    ],
)
def test_execute_recognizes_npm_audit_after_global_options(command):
    payload = json.loads(
        _format_nonzero_command_output(
            command,
            1,
            '{"auditReportVersion": 2, "vulnerabilities": {}}',
            "",
            "",
        )
    )
    assert "npm audit completed" in payload["interpretation"]


def test_execute_does_not_misclassify_unrelated_npm_command():
    payload = json.loads(
        _format_nonzero_command_output(
            "npm run audit",
            1,
            "vulnerabilities found by custom script",
            "",
            "",
        )
    )
    assert "npm audit completed" not in payload["interpretation"]


def test_execute_classifies_missing_package_without_suggesting_tool_churn():
    payload = json.loads(
        _format_nonzero_command_output(
            "conda create -n smb samba",
            1,
            "PackagesNotFoundError: samba",
            "",
            "",
        )
    )
    interpretation = payload["interpretation"]
    assert "package" in interpretation.lower()
    assert "package manager" in interpretation.lower()
    assert "do not" in interpretation.lower()


def test_execute_classifies_quarantine_as_application_outcome():
    payload = json.loads(
        _format_nonzero_command_output(
            "python split_all.py --input-dir input --output-dir output",
            1,
            "QUARANTINED: output/quarantine/run-id\nQuarantined runs: 1",
            "",
            "",
        )
    )

    interpretation = payload["interpretation"].lower()
    assert "application" in interpretation
    assert "quarantine" in interpretation
    assert "manifest" in interpretation


def test_execute_identifies_primary_and_cleanup_missing_executables():
    stderr = """Traceback (most recent call last):
  File \"publish_sandbox.py\", line 57, in main
    subprocess.run([\"kinit\", principal])
  File \"/usr/lib/python3.12/subprocess.py\", line 1955, in _execute_child
FileNotFoundError: [Errno 2] No such file or directory: 'kinit'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File \"publish_sandbox.py\", line 73, in main
    subprocess.run([\"kdestroy\"])
  File \"/usr/lib/python3.12/subprocess.py\", line 1955, in _execute_child
FileNotFoundError: [Errno 2] No such file or directory: 'kdestroy'
"""
    payload = json.loads(
        _format_nonzero_command_output("python3 publish_sandbox.py", 1, "", stderr, "")
    )

    interpretation = payload["interpretation"].lower()
    assert "missing required external executable `kinit`" in interpretation
    assert "`kdestroy`" in interpretation
    assert "cleanup" in interpretation
    assert "secondary" in interpretation
    assert "command -v" in interpretation
    assert "did not reach" in interpretation


def test_execute_explains_empty_compound_failure_with_redirected_output():
    command = (
        "rm -rf experiment/naming-test && mkdir -p experiment/naming-test/input "
        "&& python split_all.py --profile billing_img >/tmp/hca-naming-test.log "
        "&& python -c \"print('validate')\""
    )
    payload = json.loads(
        _format_nonzero_command_output(
            command,
            1,
            "",
            "",
            "[bash_validator: WARN DESTRUCTIVE — rm -rf is destructive]",
        )
    )

    interpretation = payload["interpretation"].lower()
    assert "advisory" in interpretation
    assert "did not cause" in interpretation
    assert "&&" in interpretation
    assert "later stages" in interpretation
    assert "/tmp/hca-naming-test.log" in interpretation
    assert "redirect" in interpretation
    assert "rerun" in interpretation


def test_execute_classifies_missing_input_and_empty_json_as_cascade():
    command = (
        "for pdf in missing-a.pdf present.pdf missing-b.pdf; do "
        "python diagnose.py \"$pdf\" >\"/tmp/$pdf.json\"; "
        "python -c 'import json,sys; json.load(open(sys.argv[1]))' "
        "\"/tmp/$pdf.json\"; done"
    )
    stderr = """split_hca_pdf.SplitError: Input file does not exist: missing-a.pdf
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
split_hca_pdf.SplitError: Input file does not exist: missing-b.pdf
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
"""
    stdout = "present.pdf:\npages 169 packets 54\n"
    payload = json.loads(
        _format_nonzero_command_output(command, 1, stdout, stderr, "")
    )

    interpretation = payload["interpretation"].lower()
    assert "primary" in interpretation
    assert "missing-a.pdf" in interpretation
    assert "missing-b.pdf" in interpretation
    assert "secondary" in interpretation
    assert "empty" in interpretation
    assert "redirection" in interpretation
    assert "preflight" in interpretation
    assert "producer" in interpretation
    assert "consumer" in interpretation
    assert "for` loop" in interpretation
    assert "continues" in interpretation
    assert "partial success" in interpretation


def test_execute_classifies_missing_python_module_as_interpreter_issue():
    payload = json.loads(
        _format_nonzero_command_output(
            "python3 - <<'PY'\nfrom pypdf import PdfReader\nPY",
            1,
            "",
            "ModuleNotFoundError: No module named 'pypdf'",
            "",
        )
    )

    interpretation = payload["interpretation"].lower()
    assert "selected python interpreter" in interpretation
    assert "`pypdf`" in interpretation
    assert "did not reach" in interpretation
    assert ".venv/bin/python" in interpretation
    assert "same interpreter" in interpretation
    assert "-m pip show" in interpretation
    assert "global" in interpretation


@pytest.mark.asyncio
@pytest.mark.parametrize("program", ["grep", "rg"])
async def test_execute_normalizes_search_exit_one_with_no_output(program: str):
    class Backend:
        async def exec(self, command, timeout=None, cwd=None, env=None):
            return ExecResult(stdout="", stderr="", exit_code=1)

    tool = create_exec_tools(Backend())[0]
    result = await tool.execute({"command": f"{program} needle manifest.json"})

    assert result.success is True
    assert "no matches" in str(result.output).lower()


@pytest.mark.asyncio
async def test_execute_keeps_non_search_exit_one_as_failure():
    class Backend:
        async def exec(self, command, timeout=None, cwd=None, env=None):
            return ExecResult(stdout="", stderr="", exit_code=1)

    tool = create_exec_tools(Backend())[0]
    result = await tool.execute({"command": "pytest"})

    assert result.success is False


def test_execute_redacts_high_entropy_shell_command_failure():
    secret = "vP7Vf5uipuaO"
    payload = json.loads(
        _format_nonzero_command_output(
            "python3 hca_smb.py ls",
            127,
            "",
            f"bash: line 1: {secret}: command not found",
            "",
        )
    )
    assert secret not in payload["stderr"]
    assert "[REDACTED:SHELL_SECRET]" in payload["stderr"]
    assert "unsafe secret interpolation" in payload["interpretation"]


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
    assert "Command exited with code 1" in transcript
    assert "FAILED" in transcript
    assert any(
        "execute command" in hint and "nonzero" in hint and "Do not rerun" in hint
        for hint in hints
    )


@pytest.mark.asyncio
async def test_three_failures_trigger_rethink_without_opt_in_flag():
    class FailureAwareLLM:
        name = "failure-aware"

        def __init__(self):
            self.calls = 0
            self.saw_rethink = False

        async def chat(self, messages, **kwargs):
            self.calls += 1
            transcript = "\n".join(str(message.content) for message in messages)
            self.saw_rethink = "Classify the failure" in transcript
            if self.saw_rethink:
                return LLMResponse(content="stopped", model="fake", tokens_used=1)
            return LLMResponse(
                content="",
                model="fake",
                tokens_used=1,
                tool_calls=[
                    NativeToolCall(
                        "probe",
                        {"attempt": self.calls},
                        tool_call_id=f"probe_{self.calls}",
                    )
                ],
            )

    class ProbeTool:
        name = "probe"
        description = "Probe an external dependency"
        parameters = {"attempt": {"type": "integer", "required": True}}

        async def execute(self, args):
            return ToolResult(False, "external service rejected request", "probe failed")

    llm = FailureAwareLLM()
    registry = ToolRegistry()
    registry.register(ProbeTool())
    result = await run_agent_graph(
        "diagnose external service",
        llm,
        tools=registry,
        max_iterations=8,
        streaming=False,
        use_native_tools=True,
        rethink=False,
    )
    assert result.result == "stopped"
    assert llm.saw_rethink is True
    # Three failing turns + the recovery turn; a configured final-check pass
    # may make one additional model call.
    assert llm.calls in (4, 5)


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
