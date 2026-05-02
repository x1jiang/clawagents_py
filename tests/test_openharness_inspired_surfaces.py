import asyncio
import json
import sys
from pathlib import Path


def test_provider_profile_resolves_builtin_and_explicit_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    from clawagents.provider_profiles import resolve_provider_profile

    resolved = resolve_provider_profile("ollama")
    assert resolved.model == "llama3.1"
    assert resolved.base_url == "http://localhost:11434/v1"
    assert resolved.api_key == "ollama"

    overridden = resolve_provider_profile(
        "ollama",
        model="gpt-5.4-nano",
        base_url="https://example.test/v1",
        api_key="explicit",
    )
    assert overridden.model == "gpt-5.4-nano"
    assert overridden.base_url == "https://example.test/v1"
    assert overridden.api_key == "explicit"


def test_dry_run_preview_is_static_and_reports_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    from clawagents.dry_run import build_dry_run_preview

    preview = build_dry_run_preview(task="grep for failing tests", profile="ollama")
    assert preview["dry_run"] is True
    assert preview["status"] == "ready"
    assert preview["provider"]["profile"] == "ollama"
    assert preview["provider"]["model"] == "llama3.1"
    assert preview["tool_count"] > 0
    assert "tool_discover" in preview["matching_tools"]


def test_permission_decision_reports_confirm_and_sensitive_path():
    from clawagents.permissions.mode import PermissionMode, evaluate_tool_permission

    default_decision = evaluate_tool_permission(
        "execute",
        mode=PermissionMode.DEFAULT,
        is_read_only=False,
        command="pip install demo",
    )
    assert default_decision.allowed is False
    assert default_decision.requires_confirmation is True
    assert "Package installation" in default_decision.reason

    sensitive = evaluate_tool_permission(
        "read_file",
        mode=PermissionMode.BYPASS,
        is_read_only=True,
        file_path=str(Path.home() / ".ssh" / "id_rsa"),
    )
    assert sensitive.allowed is False
    assert "sensitive credential path" in sensitive.reason


async def test_background_task_tools_run_and_return_output(tmp_path):
    from clawagents.tools.background_task import create_background_task_tools

    tools = {tool.name: tool for tool in create_background_task_tools()}
    created = await tools["task_create"].execute({
        "command": [sys.executable, "-c", "print('task-ok')"],
        "cwd": str(tmp_path),
    })
    assert created.success
    job_id = json.loads(str(created.output))["job_id"]

    await asyncio.sleep(0.2)
    status = await tools["task_status"].execute({"job_id": job_id})
    assert status.success
    assert json.loads(str(status.output))["running"] is False

    output = await tools["task_output"].execute({"job_id": job_id})
    assert output.success
    assert "task-ok" in str(output.output)


def test_plugin_compat_loader_reads_claude_style_manifest(tmp_path):
    plugin = tmp_path / "demo"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / "skills" / "review").mkdir(parents=True)
    (plugin / "commands").mkdir()
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({
            "name": "demo-plugin",
            "description": "Demo plugin",
            "skills_dir": "skills",
            "commands_dir": "commands",
            "mcp_file": ".mcp.json",
        }),
        encoding="utf-8",
    )
    (plugin / "skills" / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nBody",
        encoding="utf-8",
    )
    (plugin / "commands" / "hello.md").write_text("# Hello\nRun hello", encoding="utf-8")
    (plugin / ".mcp.json").write_text(json.dumps({"servers": {"demo": {"command": "x"}}}), encoding="utf-8")

    from clawagents.plugin_compat import load_plugin

    loaded = load_plugin(plugin)
    assert loaded is not None
    assert loaded.name == "demo-plugin"
    assert [skill.name for skill in loaded.skills] == ["review"]
    assert [command.name for command in loaded.commands] == ["hello"]
    assert "demo" in loaded.mcp_servers


async def test_mcp_auth_tool_updates_config_and_reconnects():
    from clawagents.mcp.manager import MCPServerManager
    from clawagents.tools.mcp_auth import MCPAuthTool

    class FakeServer:
        name = "demo"

        def __init__(self):
            self.params = {"url": "https://example.test/mcp", "headers": {}}
            self.reconnected = 0

        async def connect(self):
            self.reconnected += 1

        async def shutdown(self):
            pass

    server = FakeServer()
    manager = MCPServerManager([server])  # type: ignore[list-item]
    result = await MCPAuthTool(manager).execute({
        "server_name": "demo",
        "mode": "bearer",
        "value": "secret",
    })

    assert result.success
    assert server.params["headers"]["Authorization"] == "Bearer secret"
    assert server.reconnected == 1


async def test_compaction_preserves_carryover_and_emits_progress_events(tmp_path, monkeypatch):
    from clawagents.context.carryover import set_compaction_carryover
    from clawagents.graph.agent_loop import _compact_if_needed
    from clawagents.providers.llm import LLMMessage, LLMResponse
    from clawagents.run_context import RunContext

    monkeypatch.chdir(tmp_path)

    class FakeLLM:
        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
            return LLMResponse("summarized old work", model="fake", tokens_used=12)

    messages = [LLMMessage(role="system", content="system")]
    for idx in range(24):
        messages.append(LLMMessage(role="user", content=f"history {idx} " + ("x" * 500)))

    ctx = RunContext()
    set_compaction_carryover(
        ctx,
        task_focus="finish runtime continuity",
        recent_files=["src/clawagents/graph/agent_loop.py"],
        recent_work_log=["added failing tests"],
        invoked_skills=["autopilot"],
        active_workers=["worker-a"],
        channel_log=[{
            "channel_id": "telegram",
            "conversation_id": "chat-1",
            "body": "/status now",
        }],
        metadata={"release": "6.8"},
    )

    events = []
    compacted = await _compact_if_needed(
        messages,
        200,
        FakeLLM(),
        lambda kind, data: events.append((kind, data)),
        1.0,
        None,
        run_context=ctx,
    )

    summary = next(m.content for m in compacted if isinstance(m.content, str) and "Compacted History" in m.content)
    assert "## Carryover State" in summary
    assert "finish runtime continuity" in summary
    assert "src/clawagents/graph/agent_loop.py" in summary
    assert "/status now" in summary

    phases = [data["phase"] for kind, data in events if kind == "compact_progress"]
    assert phases[0] == "start"
    assert "end" in phases


async def test_subprocess_worker_backend_runs_headless_json_protocol():
    from clawagents.graph.coordinator import SubprocessWorkerBackend, WorkerTask

    script = (
        "import json,sys;"
        "payload=json.load(sys.stdin);"
        "print(json.dumps({'status':'done','result':'subprocess:' + payload['id'] + ':' + payload['prompt']}))"
    )
    backend = SubprocessWorkerBackend([sys.executable, "-c", script])
    task = WorkerTask(id="task_1", prompt="hello", tools=["read_file"], status="running")

    result = await backend.run(task, llm=None, tools=None, context_window=123)

    assert result.status == "done"
    assert result.result == "subprocess:task_1:hello"
    assert result.duration_s >= 0


def test_channel_messages_parse_commands_and_normalize_attachments():
    from clawagents.channels import ChannelMessage, channel_message_to_agent_input

    msg = ChannelMessage(
        channel_id="telegram",
        sender_id="u1",
        conversation_id="chat-1",
        body="/deploy staging now",
        timestamp=1.0,
        media=[{"url": "file:///tmp/log.txt", "mimeType": "text/plain", "filename": "log.txt"}],
    )

    assert msg.command is not None
    assert msg.command.name == "deploy"
    assert msg.command.args == "staging now"
    assert msg.media[0].mime_type == "text/plain"

    prompt = channel_message_to_agent_input(msg)
    assert "[Channel Command: deploy]" in prompt
    assert "staging now" in prompt
    assert "log.txt" in prompt
