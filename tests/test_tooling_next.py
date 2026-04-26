import pytest

from clawagents.tools.registry import ToolRegistry, ToolResult, truncate_tool_output


class EchoTool:
    name = "echo"
    description = "Echo a message"
    parameters = {
        "message": {"type": "string", "description": "Message to echo", "required": True}
    }

    async def execute(self, args):
        return ToolResult(success=True, output=str(args.get("message", "")))


def test_truncate_tool_output_respects_small_custom_budgets_and_preserves_tail():
    out = truncate_tool_output("a" * 200 + "TAIL", max_chars=80)

    assert isinstance(out, str)
    assert len(out) <= 120
    assert "truncated" in out
    assert out.endswith("TAIL")


def test_tool_registry_exposes_inspectable_native_compatible_catalog():
    registry = ToolRegistry()
    registry.register(EchoTool())

    catalog = registry.inspect_tools()
    assert len(catalog) == 1
    assert catalog[0]["name"] == "echo"
    assert catalog[0]["parameters"] == EchoTool.parameters
    assert registry.to_native_schemas()[0].parameters == catalog[0]["parameters"]


def test_normalize_sandbox_manifest_validates_explicit_workspace_entries():
    from clawagents.sandbox.manifest import normalize_sandbox_manifest

    manifest = normalize_sandbox_manifest({
        "entries": {
            "repo": {"type": "git", "repo": "x1jiang/clawagents_py", "ref": "main", "target": "repo"},
            "cache": {"type": "path", "source": "/tmp/cache", "target": "cache", "read_only": True},
        },
        "env": {"PYTHONHASHSEED": "0"},
        "workdir": "repo",
    })

    assert len(manifest.entries) == 2
    assert manifest.entries[0].name == "repo"
    assert manifest.env["PYTHONHASHSEED"] == "0"
    with pytest.raises(ValueError, match="source"):
        normalize_sandbox_manifest({"entries": {"bad": {"type": "path", "source": ""}}})


@pytest.mark.asyncio
async def test_run_text_environment_records_observations_rewards_done_and_metrics():
    from clawagents.eval import run_text_environment

    turn = 0

    async def responder(messages):
        return f"reply:{messages[-1]['content']}"

    class Env:
        async def init(self):
            return {"observations": [{"role": "user", "content": "start"}], "metadata": {"id": "case-1"}}

        async def step(self, action):
            nonlocal turn
            turn += 1
            return {
                "observations": [{"role": "user", "content": f"turn-{turn}"}],
                "reward": 1 if "start" in action else 0,
                "done": turn >= 1,
                "metadata": {"turn": turn},
            }

        def get_metrics(self):
            return {"custom": 7}

    result = await run_text_environment(responder, Env())

    assert result.total_reward == 1
    assert len(result.steps) == 1
    assert result.metrics == {"custom": 7}


@pytest.mark.asyncio
async def test_tool_program_runs_bounded_read_only_sequence_with_substitutions():
    from clawagents.tools.tool_program import create_tool_program_tool

    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(create_tool_program_tool(registry, allowed_tools={"echo"}, max_steps=3))

    result = await registry.execute_tool("tool_program", {
        "steps": [
            {"id": "first", "tool": "echo", "args": {"message": "hello"}},
            {"tool": "echo", "args": {"message": "${first.output} world"}},
        ]
    })

    assert result.success is True
    assert result.output == "hello world"


@pytest.mark.asyncio
async def test_single_file_grep_caps_match_output(tmp_path):
    from clawagents.sandbox.local import LocalBackend
    from clawagents.tools.filesystem import GrepTool

    path = tmp_path / "many.txt"
    path.write_text("\n".join("needle" for _ in range(150)), encoding="utf-8")

    result = await GrepTool(LocalBackend(root=str(tmp_path))).execute({
        "path": str(path),
        "pattern": "needle",
    })

    assert result.success is True
    assert "100 match(es)" in result.output
    assert "truncated at 100" in result.output
