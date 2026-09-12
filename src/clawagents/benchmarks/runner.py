"""Bounded real-loop runner; filesystem and command execution stay in memory."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import time
from typing import Any, cast

from .efficiency import ARMS, TOKEN_FIELDS, cost_from_usage, digest, runtime_digest, validate_manifest, verify_source


async def trial(manifest: dict[str, Any], task: dict[str, Any], repeat: int, arm: str,
                *, allow_live: bool = False) -> dict[str, Any]:
    from clawagents.agent import ClawAgent
    from clawagents.config.features import _FEATURE_DEFAULTS
    from clawagents.providers.llm import LLMProvider, LLMResponse, NativeToolCall
    from clawagents.run_context import RunContext
    from clawagents.sandbox.backend import ExecResult
    from clawagents.sandbox.memory import InMemoryBackend
    from clawagents.tools.exec import create_exec_tools
    from clawagents.tools.filesystem import create_filesystem_tools
    from clawagents.tools.registry import Tool, ToolRegistry
    from clawagents.tools.retrieve_tool_result import RetrieveToolResultTool

    live = manifest["provider"] != "fixture"
    if live and not allow_live:
        raise ValueError("live provider execution requires --allow-live")
    if runtime_digest() != manifest["arms"][arm]["runtime_sha256"]:
        raise ValueError("running package differs from frozen arm source")
    full_sends = manifest["arms"][arm]["observation_full_sends"]
    context: RunContext[Any] = RunContext(skip_memory=True)
    if full_sends and not hasattr(context, "observation_full_sends"):
        raise ValueError("this runtime does not implement observation_full_sends")
    if hasattr(context, "observation_full_sends"):
        context.observation_full_sends = full_sends

    async def execute_stub(command: str, **kwargs: Any) -> ExecResult:
        if command.strip() != "verify":
            return ExecResult(stdout="", stderr="Only the fixed 'verify' command is available.", exit_code=1)
        passed = verify_source(backend.snapshot().get("answer.py", ""), task)
        output = ("PASS numeric contract\n" if passed else "FAIL numeric contract; edit answer.py to satisfy the task\n")
        return ExecResult(stdout=output, stderr="", exit_code=0 if passed else 1)

    backend = InMemoryBackend("/benchmark", exec_stub=execute_stub)
    backend.seed({"answer.py": task["source"],
                  "notes.txt": "".join(f"Archive note {i}: preserve the public numeric function signature.\n" for i in range(900)) + task["prompt"]})
    registry = ToolRegistry()
    for tool in create_filesystem_tools(backend) + create_exec_tools(backend) + [RetrieveToolResultTool()]:
        registry.register(cast(Tool, tool))
    calls: list[Any] = []
    response_models: list[str] = []
    reported_usage = True

    class Fixture(LLMProvider):
        name = "fixture"
        model = manifest["model"]

        async def chat(self, messages: list[Any], *args: Any, **kwargs: Any) -> LLMResponse:
            sequence = [
                ("read_file", {"path": "notes.txt", "tier": "L2"}),
                ("read_file", {"path": "answer.py"}),
                ("execute", {"command": "verify"}),
                ("write_file", {"path": "answer.py", "content": task["solution"], "then_run": {"command": "verify"}}),
                ("read_file", {"path": "answer.py"}),
            ]
            index = len(calls)
            content = "Completed and verified." if index >= len(sequence) else ""
            tool_calls = [] if index >= len(sequence) else [NativeToolCall(*sequence[index], tool_call_id=f"fixture-{index}")]
            # Deliberately labeled estimates, using actual projected messages.
            prompt = sum(len(str(m.content)) for m in messages) // 4
            return LLMResponse(content=content, model=self.model, tokens_used=prompt + 32,
                               prompt_tokens=prompt, tool_calls=tool_calls)

    row: dict[str, Any] = {
        "manifest_sha256": manifest["sha256"], "runtime_sha256": runtime_digest(),
        "task_sha256": digest(task), "provider": manifest["provider"], "model": manifest["model"],
        "case": task["id"], "repeat": repeat, "arm": arm,
        "mode": "live" if live else "fixture", "passed": False,
        "provider_retries": None, "error_type": None,
    }
    started = time.perf_counter()
    provider = None
    try:
        if live:
            from clawagents.config.config import EngineConfig
            from clawagents.providers.llm import create_provider
            # Freeze all serving choices; only secrets come from the environment.
            config = EngineConfig(openai_base_url=manifest["base_url"],
                                  openai_model=manifest["model"], streaming=False,
                                  openai_wire_api="responses", openai_api_type="",
                                  openai_api_version="", openai_ssl_verify=True,
                                  **manifest["serving"])
            provider = create_provider(manifest["model"], config, provider_hint="openai")
        else:
            provider = Fixture()
        original_chat = provider.chat

        async def traced_chat(messages: list[Any], *args: Any, **kwargs: Any) -> Any:
            nonlocal reported_usage
            response = await original_chat(messages, *args, **kwargs)
            calls.append(response)
            response_models.append(response.model)
            if response.tokens_used <= 0 or response.prompt_tokens <= 0:
                reported_usage = False
            return response

        setattr(provider, "chat", traced_chat)
        features = {key: False for key in _FEATURE_DEFAULTS}
        features.update(micro_compact=True, aggressive_tool_crush=True, compact_tool_pair_safe=True)
        with tempfile.TemporaryDirectory(prefix="claw-efficiency-") as workspace:
            agent = ClawAgent(provider, registry, streaming=False, workspace=workspace,
                              features=features, max_iterations=manifest["max_iterations"],
                              instruction="Solve the small numeric coding task using the provided in-memory filesystem. Read notes.txt with tier L2 and answer.py. Only execute command 'verify' is supported; it checks the numeric contract. You can use then_run={command: verify} on edits. No network, imports or arbitrary commands are available.")
            state = await asyncio.wait_for(agent.invoke(task["prompt"], run_context=context,
                                                       max_iterations=manifest["max_iterations"]),
                                           timeout=manifest["timeout_seconds"])
            row["passed"] = state.status == "done" and verify_source(backend.snapshot().get("answer.py", ""), task)
            row["status"] = state.status
    except Exception as exc:
        # No provider exception text: it can include endpoint credentials.
        row["error_type"] = type(exc).__name__
        reported_usage = False
    finally:
        row["latency_seconds"] = time.perf_counter() - started
        client = getattr(provider, "client", None)
        if client is not None and hasattr(client, "close"):
            try:
                await client.close()
            except Exception as exc:
                row["close_error_type"] = type(exc).__name__
                reported_usage = False
    usage = context.usage.to_dict()
    row["usage"] = {key: usage[key] for key in TOKEN_FIELDS}
    row["request_usage"] = [{key: request[key] for key in TOKEN_FIELDS[:-1]} for request in usage["per_request"]]
    row["response_models"] = sorted(set(response_models))
    row["measurement_complete"] = reported_usage and not row["error_type"]
    row["estimated_cost_usd"] = cost_from_usage(usage, manifest["prices_per_million"]) if live else None
    row["token_measurement"] = "provider_reported" if live else "estimated_characters_divided_by_four"
    row["then_run_requested"] = sum(1 for response in calls for call in response.tool_calls or [] if call.args.get("then_run"))
    row["then_run_completed"] = context.efficiency.get("round_trips_avoided", 0)
    row["answer_sha256"] = digest(backend.snapshot().get("answer.py", ""))
    row["efficiency"] = context.efficiency
    return row


async def run(manifest: dict[str, Any], suite: dict[str, Any], *, split: str, output: Path,
              arm: str = "both", allow_live: bool = False) -> list[dict[str, Any]]:
    validate_manifest(manifest, suite)
    if split not in ("development", "heldout") or arm not in (*ARMS, "both"):
        raise ValueError("invalid split/arm")
    if manifest["provider"] != "fixture" and not allow_live:
        raise ValueError("live provider execution requires --allow-live")
    # Never overwrite earlier held-out outcomes; partial runs remain inspectable.
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("x", encoding="utf-8") as stream:
        for repeat in range(manifest["repeats"]):
            for task in suite["tasks"]:
                if task["split"] != split:
                    continue
                arms = list(ARMS) if arm == "both" else [arm]
                if repeat % 2:
                    arms.reverse()
                for selected in arms:
                    row = await trial(manifest, task, repeat, selected, allow_live=allow_live)
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
    return rows
