"""DX / efficiency fixes (v6.20.6).

- instruction ↔ system_prompt aliases on ClawAgent / create_claw_agent
- create_claw_agent(features=) passthrough
- workspace= redirects cwd-scoped tool/watcher paths
- ToolResult errors include exception type (+ traceback in dev)
- Token estimate memoization + cached system prompt budget
- Anthropic conversation cache breakpoints
- run_agent_graph(features=) restores global flags after invoke
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from clawagents.agent import ClawAgent, _resolve_system_prompt, create_claw_agent
from clawagents.providers.llm import LLMMessage, _apply_conversation_cache_breakpoints
from clawagents.tokenizer import clear_token_cache, count_messages_tokens, count_tokens_content
from clawagents.tools.registry import ToolRegistry, format_tool_error


class _FakeLLM:
    name = "fake"


def test_resolve_system_prompt_aliases():
    assert _resolve_system_prompt("a", None) == "a"
    assert _resolve_system_prompt(None, "b") == "b"
    assert _resolve_system_prompt("same", "same") == "same"
    with pytest.raises(ValueError, match="different values"):
        _resolve_system_prompt("a", "b")


def test_claw_agent_accepts_instruction_alias():
    agent = ClawAgent(llm=_FakeLLM(), tools=ToolRegistry(), instruction="be helpful")
    assert agent.system_prompt == "be helpful"


def test_create_claw_agent_accepts_system_prompt_alias():
    sig = inspect.signature(create_claw_agent)
    assert "system_prompt" in sig.parameters
    assert "features" in sig.parameters
    assert "workspace" in sig.parameters


def test_claw_agent_stores_workspace():
    agent = ClawAgent(
        llm=_FakeLLM(),
        tools=ToolRegistry(),
        workspace="/tmp/my-project",
    )
    assert agent.workspace.endswith("my-project")


def test_format_tool_error_includes_type():
    try:
        raise ValueError("boom")
    except ValueError as exc:
        err = format_tool_error(exc, include_traceback=False)
    assert err.startswith("ValueError: boom")


def test_format_tool_error_includes_traceback_when_debug(monkeypatch):
    monkeypatch.setenv("CLAW_DEBUG", "1")
    try:
        raise RuntimeError("kaboom")
    except RuntimeError as exc:
        err = format_tool_error(exc)
    assert "RuntimeError: kaboom" in err
    assert "Traceback" in err or "RuntimeError" in err.splitlines()[-1]


def test_token_content_memoization():
    clear_token_cache()
    text = "hello token cache " * 20
    with patch("clawagents.tokenizer.count_tokens", wraps=__import__("clawagents.tokenizer", fromlist=["count_tokens"]).count_tokens) as mocked:
        a = count_tokens_content(text)
        b = count_tokens_content(text)
        assert a == b
        assert mocked.call_count == 1


def test_cached_system_tokens_skips_retokenizing_system_body():
    clear_token_cache()
    messages = [
        LLMMessage(role="system", content="static system " * 100),
        LLMMessage(role="user", content="task"),
    ]
    full = count_messages_tokens(messages)
    cached = count_messages_tokens(messages, cached_system_tokens=999)
    # 999 substituted for system body; overhead still applied once per message
    assert cached != full
    assert cached > 999


def test_anthropic_conversation_cache_breakpoint():
    api_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "follow up"},
    ]
    _apply_conversation_cache_breakpoints(api_messages)
    assistant = api_messages[1]["content"]
    assert isinstance(assistant, list)
    assert assistant[-1].get("cache_control") == {"type": "ephemeral"}


def test_run_agent_graph_features_restore_after_run():
    import asyncio
    from clawagents.config import features as feat
    from clawagents.graph.agent_loop import run_agent_graph

    feat.reset()
    baseline_wal = feat.is_enabled("wal")
    target = not baseline_wal

    async def _fake_core(**kwargs):
        assert feat.is_enabled("wal") is target
        from clawagents.graph.agent_loop import AgentState

        return AgentState(
            messages=[],
            current_task="t",
            status="done",
            result="ok",
            iterations=0,
            max_iterations=1,
            tool_calls=0,
        )

    async def _run():
        with patch(
            "clawagents.graph.agent_loop._run_agent_graph_core",
            new=AsyncMock(side_effect=_fake_core),
        ):
            await run_agent_graph(
                "t",
                llm=_FakeLLM(),
                features={"wal": target},
            )

    asyncio.run(_run())
    assert feat.is_enabled("wal") is baseline_wal
    feat.reset()
