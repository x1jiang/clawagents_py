"""Regression tests for dead handoffs: model pin, exec streaming, SESSION_ID."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from clawagents.providers.llm import LLMProvider, LLMResponse
from clawagents.run_context import RunContext
from clawagents.tools.subagent import TaskTool, _pin_llm_model


class _StubLLM(LLMProvider):
    name = "stub"

    def __init__(self, model: str = "parent-model") -> None:
        self.model = model

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None) -> LLMResponse:
        return LLMResponse(content="ok", model=self.model, tokens_used=1)


def test_pin_llm_model_clones_provider():
    parent = _StubLLM("parent-model")
    pinned = _pin_llm_model(parent, "claude-4.5-haiku")
    assert pinned is not parent
    assert pinned.model == "claude-4.5-haiku"
    assert parent.model == "parent-model"


@pytest.mark.asyncio
async def test_subagent_model_pin_applied_to_child_llm():
    tool = TaskTool(llm=_StubLLM("parent-model"), tools=None)  # type: ignore[arg-type]
    captured: dict[str, Any] = {}

    async def fake_run_agent_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)

        class _State:
            status = "done"
            result = "ok"
            tool_calls = 0
            iterations = 1

        return _State()

    with patch(
        "clawagents.graph.agent_loop.run_agent_graph",
        new=fake_run_agent_graph,
    ):
        result = await tool.execute(
            {"description": "summarize", "model": "claude-4.5-haiku"},
            run_context=RunContext(),
        )

    assert result.success is True
    child_llm = captured["llm"]
    assert getattr(child_llm, "model", None) == "claude-4.5-haiku"


@pytest.mark.asyncio
async def test_run_agent_graph_wires_on_event_to_run_context():
    from clawagents.graph.agent_loop import run_agent_graph
    from clawagents.tools.registry import ToolRegistry

    events: list[tuple[str, dict]] = []

    def on_event(kind: str, data: dict | None = None) -> None:
        events.append((kind, data or {}))

    class _FastLLM(_StubLLM):
        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None) -> LLMResponse:
            return LLMResponse(content="done", model="stub", tokens_used=1)

    ctx = RunContext()
    with patch("clawagents.config.features.is_enabled", return_value=False):
        await run_agent_graph(
            task="hello",
            llm=_FastLLM(),
            tools=ToolRegistry(),
            max_iterations=1,
            streaming=False,
            on_event=on_event,
            run_context=ctx,
            session_end_tail=False,
        )

    assert ctx.on_event is on_event
    assert any(k == "context" for k, _ in events)


@pytest.mark.asyncio
async def test_run_agent_graph_sets_session_id_when_persistence_enabled():
    from clawagents.graph.agent_loop import run_agent_graph
    from clawagents.tools.registry import ToolRegistry

    class _FastLLM(_StubLLM):
        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None) -> LLMResponse:
            return LLMResponse(content="done", model="stub", tokens_used=1)

    ctx = RunContext()

    def _feat(name: str) -> bool:
        return name == "session_persistence"

    with patch("clawagents.config.features.is_enabled", side_effect=_feat):
        await run_agent_graph(
            task="hello",
            llm=_FastLLM(),
            tools=ToolRegistry(),
            max_iterations=1,
            streaming=False,
            run_context=ctx,
            session_end_tail=False,
        )

    assert ctx.session_id
    assert ctx._metadata.get("session_id") == ctx.session_id


@pytest.mark.asyncio
async def test_use_skill_expands_session_id_from_run_context(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_SKILL_SUBSTITUTIONS", "1")
    from clawagents.config import features as feat

    feat._resolved = None  # type: ignore[attr-defined]

    from clawagents.tools.skills import SkillStore, create_skill_tools

    root = tmp_path / "skills" / "sess-demo"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: sess-demo\ndescription: demo\n---\n"
        "Session token: ${SESSION_ID}\n",
        encoding="utf-8",
    )
    store = SkillStore()
    store.add_directory(tmp_path / "skills")
    store.reload()
    tools = {t.name: t for t in create_skill_tools(store)}

    ctx = RunContext()
    ctx.session_id = "sess-abc-123"
    ctx._metadata["session_id"] = "sess-abc-123"

    result = await tools["use_skill"].execute({"name": "sess-demo"}, run_context=ctx)
    assert result.success, result.error
    assert "sess-abc-123" in result.output
    assert "${SESSION_ID}" not in result.output
