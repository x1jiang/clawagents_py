"""Tests for search_history, output_format, and PTRL lesson promotion."""

from __future__ import annotations

import asyncio
import json


def test_cross_session_history_search(tmp_path):
    from clawagents.providers.llm import LLMMessage
    from clawagents.session.backends import SQLiteSession
    from clawagents.session.history_search import search_history

    async def _run():
        db = tmp_path / ".clawagents" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        s1 = SQLiteSession("alpha", db_path=db)
        s2 = SQLiteSession("beta", db_path=db)
        await s1.add_items([LLMMessage(role="user", content="fix pytest timeout in api tests")])
        await s2.add_items([LLMMessage(role="assistant", content="grep logs for pytest failure")])

        hits = search_history("pytest", workspace=tmp_path, limit=10)
        session_ids = {h.session_id for h in hits}
        assert "alpha" in session_ids
        assert "beta" in session_ids

    asyncio.run(_run())


def test_search_history_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.providers.llm import LLMMessage
    from clawagents.session.backends import SQLiteSession
    from clawagents.tools.search_history import create_search_history_tool

    async def _run():
        db = tmp_path / ".clawagents" / "sessions.db"
        session = SQLiteSession("s1", db_path=db)
        await session.add_items([LLMMessage(role="user", content="deploy canary to staging")])

        tool = create_search_history_tool(workspace=str(tmp_path))
        out = await tool.execute({"query": "canary", "include_jsonl": False})
        assert out.success
        assert "staging" in out.output.lower()

    asyncio.run(_run())


def test_serialize_agent_state():
    from clawagents.graph.agent_loop import AgentState
    from clawagents.output_format import OutputFormat, serialize_agent_state

    state = AgentState(
        messages=[],
        current_task="hi",
        status="done",
        result="hello world",
        iterations=2,
        max_iterations=10,
        tool_calls=1,
    )
    payload = serialize_agent_state(state)
    assert payload["status"] == "done"
    assert payload["result"] == "hello world"
    assert payload["iterations"] == 2
    assert OutputFormat.JSON.value == "json"


def test_lesson_promotion_creates_workshop_proposal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.trajectory.lesson_promotion import maybe_promote_recurring_lessons

    lesson = "- Prefer grep before reading large log files"
    md = f"{lesson}\n"
    created = []
    for _ in range(3):
        created = maybe_promote_recurring_lessons(md, task="debug logs", min_occurrences=3)
    assert created
    assert created[0].get("status") == "pending"
    index = json.loads((tmp_path / ".clawagents" / "lesson-index.json").read_text(encoding="utf-8"))
    assert any(e.get("promoted_proposal_id") for e in index["lessons"].values())
