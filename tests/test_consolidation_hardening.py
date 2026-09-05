"""Integration tests for consolidated search, tools, lessons, and output paths."""

from __future__ import annotations

import asyncio
import json


def test_lesson_utilities_live_in_lessons_module():
    from clawagents.trajectory import lesson_promotion, lessons

    bullet = "- Always grep before reading large files"
    assert lessons.normalize_lesson(bullet) == lessons.normalize_lesson("  - Always grep before reading large files  ")
    assert lessons.lesson_key(bullet) == lessons.lesson_key("- always grep before reading large files")
    assert bullet.lstrip("- ").strip() in lessons.parse_lesson_bullets(f"intro\n{bullet}\n")
    slug = lessons.slugify_lesson_name(bullet)
    assert slug and "-" in slug
    # promotion module imports these — no duplicate definitions
    assert lesson_promotion.lesson_key is lessons.lesson_key


def test_search_history_session_filter_and_json_format(tmp_path):
    from clawagents.providers.llm import LLMMessage
    from clawagents.session.backends import SQLiteSession
    from clawagents.session.history_search import format_search_history_response, search_history
    from clawagents.tools.search_history import create_search_history_tool

    async def _run():
        db = tmp_path / ".clawagents" / "sessions.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        s1 = SQLiteSession("only-alpha", db_path=db)
        s2 = SQLiteSession("only-beta", db_path=db)
        await s1.add_items([LLMMessage(role="user", content="alpha canary token")])
        await s2.add_items([LLMMessage(role="user", content="beta canary token")])

        filtered = search_history("canary", workspace=tmp_path, session_id="only-alpha", limit=10)
        assert filtered
        assert all(h.session_id == "only-alpha" for h in filtered)

        tool = create_search_history_tool(workspace=str(tmp_path))
        json_out = await tool.execute({"query": "canary", "session_id": "only-beta", "format": "json", "include_jsonl": False})
        assert json_out.success
        payload = json.loads(json_out.output)
        assert payload["query"] == "canary"
        assert payload["hits"]
        assert all(h["session_id"] == "only-beta" for h in payload["hits"])

        text_out = format_search_history_response("canary", filtered, as_json=False)
        assert "Found" in text_out
        assert "only-alpha" in text_out

    asyncio.run(_run())


def test_search_history_jsonl_archive(tmp_path):
    from clawagents.session.history_search import search_history

    sessions_dir = tmp_path / ".clawagents" / "sessions"
    sessions_dir.mkdir(parents=True)
    log = sessions_dir / "jsonl-session.jsonl"
    log.write_text(
        json.dumps({"type": "assistant_message", "content": "jsonl unique marker xyzzy", "ts": 1.0}) + "\n",
        encoding="utf-8",
    )
    hits = search_history("xyzzy", workspace=tmp_path, include_jsonl=True, limit=5)
    assert any(h.source == "jsonl" and h.session_id == "jsonl-session" for h in hits)


def test_skill_workshop_tool_end_to_end(tmp_path):
    from clawagents.tools.skill_workshop import create_skill_workshop_tool

    skills = tmp_path / "skills"
    skills.mkdir()
    tool = create_skill_workshop_tool(workspace=str(tmp_path), skills_dir=str(skills))

    async def _run():
        created = await tool.execute(
            {
                "action": "create",
                "name": "consolidated-skill",
                "description": "From tool",
                "body": "# Consolidated\nDo things.",
                "goal": "test",
            }
        )
        assert created.success, created.error
        data = json.loads(created.output)
        proposal_id = data["id"]
        assert data["status"] == "pending"

        applied = await tool.execute({"action": "apply", "proposal_id": proposal_id})
        assert applied.success, applied.error
        assert (skills / "consolidated-skill" / "SKILL.md").is_file()

        listed = await tool.execute({"action": "list"})
        assert listed.success
        listed_data = json.loads(listed.output)
        assert any(p.get("id") == proposal_id for p in listed_data.get("proposals", []))

    asyncio.run(_run())


def test_snippet_shared_between_session_and_history_search(tmp_path):
    from clawagents.session.backends import SQLiteSession
    from clawagents.session.history_search import search_history
    from clawagents.session.search import search_session_messages
    from clawagents.providers.llm import LLMMessage
    import sqlite3

    async def _run():
        db_path = tmp_path / ".clawagents" / "sessions.db"
        session = SQLiteSession("snip", db_path=db_path)
        needle = "UNIQUE_SNIPPET_NEEDLE_42"
        await session.add_items([LLMMessage(role="user", content=f"prefix {needle} suffix")])

        conn = sqlite3.connect(str(db_path))
        try:
            in_session = search_session_messages(conn, "snip", needle, limit=5)
        finally:
            conn.close()
        cross = search_history(needle, workspace=tmp_path, limit=5, include_jsonl=False)
        assert in_session and cross
        assert needle in in_session[0].snippet or f"[{needle}]" in in_session[0].snippet
        assert needle in cross[0].snippet or f"[{needle}]" in cross[0].snippet

    asyncio.run(_run())


def test_create_claw_agent_registers_consolidated_tools(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from clawagents.agent import create_claw_agent
    from clawagents.providers.llm import LLMResponse, LLMProvider

    class FakeLLM(LLMProvider):
        name = "fake"

        async def chat(self, messages, **kwargs):
            return LLMResponse(content="ok", role="assistant")

    agent = create_claw_agent(FakeLLM(), memory=[], skills=[])
    tool_names = {t["name"] for t in agent.tools.inspect_tools()}
    assert "search_history" in tool_names
    assert "skill_workshop" in tool_names
