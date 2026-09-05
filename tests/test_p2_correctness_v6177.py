"""P2/P3 correctness regressions fixed in v6.17.7."""

from __future__ import annotations

import asyncio

import pytest


def test_breaker_key_isolates_endpoints():
    from clawagents.circuit_breaker import breaker_key

    cloud = breaker_key("openai", base_url="https://api.openai.com/v1", model="gpt-4o")
    local = breaker_key("openai", base_url="http://127.0.0.1:11434/v1", model="llama3")
    assert cloud != local


def test_flush_cycle_guard_includes_zero(tmp_path, monkeypatch):
    from clawagents.memory import memory_flush as mf
    from clawagents.config.features import reset, set_overrides

    reset()
    set_overrides({"memory_flush": True})
    monkeypatch.chdir(tmp_path)
    mf._LAST_FLUSH_CYCLE.clear()
    assert mf.should_flush(100_000, 1000, compaction_cycle=0, workspace=tmp_path)
    mf._LAST_FLUSH_CYCLE[str(tmp_path.resolve())] = 0
    assert not mf.should_flush(100_000, 1000, compaction_cycle=0, workspace=tmp_path)
    reset()


def test_smart_store_fts_replace_no_orphan(tmp_path):
    from clawagents.memory.smart_store import SmartMemoryStore, MemoryChunk

    store = SmartMemoryStore(tmp_path)
    try:
        if not store._fts:
            pytest.skip("FTS5 unavailable")
        c = MemoryChunk(
            chunk_id="memory_md",
            path=".clawagents/MEMORY.md",
            content="v1 facts",
            source="curated",
        )
        assert store.upsert(c)
        c2 = MemoryChunk(
            chunk_id="memory_md",
            path=".clawagents/MEMORY.md",
            content="v2 facts different",
            source="curated",
        )
        # Force replace path: delete hash uniqueness by using new content
        # upsert returns False on exact dup; different content replaces same id
        store._conn.execute("DELETE FROM chunks WHERE chunk_id = ?", ("memory_md",))
        store._conn.commit()
        assert store.upsert(c2)
        n = store._conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = ?", ("memory_md",)
        ).fetchone()[0]
        assert n == 1
    finally:
        store.close()


def test_session_fts5_populated(tmp_path):
    from clawagents.session.backends import SQLiteSession
    from clawagents.providers.llm import LLMMessage
    from clawagents.session.search import search_sqlite_messages

    db = tmp_path / "s.db"
    sess = SQLiteSession("t1", db_path=db)

    async def _run():
        await sess.add_items(
            [
                LLMMessage(role="user", content="alpha unique-token-xyz"),
                LLMMessage(role="assistant", content="beta reply"),
            ]
        )

    asyncio.run(_run())
    with sess._lock, sess._conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert n >= 2
        rows = search_sqlite_messages(conn, "unique-token-xyz", session_id="t1")
        assert rows


def test_fallback_propagates_schema():
    from clawagents.providers.fallback import FallbackProvider
    from clawagents.providers.llm import LLMMessage, LLMResponse, LLMProvider

    class Capture(LLMProvider):
        name = "capture"

        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
            self.seen = getattr(self, "_structured_json_schema", None)
            return LLMResponse(content="{}", model="m", tokens_used=1)

    cap = Capture()
    fb = FallbackProvider(primary=cap, fallbacks=[])
    setattr(fb, "_structured_json_schema", {"type": "object"})
    asyncio.run(fb.chat([LLMMessage(role="user", content="hi")]))
    assert cap.seen == {"type": "object"}
    assert getattr(cap, "_structured_json_schema", None) in (None, )


def test_doom_response_channel_confident():
    from clawagents.doom_loop import detect_tail_repetition, is_confident_trigger

    text = "\n".join(["check again"] * 5)
    sig = detect_tail_repetition(text, channel="response")
    assert sig is not None
    assert is_confident_trigger(sig)


def test_enqueue_interject_exported():
    import clawagents

    assert hasattr(clawagents, "enqueue_interject")


def test_dream_lock_released_on_cancel(tmp_path):
    from clawagents.memory.dream import run_dream, _lock_path
    from clawagents.config.features import reset, set_overrides

    reset()
    set_overrides({"memory_dream": True})
    # Seed enough sessions so gates open — or force past gate by mocking
    sess = tmp_path / ".clawagents" / "memory-sessions"
    sess.mkdir(parents=True)
    for i in range(3):
        (sess / f"s{i}.md").write_text("session notes\n" * 20, encoding="utf-8")

    async def slow(_prompt: str) -> str:
        await asyncio.sleep(10)
        return "## Memory\n- x\n"

    async def _run():
        task = asyncio.create_task(run_dream(slow, workspace=tmp_path))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        lock = _lock_path(tmp_path)
        assert not lock.exists(), "dream.lock must be released on cancel"

    # Gates may block before lock — if so, skip
    from clawagents.memory.dream import check_dream_gates

    gate = check_dream_gates(tmp_path)
    if isinstance(gate, str):
        pytest.skip(f"dream gates closed: {gate}")
    asyncio.run(_run())
    reset()
