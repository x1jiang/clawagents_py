"""Tests for v6.17 Grok-Build parity pack."""

from __future__ import annotations

from pathlib import Path



def test_smart_memory_dedup_and_access_boost(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_SMART_MEMORY", "1")
    monkeypatch.setenv("CLAW_FEATURE_HYBRID_MEMORY_SEARCH", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.smart_store import (
        MemoryChunk,
        MemorySearchConfig,
        SmartMemoryStore,
        ingest_text,
    )

    store = SmartMemoryStore(tmp_path)
    c1 = MemoryChunk(
        chunk_id="a",
        path="notes.md",
        content="Use workspace sandbox for file writes.",
        source="session",
    )
    assert store.upsert(c1) is True
    assert store.upsert(c1) is False  # exact hash dedup
    assert ingest_text("Unique fact about alpha", path="f.md", workspace=tmp_path)

    # Access boost: hit same chunk twice via search
    hits = store.hybrid_search(
        "sandbox file writes",
        MemorySearchConfig(min_score=0.0, max_results=5, mmr_enabled=False),
    )
    assert hits
    before = store._load_chunk(hits[0].chunk_id).access_count
    store.hybrid_search("sandbox", MemorySearchConfig(min_score=0.0))
    after = store._load_chunk(hits[0].chunk_id).access_count
    assert after >= before
    store.close()


def test_temporal_decay_evergreen_exempt(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_SMART_MEMORY", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.smart_store import MemorySearchConfig, SmartMemoryStore
    import time

    store = SmartMemoryStore(tmp_path)
    cfg = MemorySearchConfig(temporal_decay=True, half_life_days=7.0)
    now = time.time()
    old = now - 30 * 86400
    d_sess = store._temporal_decay("session", old, cfg, now)
    d_cur = store._temporal_decay("curated", old, cfg, now)
    assert d_cur == 1.0
    assert d_sess < 0.2
    store.close()


def test_dream_gates_and_process(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_MEMORY_DREAM", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.dream import (
        DreamConfig,
        append_session_log,
        check_dream_gates,
        process_dream_response,
    )

    assert process_dream_response("NO_REPLY") is None
    assert process_dream_response("## Facts\n- a") is not None
    for i in range(3):
        append_session_log(f"session note {i}\n## Detail\nfoo", workspace=tmp_path, stem=f"s{i}")
    gate = check_dream_gates(tmp_path, DreamConfig(min_hours=0, min_sessions=3))
    assert hasattr(gate, "sessions")


def test_memory_flush_window_and_gate(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_MEMORY_FLUSH", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.memory_flush import should_flush, select_flush_window, process_flush_response
    from clawagents.providers.llm import LLMMessage

    assert should_flush(90_000, 100_000, compaction_cycle=1, workspace=tmp_path)
    assert not should_flush(10, 100_000, workspace=tmp_path)
    msgs = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="assistant", content="yo"),
    ]
    win = select_flush_window(msgs, recent_n=2)
    assert all(m.role != "system" for m in win)
    assert process_flush_response("NO_REPLY") is None


def test_doom_loop_confident_resample():
    from clawagents.doom_loop import (
        DoomLoopRecoveryPolicy,
        DoomLoopState,
        detect_tail_repetition,
        is_confident_trigger,
        should_resample,
    )

    text = "\n".join(["I should check the logs again"] * 5)
    sig = detect_tail_repetition(text, channel="thinking")
    assert sig is not None
    assert is_confident_trigger(sig)
    state = DoomLoopState()
    assert should_resample(sig, state, DoomLoopRecoveryPolicy(max_retries=2))
    state.retry_count = 2
    assert not should_resample(sig, state, DoomLoopRecoveryPolicy(max_retries=2))


def test_pty_key_parsing():
    from clawagents.tools.pty_session import PtySession

    assert PtySession.parse_keys("ab<CR>") == b"ab\r"
    assert PtySession.parse_keys("<Esc>") == b"\x1b"
    assert PtySession.parse_keys("<C-c>") == bytes([ord("c") & 0x1F])


def test_structured_output_openai_envelope():
    from clawagents.structured_output import openai_chat_response_format, anthropic_output_format

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    oai = openai_chat_response_format(schema)
    assert oai["type"] == "json_schema"
    assert oai["json_schema"]["strict"] is True
    ant = anthropic_output_format(schema)
    assert ant["type"] == "json_schema"
    assert "schema" in ant


def test_compaction_segments_and_failure_class(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_COMPACTION_SEGMENTS", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.compaction_segments import (
        write_segment,
        classify_compaction_failure,
        should_compact_steps_after_history,
        separate_prior_user_queries,
        wrap_user_query,
    )

    seg = write_segment("## Turn\nhello world sandbox", workspace=tmp_path, turns=3)
    assert (tmp_path / ".clawagents" / "compaction" / "INDEX.md").is_file()
    assert (tmp_path / ".clawagents" / "compaction" / f"segment_{seg.index:03d}.md").is_file()
    assert classify_compaction_failure("context length exceeded").kind == "deterministic"
    assert classify_compaction_failure("timeout waiting").kind == "transient"
    assert should_compact_steps_after_history(100, 40) is True
    assert should_compact_steps_after_history(100, 10) is False
    prior, body = separate_prior_user_queries(
        "<user_query>\nold\n</user_query>\nnew body"
    )
    assert "old" in prior
    assert "new body" in body
    assert "<user_query>" in wrap_user_query("x")


def test_hook_ssrf_and_exit2_deny():
    from clawagents.hooks.taxonomy import (
        validate_hook_url,
        parse_blocking_result,
        is_blocked_ip,
        DENY_EXIT_CODE,
    )

    assert is_blocked_ip("10.0.0.1")
    assert not is_blocked_ip("127.0.0.1")
    ok, reason = validate_hook_url("http://example.com/hook")
    assert ok is False and reason == "https_only"
    d = parse_blocking_result('{"decision":"deny","reason":"nope"}', 0)
    assert d.allowed is False
    d2 = parse_blocking_result("", DENY_EXIT_CODE)
    assert d2.allowed is False


def test_hunk_rewind_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_SESSION_REWIND", "1")
    monkeypatch.setenv("CLAW_FEATURE_HUNK_WATCHER", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.hunk_watcher import HunkWatcher

    f = tmp_path / "a.txt"
    f.write_text("v1\n", encoding="utf-8")
    w = HunkWatcher(tmp_path)
    w.record_agent_write("a.txt", "v1\n", prompt_index=1)
    w.snapshot_turn(1, user_text="first")
    f.write_text("v2\n", encoding="utf-8")
    w.record_agent_write("a.txt", "v2\n", prompt_index=2)
    w.snapshot_turn(2, user_text="second")
    result = w.rewind_to_prompt(1)
    assert result["ok"] is True
    assert f.read_text(encoding="utf-8") == "v1\n"
    assert result.get("user_text") == "first"
    assert result.get("truncate_to_user_text") == "first"


def test_hunk_attribution_agent_and_external(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_HUNK_WATCHER", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.attributed_hunks import (
        agent_edit_attribution,
        external_edit_attribution,
        list_hunks,
        refresh_file_hunks,
    )

    f = tmp_path / "a.txt"
    f.write_text("base\n", encoding="utf-8")
    refresh_file_hunks("a.txt", workspace=tmp_path, seed_baseline_if_missing=True)
    f.write_text("agent edit\n", encoding="utf-8")
    refresh_file_hunks(
        "a.txt",
        workspace=tmp_path,
        turn_index=3,
        tool="write",
        source="agent",
        attribution=agent_edit_attribution(3),
        seed_baseline_if_missing=False,
    )
    rows = list_hunks(workspace=tmp_path, path="a.txt")
    assert rows
    assert rows[0].attribution == "AgentEdit3"
    assert rows[0].source == "agent"

    f.write_text("external edit\n", encoding="utf-8")
    refresh_file_hunks(
        "a.txt",
        workspace=tmp_path,
        source="external_on_agent",
        attribution=external_edit_attribution(on_agent_file=True),
        seed_baseline_if_missing=False,
    )
    rows2 = list_hunks(workspace=tmp_path, path="a.txt")
    assert any(h.attribution == "ExternalEditOnAgentFile" for h in rows2)


def test_rewind_snapshot_conversation_marker(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_SESSION_REWIND", "1")
    from clawagents.config.features import reset

    reset()
    from clawagents.memory.hunk_watcher import HunkWatcher

    w = HunkWatcher(tmp_path)
    w.snapshot_turn(
        1,
        user_text="hello",
        message_count=4,
        conversation_marker=[
            {"role": "user", "preview": "hello"},
            {"role": "assistant", "preview": "hi"},
        ],
    )
    result = w.rewind_to_prompt(1)
    assert result["message_count"] == 4
    assert result["conversation_marker"][0]["role"] == "user"


def test_bwrap_secret_overlay_paths(tmp_path: Path):
    from clawagents.sandbox.profiles import _resolve_secret_overlay_paths

    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "token.pem").write_text("x", encoding="utf-8")
    paths = _resolve_secret_overlay_paths(
        str(tmp_path),
        (".env", "**/*.pem"),
    )
    assert any(p.endswith(".env") for p in paths)
    assert any(p.endswith("token.pem") for p in paths)
    # Missing .env path still included for fail-closed bind
    (tmp_path / ".env").unlink()
    paths2 = _resolve_secret_overlay_paths(str(tmp_path), (".env",))
    assert any(p.endswith(".env") for p in paths2)


def test_sandbox_project_add_only(tmp_path: Path):
    from clawagents.sandbox.profiles import load_project_sandbox_toml, get_profile
    import json

    cfg = tmp_path / ".clawagents" / "sandbox.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps(
            {
                "profiles": {
                    "workspace": {"backend": "local"},  # conflict — ignored
                    "lab": {"backend": "local", "network": False},
                }
            }
        ),
        encoding="utf-8",
    )
    # chdir so loader finds it
    import os

    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        found = load_project_sandbox_toml(tmp_path)
        assert "lab" in found
        assert "workspace" not in found  # cannot redefine builtin
        p = get_profile("lab")
        assert p.network is False
    finally:
        os.chdir(old)
