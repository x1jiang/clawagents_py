"""Tests for Context Observatory session directory structure and package import/export."""

import json

from clawagents.context_observatory.events import (
    LLMCallEvent,
    MessageSnapshot,
)
from clawagents.context_observatory.store import EventStore
from clawagents.paths import get_context_observatory_dir


def test_get_context_observatory_dir(tmp_path, monkeypatch):
    """Verify get_context_observatory_dir creates and resolves directory."""
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))
    obs_dir = get_context_observatory_dir(create=True)
    assert obs_dir.exists()
    assert obs_dir.name == "context-observatory"
    assert obs_dir == tmp_path / ".clawagents" / "context-observatory"


def test_auto_save_session_directory(tmp_path, monkeypatch):
    """Verify auto_save creates session subdirectory with session.json and events.jsonl."""
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))

    store = EventStore()
    store.set_session_meta(model="test-model", context_window=128000, session_cost_usd=0.005)

    msg = MessageSnapshot(role="user", content_preview="Hello world", content_length=11, token_count=3, full_content="Hello world")
    event = LLMCallEvent(turn=1, timestamp=100.0, model="test-model", messages=[msg])
    store.append(event)

    session_path = store.auto_save(chat_id="test_session_123")
    assert session_path is not None
    assert session_path.exists()
    assert session_path.name == "session.json"

    session_dir = session_path.parent
    assert session_dir.name == "test_session_123"
    assert (session_dir / "events.jsonl").exists()

    # Load session back
    loaded = EventStore.load_from_json(session_dir)
    assert len(loaded.events) == 1
    assert loaded.session_meta.get("model") == "test-model"


def test_export_and_import_zip_package(tmp_path, monkeypatch):
    """Verify exporting session to ZIP and importing it into EventStore."""
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))

    store = EventStore()
    store.set_session_meta(model="gpt-5-nano", context_window=128000)

    msg = MessageSnapshot(role="assistant", content_preview="Analysis complete", content_length=17, token_count=4, full_content="Analysis complete")
    event = LLMCallEvent(turn=1, timestamp=200.0, model="gpt-5-nano", messages=[msg])
    store.append(event)

    # Export to ZIP bytes
    zip_bytes = store.export_package_zip()
    assert len(zip_bytes) > 0

    # Write to a zip file
    zip_file = tmp_path / "exported_session.zip"
    zip_file.write_bytes(zip_bytes)

    # Import back from zip
    imported_store = EventStore.load_from_json(zip_file)
    assert len(imported_store.events) == 1
    assert imported_store.events[0].model == "gpt-5-nano"


def test_auto_save_external_file_roundtrip(tmp_path, monkeypatch):
    """Large message bodies externalized on save must reload without NameError/TypeError."""
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))

    store = EventStore()
    store.set_session_meta(model="test-model", context_window=128000)
    huge = "x" * 60_000
    msg = MessageSnapshot(
        role="system",
        content_preview=huge[:200],
        content_length=len(huge),
        token_count=1000,
        full_content=huge,
    )
    store.append(LLMCallEvent(turn=1, timestamp=100.0, model="test-model", messages=[msg]))

    session_path = store.auto_save(chat_id="huge_session")
    assert session_path is not None
    raw = json.loads(session_path.read_text(encoding="utf-8"))
    saved_msg = raw["events"][0]["messages"][0]
    assert "external_file" in saved_msg
    assert "full_content" not in saved_msg

    loaded = EventStore.load_from_json(session_path.parent)
    assert len(loaded.events) == 1
    assert loaded.events[0].messages[0].full_content == huge


def test_list_history_scans_directories(tmp_path, monkeypatch):
    """Verify list_history detects session directories and legacy json files."""
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))

    store = EventStore()
    store.set_session_meta(model="claude-3-5-sonnet", session_cost_usd=0.02)
    store.append(LLMCallEvent(turn=1, timestamp=300.0, model="claude-3-5-sonnet"))
    store.auto_save(chat_id="demo_chat_99")

    entries = EventStore.list_history()
    assert len(entries) >= 1
    demo_entry = next((e for e in entries if "demo_chat_99" in e.get("filename", "")), None)
    assert demo_entry is not None
    assert demo_entry["is_directory"] is True
    assert demo_entry["model"] == "claude-3-5-sonnet"
