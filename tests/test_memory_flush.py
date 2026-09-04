"""Hermetic tests for pre-compaction memory flush (Grok memory_flush parity)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clawagents.config.features import temporary_overrides
from clawagents.memory.memory_flush import (
    FlushConfig,
    _format_window,
    process_flush_response,
    run_memory_flush,
    select_flush_window,
    should_flush,
)
from clawagents.providers.llm import LLMMessage


def test_should_flush_threshold_and_cycle(tmp_path: Path):
    with temporary_overrides({"memory_flush": True}):
        cfg = FlushConfig(soft_threshold_tokens=1000, compact_pct=0.8)
        # 10,000 * 0.8 = 8,000; threshold = 8,000 - 1,000 = 7,000
        assert not should_flush(6999, 10000, workspace=tmp_path, config=cfg, compaction_cycle=1)
        assert should_flush(7000, 10000, workspace=tmp_path, config=cfg, compaction_cycle=1)

        # When feature is disabled
        with temporary_overrides({"memory_flush": False}):
            assert not should_flush(8000, 10000, workspace=tmp_path, config=cfg, compaction_cycle=1)


def test_select_flush_window_drops_system_and_keeps_user_boundary():
    messages = [
        LLMMessage(role="system", content="System instruction"),
        LLMMessage(role="user", content="First question"),
        LLMMessage(role="assistant", content="First answer"),
        LLMMessage(role="user", content="Second question"),
        LLMMessage(role="assistant", content="Second answer"),
    ]
    window = select_flush_window(messages, recent_n=2)
    assert len(window) >= 2
    # Ensure system prompt is dropped
    assert all(m.role != "system" for m in window)
    # Starts on user boundary
    assert window[0].role == "user"


def test_process_flush_response_validates_markdown_and_no_reply():
    assert process_flush_response("NO_REPLY") is None
    assert process_flush_response("   no_reply  ") is None
    assert process_flush_response("Just plain text with no headers or bullets") is None
    valid = "## Architecture\n- Use SQLite for session persistence\n"
    assert process_flush_response(valid) == valid.strip()


def test_run_memory_flush_stores_and_logs(tmp_path: Path):
    with temporary_overrides({"memory_flush": True, "smart_memory": True}):
        messages = [
            LLMMessage(role="user", content="Let's use PostgreSQL for auth and Redis for caching"),
            LLMMessage(role="assistant", content="Understood."),
        ]

        async def _mock_llm(prompt: str) -> str:
            assert "PostgreSQL" in prompt
            return "## Decisions\n- Auth database is PostgreSQL\n- Cache is Redis\n"

        async def _run():
            outcome = await run_memory_flush(
                messages,
                _mock_llm,
                workspace=tmp_path,
                compaction_cycle=1,
            )
            assert outcome.status in ("accepted", "stored")
            assert outcome.stored is True

            # Verify session log file was written
            log_dir = tmp_path / ".clawagents" / "memory-sessions"
            assert log_dir.is_dir()
            logs = list(log_dir.glob("flush_*.md"))
            assert len(logs) == 1
            content = logs[0].read_text(encoding="utf-8")
            assert "Auth database is PostgreSQL" in content

        asyncio.run(_run())
