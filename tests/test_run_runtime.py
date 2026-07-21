"""Tests for run-scoped infrastructure extracted from ``agent_loop``."""

from __future__ import annotations

import pytest

from clawagents.graph.run_runtime import SessionMessageJournal
from clawagents.providers.llm import LLMMessage


class _Session:
    def __init__(self, items: list[LLMMessage]) -> None:
        self.items = list(items)
        self.saved: list[LLMMessage] = []

    async def get_items(self, *, limit: int | None = None) -> list[LLMMessage]:
        return self.items[-limit:] if limit is not None else list(self.items)

    async def add_items(self, items: list[LLMMessage]) -> None:
        self.saved.extend(items)


@pytest.mark.asyncio
async def test_session_journal_persists_current_task_not_preloaded_user() -> None:
    session = _Session([LLMMessage(role="user", content="older question")])
    journal = SessionMessageJournal(session)
    current_task = LLMMessage(role="user", content="current question")
    messages = [LLMMessage(role="system", content="system"), current_task]

    result = await journal.preload(
        messages,
        limit=200,
        repair=lambda values: values,
        drop_leading_orphans=lambda values: values,
    )
    result.append(LLMMessage(role="assistant", content="current answer"))
    await journal.persist(result)

    assert [message.content for message in session.saved] == [
        "current question",
        "current answer",
    ]


@pytest.mark.asyncio
async def test_session_journal_excludes_framework_messages_marked_non_durable() -> None:
    session = _Session([])
    journal = SessionMessageJournal(session)
    messages = [
        LLMMessage(role="system", content="system"),
        LLMMessage(role="user", content="task"),
    ]
    messages = await journal.preload(
        messages,
        limit=200,
        repair=lambda values: values,
        drop_leading_orphans=lambda values: values,
    )
    messages.append(LLMMessage(role="user", content="[compaction summary]"))
    journal.note(messages, durable=False)
    messages.append(LLMMessage(role="assistant", content="answer"))
    await journal.persist(messages)

    assert [message.content for message in session.saved] == ["task", "answer"]
