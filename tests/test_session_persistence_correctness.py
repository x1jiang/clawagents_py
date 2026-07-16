"""Regression tests for session persistence correctness (P1 backlog).

The old implementation captured a numeric index (``_session_start_cursor``)
into ``messages`` at preload time and persisted ``messages[cursor:]`` at the
end of the run. Compaction rebuilds the list (shrinking it) and dangling
tool-call patching inserts items, so the slice persisted the wrong range:
after compaction the run's new turns were silently lost, and patch inserts
re-persisted duplicated preloaded messages.

The fix tracks message *objects* by identity instead of by index.
"""

import asyncio
import json
from typing import Any, Dict

from clawagents.graph.agent_loop import run_agent_graph
from clawagents.providers.llm import LLMMessage, LLMResponse, NativeToolCall
from clawagents.tools.registry import ToolRegistry, ToolResult


class _EchoTool:
    name = "echo"
    description = "Echo back the input"
    parameters: Dict[str, Dict[str, Any]] = {
        "x": {"type": "string", "required": True},
    }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output=f"echo:{json.dumps(args)}")


class _RecordingSession:
    """Session backend that preloads a long history and records persists."""

    def __init__(self, preloaded: list[LLMMessage]):
        self._preloaded = preloaded
        self.persisted: list[LLMMessage] = []

    async def get_items(self, limit=None):
        return list(self._preloaded)

    async def add_items(self, items):
        self.persisted.extend(items)


class _CompactingMockLLM:
    """Returns a tool call, then a final answer; answers summarize prompts.

    The compaction path calls ``llm.chat`` with a single user message that
    starts with "You are summarizing a chunk" — detect those and return a
    short summary so compaction succeeds deterministically.
    """

    name = "mock"

    def __init__(self):
        self.main_calls = 0
        self.summarize_calls = 0

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
        # Both summarizer engines must be recognized: the legacy chunk
        # summarizer and the compress_messages_safe summarization engine
        # (v6.10.6+). Otherwise their calls consume the scripted main-turn
        # responses and the test exercises a different flow than intended.
        is_summarize = any(
            isinstance(m.content, str)
            and (
                m.content.startswith("You are summarizing a chunk")
                or m.content.startswith("You are a summarization engine")
                or m.content.startswith(
                    "Your task is to produce a faithful, concise summary"
                )
            )
            for m in messages
        )
        if is_summarize:
            self.summarize_calls += 1
            # Long enough to pass full-replace degenerate-summary gate (≥500 chars).
            return LLMResponse(
                content="<summary>\n" + ("COMPACTION_SUMMARY detail " * 40) + "\n</summary>",
                model="mock",
                tokens_used=5,
            )

        self.main_calls += 1
        if self.main_calls == 1:
            return LLMResponse(
                content="",
                model="mock",
                tokens_used=10,
                tool_calls=[NativeToolCall("echo", {"x": "hello"}, "tc-1")],
            )
        return LLMResponse(content="FINAL_ANSWER", model="mock", tokens_used=10)


def _preloaded_history(n_pairs: int) -> list[LLMMessage]:
    msgs: list[LLMMessage] = []
    for i in range(n_pairs):
        msgs.append(LLMMessage(role="user", content=f"prior-user-{i} " + "x" * 600))
        msgs.append(LLMMessage(role="assistant", content=f"prior-assistant-{i} " + "y" * 600))
    return msgs


def test_compaction_does_not_lose_or_duplicate_persisted_turns():
    """End-to-end: preload -> compaction -> tool round -> final answer.

    The preloaded history is large enough (vs. the tiny context window)
    that compaction fires and rebuilds ``messages`` mid-run. The persisted
    items must still be exactly the run's new turns:
      - the tool round and the final assistant answer are present (not lost)
      - no preloaded message is re-persisted (no duplication)
      - no framework-synthesized compaction summary is persisted
    """
    session = _RecordingSession(_preloaded_history(15))  # 30 messages
    llm = _CompactingMockLLM()
    registry = ToolRegistry()
    registry.register(_EchoTool())

    state = asyncio.run(run_agent_graph(
        "TASK_MARKER do the thing",
        llm,
        tools=registry,
        max_iterations=6,
        streaming=False,
        context_window=2000,
        use_native_tools=True,
        session=session,
    ))

    assert state.status == "done"
    assert llm.summarize_calls > 0, (
        "test setup problem: compaction never fired, so this test no longer "
        "exercises the cursor-vs-compaction interaction"
    )

    persisted_contents = [
        m.content if isinstance(m.content, str) else json.dumps(m.content)
        for m in session.persisted
    ]

    # No preloaded message may be re-persisted.
    dupes = [c for c in persisted_contents if c.startswith("prior-")]
    assert dupes == [], f"preloaded messages re-persisted: {dupes[:3]}"

    # No compaction artifact may be persisted.
    summaries = [c for c in persisted_contents if "COMPACTION_SUMMARY" in c]
    assert summaries == [], "compaction summary leaked into the session store"

    # The run's new turns must be present: tool call, tool result, final.
    assert any(
        m.role == "assistant" and m.tool_calls_meta
        and any(tc.get("name") == "echo" for tc in m.tool_calls_meta)
        for m in session.persisted
    ), "assistant tool-call turn was lost from the session store"
    assert any(
        m.role == "tool" and isinstance(m.content, str) and m.content.startswith("echo:")
        for m in session.persisted
    ), "tool result turn was lost from the session store"
    assert any(c == "FINAL_ANSWER" for c in persisted_contents), (
        "final assistant answer was lost from the session store"
    )

    # And nothing is persisted twice.
    assert len(persisted_contents) == len(set(persisted_contents)), (
        f"duplicate persisted items: {persisted_contents}"
    )


def test_no_compaction_baseline_still_persists_new_turns():
    """Sanity: with a huge context window (no compaction) behavior matches
    the old cursor semantics — new turns persisted, preloaded ones not."""
    session = _RecordingSession(_preloaded_history(2))
    llm = _CompactingMockLLM()
    registry = ToolRegistry()
    registry.register(_EchoTool())

    state = asyncio.run(run_agent_graph(
        "TASK_MARKER do the thing",
        llm,
        tools=registry,
        max_iterations=6,
        streaming=False,
        context_window=1_000_000,
        use_native_tools=True,
        session=session,
    ))

    assert state.status == "done"
    assert llm.summarize_calls == 0
    contents = [
        m.content if isinstance(m.content, str) else json.dumps(m.content)
        for m in session.persisted
    ]
    assert not any(c.startswith("prior-") for c in contents)
    assert any(c == "FINAL_ANSWER" for c in contents)
    assert any(
        m.role == "tool" and isinstance(m.content, str) and m.content.startswith("echo:")
        for m in session.persisted
    )
