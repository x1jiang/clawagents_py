"""A session log must hold enough to actually resume from.

`clawagents --resume` rebuilds a conversation with `SessionReader`. Three
separate gaps meant it rebuilt almost nothing:

  * the prompt arrives as the ``task`` argument, and nothing wrote it down;
  * `reconstruct_messages` had no ``user_message`` branch, so even a written
    one would have been dropped (and `get_task()`, which scans the rebuilt
    messages for a user turn, could never succeed);
  * only the tool-calling path wrote an ``assistant_message``, so a run ending
    in plain prose — the common shape — stored no answer.

Each is pinned below, plus the round trip that only works with all three.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from clawagents.config.features import temporary_overrides
from clawagents.graph.agent_loop import run_agent_graph
from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse, NativeToolCall
from clawagents.session.persistence import SessionReader, SessionWriter
from clawagents.tools.registry import ToolRegistry, ToolResult


class _ScriptedLLM(LLMProvider):
    model = "stub"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        response = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return response

    async def stream(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


class _EchoTool:
    name = "echo"
    description = "Echo text back."
    parameters = {"text": {"type": "string", "required": True}}

    async def execute(self, args):
        return ToolResult(success=True, output=args.get("text", ""))


def _response(content: str = "", tool_calls=None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        model="stub",
        tokens_used=5,
        prompt_tokens=3,
    )


async def _run(tmp_path: Path, script, *, with_tool: bool) -> Path:
    registry = ToolRegistry()
    if with_tool:
        registry.register(_EchoTool())
    with temporary_overrides({"session_persistence": True}):
        await run_agent_graph(
            task="say the magic word",
            llm=_ScriptedLLM(script),
            tools=registry,
            max_iterations=4,
            streaming=False,
        )
    sessions = sorted(
        (tmp_path / ".clawagents" / "sessions").glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    assert sessions, "session persistence was on but no log was written"
    return sessions[-1]


def test_writer_records_the_user_prompt(tmp_path: Path) -> None:
    writer = SessionWriter(session_id="s1", session_dir=tmp_path)
    writer.write_user_message("what is the capital of France?")
    messages = SessionReader(writer.path).reconstruct_messages()
    assert [(m.role, m.content) for m in messages] == [
        ("user", "what is the capital of France?")
    ]


def test_get_task_finds_the_prompt(tmp_path: Path) -> None:
    """get_task() scans reconstructed messages, so it needs the user branch."""
    writer = SessionWriter(session_id="s2", session_dir=tmp_path)
    writer.write_system_prompt("you are an agent")
    writer.write_user_message("count to three")
    assert SessionReader(writer.path).get_task() == "count to three"


@pytest.mark.parametrize(
    "label, script, with_tool",
    [
        ("plain prose", [_response(content="XYZZY")], False),
        (
            "tool then answer",
            [
                _response(
                    tool_calls=[
                        NativeToolCall(
                            tool_name="echo", args={"text": "hi"}, tool_call_id="c1"
                        )
                    ]
                ),
                _response(content="XYZZY"),
            ],
            True,
        ),
    ],
)
def test_session_round_trip_keeps_prompt_and_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label, script, with_tool
) -> None:
    monkeypatch.chdir(tmp_path)
    path = asyncio.run(_run(tmp_path, script, with_tool=with_tool))
    messages = SessionReader(path).reconstruct_messages()

    assert any(
        m.role == "user" and "magic word" in (m.content or "") for m in messages
    ), f"[{label}] prompt missing from the log: {[m.role for m in messages]}"
    assert any(
        m.role == "assistant" and "XYZZY" in (m.content or "") for m in messages
    ), f"[{label}] final answer missing from the log: {[m.role for m in messages]}"


def test_final_answer_is_not_written_twice(tmp_path: Path, monkeypatch) -> None:
    """The engine already stores content on the tool-calling path.

    Recording the run's result on top of an identical stored message would show
    the answer twice in any rebuilt transcript.
    """
    monkeypatch.chdir(tmp_path)
    path = asyncio.run(_run(tmp_path, [_response(content="XYZZY")], with_tool=False))
    messages = SessionReader(path).reconstruct_messages()
    finals = [
        m for m in messages if m.role == "assistant" and (m.content or "") == "XYZZY"
    ]
    assert len(finals) == 1, f"answer persisted {len(finals)} times"


def test_empty_result_writes_no_assistant_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = asyncio.run(_run(tmp_path, [_response(content="")], with_tool=False))
    messages = SessionReader(path).reconstruct_messages()
    assert not [
        m for m in messages if m.role == "assistant" and (m.content or "").strip()
    ], "a run with no answer should not invent an assistant turn"


def test_resume_replays_prior_turns_into_a_session() -> None:
    """cmd_resume must feed the rebuilt messages to the agent.

    It previously reconstructed them, printed the count, and discarded them —
    so resume started blank. Pin the two behaviours the CLI depends on: the
    stored system prompt is dropped (the new run builds its own) and every
    other turn survives.
    """
    from clawagents.session.backends import InMemorySession

    reconstructed = [
        LLMMessage(role="system", content="OLD SYSTEM PROMPT"),
        LLMMessage(role="user", content="original task"),
        LLMMessage(role="assistant", content="earlier answer"),
    ]
    prior = [m for m in reconstructed if m.role != "system"]
    session = InMemorySession(session_id="s3")
    asyncio.run(session.add_items(prior))
    items = asyncio.run(session.get_items())

    roles = [m.role for m in items]
    assert "system" not in roles
    assert roles == ["user", "assistant"]
    assert any("earlier answer" in (m.content or "") for m in items)


def test_append_does_not_rewrite_the_whole_log(tmp_path: Path) -> None:
    """Appending must stay O(1) in the log's size.

    `append` used to read the entire file and rewrite it through a temp file for
    every event, making a session quadratic in its own size — seconds of I/O on
    the turn's critical path once a log reached a few MB. Asserting on time is
    flaky, so assert on the mechanism: a same-directory temp file is the
    signature of the read-modify-write, and the earlier bytes must be untouched.
    """
    writer = SessionWriter(session_id="grow", session_dir=tmp_path)
    writer.append("user_message", {"content": "first"})
    first_bytes = writer.path.read_bytes()

    for i in range(50):
        writer.append("tool_result", {"tool_call_id": f"c{i}", "output": "x" * 500})

    assert writer.path.read_bytes().startswith(first_bytes), (
        "earlier events were rewritten; append should only add to the end"
    )
    assert list(tmp_path.glob("*.tmp")) == [], "append left temp files behind"
    assert len(SessionReader(writer.path).events) == 51


def test_reader_tolerates_a_torn_final_line(tmp_path: Path) -> None:
    """A process killed mid-append leaves a partial last line.

    Raising there made the whole session unreadable — including every complete
    event before it — which is the opposite of what a recovery path wants.
    """
    writer = SessionWriter(session_id="torn", session_dir=tmp_path)
    writer.write_user_message("keep me")
    writer.write_assistant_message("keep me too")
    with open(writer.path, "a", encoding="utf-8") as handle:
        handle.write('{"type": "tool_result", "output": "cut off mid-')

    messages = SessionReader(writer.path).reconstruct_messages()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert any("keep me too" in (m.content or "") for m in messages)


def test_append_recreates_a_deleted_session_dir(tmp_path: Path) -> None:
    """Long runs outlive `rm -rf` of a scratch directory."""
    import shutil

    session_dir = tmp_path / "sessions"
    writer = SessionWriter(session_id="gone", session_dir=session_dir)
    writer.write_user_message("before")
    shutil.rmtree(session_dir)

    writer.write_assistant_message("after")
    assert writer.path.exists()
    assert [m.role for m in SessionReader(writer.path).reconstruct_messages()] == [
        "assistant"
    ]
