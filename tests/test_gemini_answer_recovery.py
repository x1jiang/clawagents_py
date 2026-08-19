"""Gemini empty / [called …] replies must not become the final answer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from clawagents.graph.completion_handler import (
    CompletionHandler,
    _UNGROUNDED_REFUSAL,
)
from clawagents.providers.llm import (
    GEMINI_EVIDENCE_MARKER,
    GEMINI_SUMMARIZE_MARKER,
    LLMMessage,
    LLMResponse,
)


class _Events:
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def emit(self, kind: str, data=None) -> None:
        self.kinds.append(kind)

    def typed(self, kind: str, data=None) -> None:
        self.kinds.append(kind)


def _handler() -> CompletionHandler:
    return CompletionHandler(
        registry=None,
        run_context=SimpleNamespace(_metadata={}),
        events=_Events(),
        recorder=None,
        llm=None,
        before_tool=None,
        action_mode="tools",
        looks_like_truncated_json=lambda _text: False,
        sanitize_assistant_text=lambda text: text,
        goal_llm_complete=lambda *_a, **_k: (lambda _s: _s),
    )


def _handle(handler: CompletionHandler, **kwargs):
    return asyncio.run(handler.handle(**kwargs))


def test_command_dump_continues_instead_of_done():
    handler = _handler()
    messages = [LLMMessage(role="user", content="who has pain 4-7?")]
    response = LLMResponse(
        content="[called write_file({path: 'x.py', content: 'script'})]",
        model="gemini-3.7-flash",
        tokens_used=10,
    )
    decision = _handle(
        handler,
        state=SimpleNamespace(),
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "continue"
    assert messages[-1].role == "user"
    assert GEMINI_SUMMARIZE_MARKER in str(messages[-1].content)


def test_empty_after_tools_continues():
    handler = _handler()
    messages = [
        LLMMessage(role="user", content="who has pain 4-7?"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c1", "name": "execute", "args": {}}],
        ),
        LLMMessage(role="tool", content="12 patients", tool_call_id="c1"),
    ]
    response = LLMResponse(content="", model="gemini-3.7-flash", tokens_used=4)
    decision = _handle(
        handler,
        state=SimpleNamespace(),
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "continue"
    assert GEMINI_SUMMARIZE_MARKER in str(messages[-1].content)


def test_plain_answer_still_completes():
    handler = _handler()
    messages = [LLMMessage(role="user", content="who has pain 4-7?")]
    response = LLMResponse(
        content="Twelve patients match on the first encounter.",
        model="gemini-3.7-flash",
        tokens_used=8,
    )
    state = SimpleNamespace(result=None, status="running")
    decision = _handle(
        handler,
        state=state,
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "done"
    assert state.result == "Twelve patients match on the first encounter."


def test_nudge_cap_stops_retry_loop():
    handler = _handler()
    messages = [
        LLMMessage(role="user", content="who has pain 4-7?"),
        LLMMessage(role="user", content=f"{GEMINI_SUMMARIZE_MARKER} already collected."),
        LLMMessage(role="user", content=f"{GEMINI_SUMMARIZE_MARKER} already collected."),
    ]
    response = LLMResponse(
        content="[called execute({cmd: 'ls'})]",
        model="gemini-3.7-flash",
        tokens_used=6,
    )
    state = SimpleNamespace(result=None, status="running")
    decision = _handle(
        handler,
        state=state,
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "done"
    assert "[called execute" in state.result


def test_evidence_nudge_cap_refuses_invented_table():
    handler = _handler()
    messages = [
        LLMMessage(role="user", content="intraday distribution"),
        LLMMessage(role="user", content=f"{GEMINI_EVIDENCE_MARKER}. Call execute."),
        LLMMessage(role="user", content=f"{GEMINI_EVIDENCE_MARKER}. Call execute."),
    ]
    response = LLMResponse(
        content=_FAKE_INTRADAY_TABLE,
        model="gemini-3.7-flash",
        tokens_used=20,
    )
    state = SimpleNamespace(result=None, status="running")
    decision = _handle(
        handler,
        state=state,
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "done"
    assert state.result == _UNGROUNDED_REFUSAL
    assert "ED_ARRIVAL_TIME" not in state.result


_FAKE_INTRADAY_TABLE = """
Done. Real Intraday Time Distribution (Using PAT_ENC_HSP.ED_ARRIVAL_TIME):

| Day | 00:00–03:59 | 04:00–07:59 | 08:00–11:59 | Total |
| --- | ---: | ---: | ---: | ---: |
| Monday | 5 | 3 | 9 | 45 |
| Tuesday | 4 | 2 | 7 | 39 |
"""


def test_invented_count_table_without_tools_continues():
    handler = _handler()
    messages = [
        LLMMessage(
            role="user",
            content="Then use some real intraday timestamp to get more accurate time distribution",
        )
    ]
    response = LLMResponse(
        content=_FAKE_INTRADAY_TABLE,
        model="gemini-3.7-flash",
        tokens_used=40,
    )
    decision = _handle(
        handler,
        state=SimpleNamespace(),
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "continue"
    assert GEMINI_EVIDENCE_MARKER in str(messages[-1].content)


def test_count_table_after_this_turn_execute_completes():
    handler = _handler()
    messages = [
        LLMMessage(role="user", content="use a real intraday timestamp"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c1", "name": "execute", "args": {}}],
        ),
        LLMMessage(
            role="tool",
            content="Monday 5 3 9 17\nTuesday 4 2 7 13",
            tool_call_id="c1",
        ),
    ]
    grounded = """
| Day | 00:00–03:59 | 04:00–07:59 | 08:00–11:59 | Total |
| --- | ---: | ---: | ---: | ---: |
| Monday | 5 | 3 | 9 | 17 |
| Tuesday | 4 | 2 | 7 | 13 |
"""
    response = LLMResponse(
        content=grounded,
        model="gemini-3.7-flash",
        tokens_used=20,
    )
    state = SimpleNamespace(result=None, status="running")
    decision = _handle(
        handler,
        state=state,
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "done"
    assert "Monday" in state.result


def test_invented_hour_cells_after_daily_total_execute_continues():
    handler = _handler()
    messages = [
        LLMMessage(role="user", content="use a real intraday timestamp"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c1", "name": "execute", "args": {}}],
        ),
        LLMMessage(role="tool", content="Monday 45\nTuesday 39", tool_call_id="c1"),
    ]
    response = LLMResponse(
        content=_FAKE_INTRADAY_TABLE,
        model="gemini-3.7-flash",
        tokens_used=20,
    )
    decision = _handle(
        handler,
        state=SimpleNamespace(),
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "continue"
    assert GEMINI_EVIDENCE_MARKER in str(messages[-1].content)
    assert "45" in str(messages[-1].content) or "missing counts" in str(messages[-1].content)


def test_use_skill_only_then_count_table_continues():
    handler = _handler()
    messages = [
        LLMMessage(role="user", content="intraday distribution"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c1", "name": "use_skill", "args": {}}],
        ),
        LLMMessage(
            role="tool",
            content="Skill body with example counts 45 39 56",
            tool_call_id="c1",
        ),
    ]
    response = LLMResponse(
        content=_FAKE_INTRADAY_TABLE,
        model="gemini-3.7-flash",
        tokens_used=20,
    )
    decision = _handle(
        handler,
        state=SimpleNamespace(),
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "continue"
    assert "use_skill` is instructions" in str(messages[-1].content)


def test_claimed_sql_execution_without_tools_continues():
    handler = _handler()
    messages = [LLMMessage(role="user", content="Have you execute any sql to get the answer?")]
    response = LLMResponse(
        content="Yes. I executed SQL and the query returned 305 qualifying encounters.",
        model="gemini-3.7-flash",
        tokens_used=12,
    )
    decision = _handle(
        handler,
        state=SimpleNamespace(),
        messages=messages,
        response=response,
        thinking=None,
        use_native_tools=True,
        consult_advisor=lambda *_a, **_k: None,
        should_final_check=False,
    )
    assert decision.action == "continue"
    assert GEMINI_EVIDENCE_MARKER in str(messages[-1].content)
