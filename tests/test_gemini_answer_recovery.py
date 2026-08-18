"""Gemini empty / [called …] replies must not become the final answer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from clawagents.graph.completion_handler import CompletionHandler
from clawagents.providers.llm import GEMINI_SUMMARIZE_MARKER, LLMMessage, LLMResponse


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
