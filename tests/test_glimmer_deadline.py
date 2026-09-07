"""Deadline recovery must preserve honest outcomes and the cached prefix."""

import asyncio
from types import SimpleNamespace

from clawagents.graph.completion_handler import (
    CompletionHandler,
    _this_turn_has_tool_work,
)
from clawagents.graph.round_scheduler import RoundScheduler
from clawagents.graph.run_runtime import RunEvents
from clawagents.iteration_budget import IterationBudget
from clawagents.providers.llm import LLMMessage, LLMResponse


def _setup(monkeypatch, *, timeout=240, now=0):
    monkeypatch.setattr("clawagents.graph.round_scheduler.time.monotonic", lambda: now)
    context = SimpleNamespace(_metadata={}, iteration_budget=IterationBudget(max_total=50))
    events = RunEvents(lambda *_: None)
    scheduler = RoundScheduler(
        run_context=context, events=events, session_writer=None,
        timeout_s=timeout, started_at=0,
    )
    llm = SimpleNamespace(_max_tokens=6144)
    handler = CompletionHandler(
        registry=None, run_context=context, events=events, recorder=None,
        llm=llm, before_tool=None, action_mode="tools",
        looks_like_truncated_json=lambda _: False,
        sanitize_assistant_text=lambda text: text,
        goal_llm_complete=lambda *_: None,
    )
    state = SimpleNamespace(status="running", result="", iterations=0, tool_calls=0)
    messages = [LLMMessage(role="system", content="stable prefix"),
                LLMMessage(role="user", content="fix bug")]
    return scheduler, handler, context, llm, state, messages


def _cut(handler, state, messages):
    return asyncio.run(handler.handle(
        state=state, messages=messages,
        response=LLMResponse(content="", model="glimmer", tokens_used=6144,
                             finish_reason="length"),
        thinking=None, use_native_tools=True,
        consult_advisor=lambda *_: None, should_final_check=False,
    ))


def _begin(scheduler, state, messages):
    return asyncio.run(scheduler.begin(
        state, messages, round_index=state.iterations, cancel_event=asyncio.Event(),
    ))


def test_near_deadline_recovers_without_growing_cap(monkeypatch):
    _, handler, _, llm, state, messages = _setup(monkeypatch, now=220)
    assert _cut(handler, state, messages).action == "continue"
    assert llm._max_tokens == 6144
    assert "20" in messages[-1].content


def test_expired_deadline_does_not_retry_or_claim_done(monkeypatch):
    _, handler, _, llm, state, messages = _setup(monkeypatch, now=240)
    assert _cut(handler, state, messages).action == "done"
    assert state.status == "error"
    assert "incomplete" in state.result.lower()
    assert llm._max_tokens == 6144


def test_no_deadline_keeps_bounded_growth_and_reports_retry_exhaustion(monkeypatch):
    _, handler, _, llm, state, messages = _setup(monkeypatch, timeout=0, now=999)
    assert _cut(handler, state, messages).action == "continue"
    assert _cut(handler, state, messages).action == "continue"
    assert llm._max_tokens == 13824
    assert _cut(handler, state, messages).action == "done"
    assert state.status == "error"
    assert "incomplete" in state.result.lower()


def test_room_before_deadline_keeps_growth(monkeypatch):
    _, handler, _, llm, state, messages = _setup(monkeypatch, now=30)
    assert _cut(handler, state, messages).action == "continue"
    assert llm._max_tokens == 9216


def test_deadline_reminder_once_keeps_prefix_and_tool_evidence(monkeypatch):
    scheduler, _, _, _, state, messages = _setup(monkeypatch, now=200)
    messages.extend([LLMMessage(role="assistant", content="checking"),
                     LLMMessage(role="tool", content="evidence", tool_call_id="1")])
    prefix = messages[0]
    assert _begin(scheduler, state, messages).action == "proceed"
    size = len(messages)
    assert messages[-1].role == "user"
    assert "40" in messages[-1].content
    assert _begin(scheduler, state, messages).action == "proceed"
    assert len(messages) == size
    assert messages[0] is prefix and prefix.content == "stable prefix"
    assert _this_turn_has_tool_work(messages)


def test_new_unlimited_run_clears_reused_context_deadline(monkeypatch):
    _, handler, context, llm, state, messages = _setup(monkeypatch, now=300)
    RoundScheduler(run_context=context, events=RunEvents(lambda *_: None),
                   session_writer=None, timeout_s=0, started_at=300)
    assert _cut(handler, state, messages).action == "continue"
    assert llm._max_tokens == 9216


def test_scheduler_stops_at_exact_deadline(monkeypatch):
    scheduler, _, _, _, state, messages = _setup(monkeypatch, now=240)
    assert _begin(scheduler, state, messages).action == "stop"
    assert state.status == "error"


def test_healthy_answer_after_deadline_can_complete(monkeypatch):
    _, handler, _, _, state, messages = _setup(monkeypatch, now=241)
    decision = asyncio.run(handler.handle(
        state=state, messages=messages,
        response=LLMResponse(content="The fix and verification are complete.",
                             model="glimmer", tokens_used=50),
        thinking=None, use_native_tools=True,
        consult_advisor=lambda *_: None, should_final_check=False,
    ))
    assert decision.action == "done"
    assert state.status == "done"


def test_output_cap_restored_once_even_after_two_growths(monkeypatch):
    _, handler, _, llm, state, messages = _setup(monkeypatch, timeout=0)
    assert _cut(handler, state, messages).action == "continue"
    assert _cut(handler, state, messages).action == "continue"
    handler.restore_output_budget()
    assert llm._max_tokens == 6144
    llm._max_tokens = 8192
    handler.restore_output_budget()
    assert llm._max_tokens == 8192


def test_restore_preserves_intervening_provider_configuration(monkeypatch):
    _, handler, _, llm, state, messages = _setup(monkeypatch)
    _cut(handler, state, messages)
    llm._max_tokens = 8192
    handler.restore_output_budget()
    assert llm._max_tokens == 8192


def test_fallback_output_cap_restores_primary(monkeypatch):
    _, handler, _, llm, state, messages = _setup(monkeypatch)
    handler._llm = SimpleNamespace(primary=llm)
    _cut(handler, state, messages)
    assert llm._max_tokens == 9216
    handler.restore_output_budget()
    assert llm._max_tokens == 6144


def test_provider_without_mutable_cap_needs_no_restoration(monkeypatch):
    _, handler, _, _, state, messages = _setup(monkeypatch)
    handler._llm = object()
    assert _cut(handler, state, messages).action == "continue"
    handler.restore_output_budget()
    handler.restore_output_budget()
