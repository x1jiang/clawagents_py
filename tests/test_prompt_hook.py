"""Tests for PromptHook (LLM-evaluated guardrail)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from clawagents.hooks.prompt_hook import PromptHook, PromptHookVerdict, _parse_verdict


def test_promptbook_validates_empty_prompt():
    with pytest.raises(ValueError):
        PromptHook(prompt="")


def test_parse_verdict_clean_json():
    v = _parse_verdict('{"ok": false, "reason": "danger"}')
    assert v.ok is False
    assert v.reason == "danger"


def test_parse_verdict_with_code_fence():
    v = _parse_verdict('```json\n{"ok": true, "reason": "fine"}\n```')
    assert v.ok is True
    assert v.reason == "fine"


def test_parse_verdict_with_prose_around():
    raw = "Sure, here is my verdict:\n{\"ok\": false, \"reason\": \"x\"}\nThanks."
    v = _parse_verdict(raw)
    assert v.ok is False
    assert v.reason == "x"


def test_parse_verdict_missing_reason_defaults_to_none():
    v = _parse_verdict('{"ok": true}')
    assert v.ok is True
    assert v.reason is None


def test_parse_verdict_no_json_fails_open():
    v = _parse_verdict("the model totally forgot to emit JSON")
    assert v.ok is True
    assert v.reason and "failed-open" in v.reason


def test_parse_verdict_bad_json_fails_open():
    v = _parse_verdict("{ ok: true } definitely not valid json")
    assert v.ok is True
    assert v.reason and "failed-open" in v.reason


def test_parse_verdict_empty_response_fails_open():
    v = _parse_verdict("")
    assert v.ok is True
    assert v.reason and "failed-open" in v.reason


@pytest.mark.asyncio
async def test_evaluate_uses_provided_resolver_and_blocks():
    """End-to-end: a stub LLM that returns ok=false → verdict.ok is False."""

    captured_messages: list[Any] = []

    class _StubLLM:
        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
            captured_messages.extend(messages)
            from clawagents.providers.llm import LLMResponse
            return LLMResponse(
                content='{"ok": false, "reason": "writes outside repo"}',
                model="stub",
                tokens_used=8,
            )

    def resolver(_model: str | None):
        return _StubLLM()

    hook = PromptHook(
        prompt="Block tool calls that write files outside the project root.",
        model="stub-model",
    )
    verdict = await hook.evaluate(
        payload={"tool": "write_file", "path": "/etc/passwd"},
        llm_resolver=resolver,
    )

    assert verdict.ok is False
    assert verdict.reason == "writes outside repo"
    # The user-message must include the rule and the event payload as JSON
    user_msg = captured_messages[-1].content
    assert "write files outside" in user_msg.lower()
    assert "/etc/passwd" in user_msg


@pytest.mark.asyncio
async def test_evaluate_fails_open_on_timeout():
    class _SlowLLM:
        async def chat(self, *args, **kwargs):
            await asyncio.sleep(10)  # would block past timeout

    hook = PromptHook(prompt="block bad things", timeout_s=0.05)
    verdict = await hook.evaluate(
        payload={"foo": "bar"},
        llm_resolver=lambda _: _SlowLLM(),
    )
    assert verdict.ok is True
    assert verdict.reason and "timeout" in verdict.reason.lower()


@pytest.mark.asyncio
async def test_evaluate_fails_open_on_llm_error():
    class _BoomLLM:
        async def chat(self, *args, **kwargs):
            raise RuntimeError("provider down")

    hook = PromptHook(prompt="block bad things")
    verdict = await hook.evaluate(
        payload={"foo": "bar"},
        llm_resolver=lambda _: _BoomLLM(),
    )
    assert verdict.ok is True
    assert verdict.reason and "failed-open" in verdict.reason
