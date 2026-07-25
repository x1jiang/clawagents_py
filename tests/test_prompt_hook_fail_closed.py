"""A deny-shaped PromptHook must be able to fail closed.

Every degraded path (no model, timeout, empty/unparseable response) allows the
action by default. That is right for an advisory hook and wrong for a
guardrail — "block writes outside the project root" would otherwise stop
applying, silently, whenever the cheap model hiccups.
"""

from __future__ import annotations

import asyncio

import pytest

from clawagents.hooks.prompt_hook import PromptHook, _parse_verdict

PROMPT = "block writes outside the project root"


class _Boom:
    async def chat(self, *a, **k):
        raise RuntimeError("model down")


class _Slow:
    async def chat(self, *a, **k):
        await asyncio.sleep(5)


class _Junk:
    async def chat(self, *a, **k):
        class _R:
            content = "sorry, I cannot comply"

        return _R()


class _Blocks:
    async def chat(self, *a, **k):
        class _R:
            content = '{"ok": false, "reason": "outside root"}'

        return _R()


class _Allows:
    async def chat(self, *a, **k):
        class _R:
            content = '{"ok": true, "reason": "inside root"}'

        return _R()


async def _verdict(llm, *, fail_closed: bool):
    hook = PromptHook(prompt=PROMPT, timeout_s=0.2, fail_closed=fail_closed)
    return await hook.evaluate({"tool": "write_file"}, llm_resolver=lambda _m: llm)


@pytest.mark.parametrize("llm", [_Boom(), _Slow(), _Junk()], ids=["error", "timeout", "junk"])
@pytest.mark.asyncio
async def test_degraded_paths_allow_by_default(llm):
    verdict = await _verdict(llm, fail_closed=False)
    assert verdict.ok is True
    assert "failed-open" in verdict.reason


@pytest.mark.parametrize("llm", [_Boom(), _Slow(), _Junk()], ids=["error", "timeout", "junk"])
@pytest.mark.asyncio
async def test_degraded_paths_block_when_fail_closed(llm):
    verdict = await _verdict(llm, fail_closed=True)
    assert verdict.ok is False
    assert "failed-closed" in verdict.reason


@pytest.mark.asyncio
async def test_fail_closed_does_not_override_a_real_verdict():
    """Only *degraded* paths invert — a genuine allow must still allow."""
    assert (await _verdict(_Allows(), fail_closed=True)).ok is True
    assert (await _verdict(_Blocks(), fail_closed=True)).ok is False
    assert (await _verdict(_Blocks(), fail_closed=False)).ok is False


def test_parse_verdict_honours_fail_closed_on_unusable_text():
    for text in ("", "no json here", "{broken"):
        assert _parse_verdict(text, fail_closed=True).ok is False
        assert _parse_verdict(text).ok is True


def test_default_stays_fail_open_for_backwards_compatibility():
    """Flipping the default would make every existing hook block on a hiccup."""
    assert PromptHook(prompt=PROMPT).fail_closed is False
