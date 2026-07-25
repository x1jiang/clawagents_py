"""Quota-vs-rate-limit classification, and the compaction reduction guard."""

from __future__ import annotations

import pytest

from clawagents.errors.taxonomy import (
    RECOVERY_RECIPES,
    ErrorClass,
    classify_error,
)
from clawagents.memory.full_replace_compaction import (
    MAX_REDUCTION_RATIO,
    REDUCTION_CHECK_MIN_TOKENS,
    apply_full_replace_compaction,
)
from clawagents.providers.llm import LLMMessage


def _cls(message: str) -> ErrorClass:
    descriptor = classify_error(RuntimeError(message))
    return getattr(descriptor, "error_class", descriptor)


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - insufficient_quota: You exceeded your current quota",
        "quota exceeded for this project",
        "Your credit balance is too low to run this request",
        "Monthly usage limit reached",
        "402 payment required",
    ],
)
def test_billing_exhaustion_is_permanent_not_a_rate_limit(message):
    """These arrive as HTTP 429 but retrying only delays the real message."""
    cls = _cls(message)
    assert cls is ErrorClass.PROVIDER_QUOTA
    recipe = RECOVERY_RECIPES[cls]
    assert recipe.retryable is False
    assert recipe.max_retries == 0
    assert "quota" in recipe.recovery_hint.lower()


@pytest.mark.parametrize(
    "message",
    [
        "429 Too Many Requests",
        "rate_limit_exceeded — slow down",
        "resource_exhausted",
    ],
)
def test_genuine_rate_limits_still_retry(message):
    cls = _cls(message)
    assert cls is ErrorClass.PROVIDER_RATE_LIMIT
    assert RECOVERY_RECIPES[cls].retryable is True


class _StubLLM:
    """Returns a summary of a caller-chosen size."""

    def __init__(self, summary_chars: int) -> None:
        self._summary = "s" * summary_chars

    async def chat(self, messages, **kwargs):  # noqa: D401 - stub
        class _R:
            content = f"<summary>\n{'s' * 0}{''}\n</summary>"

        _R.content = f"<summary>\n{self._summary}\n</summary>"
        _R.model = "stub"
        _R.tokens_used = 0
        _R.prompt_tokens = 0
        return _R


def _history(token_estimate: int) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="do the thing"),
        LLMMessage(role="assistant", content="z" * (token_estimate * 4)),
        LLMMessage(role="user", content="continue"),
        LLMMessage(role="assistant", content="ok"),
        LLMMessage(role="user", content="next"),
    ]


@pytest.mark.asyncio
async def test_summary_that_does_not_shrink_history_is_discarded():
    """A summary bigger than what it replaces is a wasted call *and* lossy."""
    out = await apply_full_replace_compaction(
        _history(20_000), _StubLLM(200_000)  # ~50k-token summary
    )
    assert out is None


@pytest.mark.asyncio
async def test_a_genuinely_smaller_summary_is_applied():
    out = await apply_full_replace_compaction(
        _history(20_000), _StubLLM(4_000)  # ~1k-token summary
    )
    assert out is not None


@pytest.mark.asyncio
async def test_ratio_is_not_enforced_on_trivially_small_history():
    """Below the floor the ratio reflects summary boilerplate, not compaction."""
    small = REDUCTION_CHECK_MIN_TOKENS // 10
    out = await apply_full_replace_compaction(_history(small), _StubLLM(2_000))
    assert out is not None


@pytest.mark.asyncio
async def test_optional_min_compactable_floor_skips_the_llm_call():
    calls = {"n": 0}

    class _Counting(_StubLLM):
        async def chat(self, messages, **kwargs):
            calls["n"] += 1
            return await super().chat(messages, **kwargs)

    out = await apply_full_replace_compaction(
        _history(100), _Counting(100), min_compactable_tokens=5_000
    )
    assert out is None
    assert calls["n"] == 0  # never spent the summarization call


def test_reduction_ratio_constant_is_a_real_threshold():
    assert 0 < MAX_REDUCTION_RATIO < 1
