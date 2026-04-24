"""Per-run Usage accumulator.

Tracks LLM token consumption across every model call in a single agent run.
Inspired by openai-agents-python ``Usage`` — exposed via ``AgentState.usage``
and ``RunContext.usage`` so callers and tools can read real-time stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RequestUsage:
    """Per-request usage record for one model call."""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cache_creation_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    """Running total of LLM token consumption for one agent run.

    Fields:
        requests: number of LLM calls made
        input_tokens: total input (prompt) tokens across all calls
        output_tokens: total output (completion) tokens across all calls
        total_tokens: input + output across all calls
        cached_input_tokens: portion of ``input_tokens`` that hit the
            provider prompt cache
        reasoning_tokens: portion of ``output_tokens`` spent on hidden
            reasoning tokens (o1-style / thinking models)
        cache_creation_tokens: tokens written to the prompt cache during
            this run (Anthropic)
        per_request: list of :class:`RequestUsage`, one per LLM call
    """
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cache_creation_tokens: int = 0
    per_request: list[RequestUsage] = field(default_factory=list)

    def add_response(
        self,
        *,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> RequestUsage:
        """Record one LLM call into the running total."""
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        req = RequestUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.cached_input_tokens += cached_input_tokens
        self.reasoning_tokens += reasoning_tokens
        self.cache_creation_tokens += cache_creation_tokens
        self.per_request.append(req)
        return req

    def merge(self, other: "Usage") -> None:
        """Merge another ``Usage`` record into this one in-place."""
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.per_request.extend(other.per_request)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "per_request": [r.to_dict() for r in self.per_request],
        }
