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
    prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cache_creation_tokens: int = 0
    time_to_first_token_ms: float | None = None
    peak_memory_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    """Running total of LLM token consumption for one agent run.

    Fields:
        requests: number of LLM calls made
        prompt_tokens: total prompt tokens including cache reads and writes
        input_tokens: uncached input tokens across all calls
        output_tokens: total output (completion) tokens across all calls
        total_tokens: input + output across all calls
        cached_input_tokens: prompt tokens read from the provider cache
        reasoning_tokens: portion of ``output_tokens`` spent on hidden
            reasoning tokens (o1-style / thinking models)
        cache_creation_tokens: tokens written to the prompt cache during
            this run (Anthropic)
        time_to_first_token_ms: TTFT for the first model request in this run
        peak_memory_bytes: highest observed RSS sampled during this run
        per_request: list of :class:`RequestUsage`, one per LLM call
    """

    requests: int = 0
    prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cache_creation_tokens: int = 0
    time_to_first_token_ms: float | None = None
    peak_memory_bytes: int = 0
    per_request: list[RequestUsage] = field(default_factory=list)

    def add_response(
        self,
        *,
        model: str = "",
        prompt_tokens: int | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
        cache_creation_tokens: int = 0,
        time_to_first_token_ms: float | None = None,
        peak_memory_bytes: int = 0,
    ) -> RequestUsage:
        """Record one LLM call into the running total."""
        if prompt_tokens is None:
            prompt_tokens = input_tokens + cached_input_tokens + cache_creation_tokens
        if total_tokens is None:
            total_tokens = prompt_tokens + output_tokens
        req = RequestUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_creation_tokens=cache_creation_tokens,
            time_to_first_token_ms=time_to_first_token_ms,
            peak_memory_bytes=peak_memory_bytes,
        )
        self.requests += 1
        self.prompt_tokens += prompt_tokens
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.cached_input_tokens += cached_input_tokens
        self.reasoning_tokens += reasoning_tokens
        self.cache_creation_tokens += cache_creation_tokens
        if self.time_to_first_token_ms is None and time_to_first_token_ms is not None:
            self.time_to_first_token_ms = time_to_first_token_ms
        self.peak_memory_bytes = max(self.peak_memory_bytes, peak_memory_bytes)
        self.per_request.append(req)
        return req

    def merge(self, other: "Usage") -> None:
        """Merge another ``Usage`` record into this one in-place."""
        self.requests += other.requests
        self.prompt_tokens += other.prompt_tokens
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        if self.time_to_first_token_ms is None:
            self.time_to_first_token_ms = other.time_to_first_token_ms
        self.peak_memory_bytes = max(self.peak_memory_bytes, other.peak_memory_bytes)
        self.per_request.extend(other.per_request)

    def sample_memory(self, memory_bytes: int | None = None) -> int:
        """Sample current RSS and update the observed run peak."""
        if memory_bytes is None:
            try:
                import resource

                # ru_maxrss is kilobytes on Linux, bytes on macOS.
                rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                import sys

                if sys.platform != "darwin":
                    rss *= 1024
                memory_bytes = rss
            except Exception:
                memory_bytes = 0
        self.peak_memory_bytes = max(self.peak_memory_bytes, int(memory_bytes or 0))
        return self.peak_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "per_request": [r.to_dict() for r in self.per_request],
        }
