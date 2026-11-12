"""Per-run Usage accumulator.

Tracks LLM token consumption across every model call in a single agent run.
Inspired by openai-agents-python ``Usage`` — exposed via ``AgentState.usage``
and ``RunContext.usage`` so callers and tools can read real-time stats.
"""

from __future__ import annotations

import time
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
    elapsed_ms: float | None = None
    decode_ms: float | None = None
    decode_tokens_per_sec: float | None = None
    peak_memory_bytes: int = 0
    # Wall-clock of the call. Prompt caches expire on idle (Anthropic's default
    # TTL is 5 minutes), so attributing a cache miss to "you were idle" rather
    # than "your context changed" needs the gap between calls.
    started_at: float = 0.0

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
        elapsed_ms: float | None = None,
        peak_memory_bytes: int = 0,
    ) -> RequestUsage:
        """Record one LLM call into the running total."""
        if prompt_tokens is None:
            prompt_tokens = input_tokens + cached_input_tokens + cache_creation_tokens
        if total_tokens is None:
            total_tokens = prompt_tokens + output_tokens
        decode_ms: float | None = None
        decode_tps: float | None = None
        if elapsed_ms is not None and elapsed_ms >= 0:
            ttft = max(0.0, time_to_first_token_ms) if time_to_first_token_ms is not None else 0.0
            decode_ms = round(max(0.0, elapsed_ms - ttft), 3)
            if output_tokens > 0 and decode_ms >= 1.0:
                decode_tps = round(output_tokens / (decode_ms / 1000.0), 2)
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
            elapsed_ms=round(elapsed_ms, 3) if elapsed_ms is not None else None,
            decode_ms=decode_ms,
            decode_tokens_per_sec=decode_tps,
            peak_memory_bytes=peak_memory_bytes,
            started_at=time.time(),
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

    @property
    def last_ttft_ms(self) -> float | None:
        for req in reversed(self.per_request):
            if req.time_to_first_token_ms is not None:
                return req.time_to_first_token_ms
        return self.time_to_first_token_ms

    @property
    def avg_ttft_ms(self) -> float | None:
        vals = [r.time_to_first_token_ms for r in self.per_request if r.time_to_first_token_ms is not None]
        if not vals:
            return self.time_to_first_token_ms
        return round(sum(vals) / len(vals), 2)

    @property
    def steady_decode_tps(self) -> float | None:
        """Steady-state decode throughput (tok/s), excluding warmup call #0 when >=2 samples exist."""
        samples = [
            (r.output_tokens, r.decode_ms)
            for r in self.per_request
            if r.output_tokens > 0 and r.decode_ms is not None and r.decode_ms >= 1.0
        ]
        if not samples:
            return None
        steady = samples[1:] if len(samples) >= 2 else samples
        tot_out = sum(o for o, _ in steady)
        tot_ms = sum(ms for _, ms in steady if ms is not None)
        if tot_out <= 0 or tot_ms < 1.0:
            return None
        return round(tot_out / (tot_ms / 1000.0), 2)

    @property
    def max_step_ms(self) -> float | None:
        vals = [r.elapsed_ms for r in self.per_request if r.elapsed_ms is not None]
        return round(max(vals), 2) if vals else None

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return round(self.cached_input_tokens / self.prompt_tokens, 4)

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
            "avg_ttft_ms": self.avg_ttft_ms,
            "steady_decode_tps": self.steady_decode_tps,
            "max_step_ms": self.max_step_ms,
            "cache_hit_ratio": self.cache_hit_ratio,
            "peak_memory_bytes": self.peak_memory_bytes,
            "per_request": [r.to_dict() for r in self.per_request],
        }
