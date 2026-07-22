"""Tool loop detection and consecutive failure tracking.

Two independent trackers used by the main agent loop to detect and break
out of unproductive tool-call cycles.

Extracted from ``agent_loop.py`` for modularity.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from clawagents.tools.registry import ParsedToolCall

if TYPE_CHECKING:
    from clawagents.loop_detection import LoopDetectionConfig


# ─── Tool Loop Detection ──────────────────────────────────────────────────


class _ToolCallTracker:
    def __init__(
        self,
        window_size: int = 30,
        soft_limit: int = 3,
        hard_limit: int = 6,
        circuit_breaker_limit: int = 30,
        loop_config: "LoopDetectionConfig | None" = None,
    ):
        from clawagents.loop_detection import resolve_loop_detection_config

        self._history: list[str] = []
        self._poll_history: list[tuple[str, str, str | None]] = []
        self._window_size = window_size
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._circuit_breaker_limit = circuit_breaker_limit
        self._loop_config = resolve_loop_detection_config(loop_config)
        self._result_hashes: dict[str, str] = {}
        self._result_outputs: dict[str, str] = {}
        self._read_history: list[tuple[str, dict, str]] = []
        self._no_progress_count = 0
        self._soft_warnings = 0
        self._poll_warnings: set[str] = set()

    def _key(self, tool_name: str, args: dict) -> str:
        try:
            return f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        except (TypeError, ValueError):
            return f"{tool_name}:{args}"

    @staticmethod
    def _hash_result(output: str) -> str:
        sample = output[:500]
        h = 0
        for ch in sample:
            h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        return str(h)

    def record(self, tool_name: str, args: dict) -> None:
        self._history.append(self._key(tool_name, args))
        if len(self._history) > self._window_size:
            self._history.pop(0)

    def cache_result_output(self, tool_name: str, args: dict, output: str) -> None:
        """Store truncated output for identical/overlapping reuse stubs."""
        key = self._key(tool_name, args)
        text = output if isinstance(output, str) else str(output or "")
        self._result_outputs[key] = text[:2_000]
        if tool_name in {"read_file", "hashline_read"}:
            self._read_history.append((tool_name, dict(args or {}), text[:2_000]))
            if len(self._read_history) > self._window_size:
                self._read_history.pop(0)

    def reuse_tool_output(self, tool_name: str, args: dict) -> str | None:
        """Return a short stub if this call (or an overlapping read) already ran."""
        from clawagents.loop_detection import detect_overlapping_read

        key = self._key(tool_name, args)
        prior = self._result_outputs.get(key)
        if prior is not None:
            return (
                f"[Reused identical {tool_name} result] Same arguments already ran "
                f"this turn — do not re-call. Prior excerpt "
                f"({min(500, len(prior))} chars):\n{prior[:500]}"
            )
        return detect_overlapping_read(
            tool_name=tool_name,
            params=args or {},
            prior_reads=self._read_history,
        )

    def record_result(self, tool_name: str, args: dict, output: str) -> None:
        """Record the result of a tool call for no-progress detection."""
        from clawagents.loop_detection import hash_tool_call

        key = self._key(tool_name, args)
        result_hash = self._hash_result(output)
        prev_hash = self._result_hashes.get(key)
        if prev_hash == result_hash:
            self._no_progress_count += 1
        else:
            self._no_progress_count = max(0, self._no_progress_count - 1)
        self._result_hashes[key] = result_hash
        self.cache_result_output(tool_name, args, output)
        call_hash = hash_tool_call(tool_name, args)
        self._poll_history.append((tool_name, call_hash, result_hash))
        if len(self._poll_history) > self._window_size:
            self._poll_history.pop(0)

    def is_ping_ponging(self) -> bool:
        """Detect A->B->A->B ping-pong oscillation (last 6 entries)."""
        if len(self._history) < 4:
            return False
        recent = self._history[-6:]
        if len(recent) < 4:
            return False
        unique = set(recent)
        if len(unique) != 2:
            return False
        for i in range(len(recent) - 1):
            if recent[i] == recent[i + 1]:
                return False
        return True

    def is_circuit_broken(self) -> bool:
        """Global circuit breaker: too many no-progress calls."""
        return self._no_progress_count >= self._circuit_breaker_limit

    def _count_occurrences(self, tool_name: str, args: dict) -> int:
        key = self._key(tool_name, args)
        return self._history.count(key)

    def is_soft_looping(self, tool_name: str, args: dict) -> bool:
        return self._count_occurrences(tool_name, args) >= self._soft_limit

    def is_hard_looping(self, tool_name: str, args: dict) -> bool:
        return self._count_occurrences(tool_name, args) >= self._hard_limit

    def is_soft_looping_batch(self, calls: list[ParsedToolCall]) -> bool:
        return any(self.is_soft_looping(c.tool_name, c.args) for c in calls)

    def is_hard_looping_batch(self, calls: list[ParsedToolCall]) -> bool:
        return any(self.is_hard_looping(c.tool_name, c.args) for c in calls)

    def record_batch(self, calls: list[ParsedToolCall]) -> None:
        for c in calls:
            self.record(c.tool_name, c.args)

    def bump_soft_warning(self) -> int:
        self._soft_warnings += 1
        return self._soft_warnings

    def check_known_poll_no_progress(self, tool_name: str, args: dict):
        from clawagents.loop_detection import detect_known_poll_no_progress

        result = detect_known_poll_no_progress(
            tool_name=tool_name,
            params=args,
            history=self._poll_history,
            config=self._loop_config,
        )
        if result and result.stuck and result.warning_key in self._poll_warnings:
            if result.level == "warning":
                return None
        if result and result.stuck and result.warning_key:
            self._poll_warnings.add(result.warning_key)
        return result


# ─── Consecutive Failure Detection ────────────────────────────────────────
# Tracks tool-call success/failure to detect persistent failure streaks.
# When N consecutive tool calls fail, injects a "step back and rethink"
# message — lightweight online adaptation inspired by OpenClaw-RL's
# next-state reward signal.

_RETHINK_THRESHOLD = 3
_MAX_RETHINKS = 3

_RETHINK_MESSAGE = (
    "[System] Your last {n} tool calls all failed. "
    "Stop before trying another workaround. Classify the failure as code/format, "
    "local environment, dependency availability, permission, or external service. "
    "Retry only after relevant state changed. If the evidence says authentication "
    "was rejected, a package is unavailable, or user-owned configuration is needed, "
    "report the exact error and request that action instead of changing runtimes or tools."
)


_SCORELESS_TOOLS: frozenset[str] = frozenset({
    "think", "todolist", "todo_write", "todo_read", "use_skill", "ask_user",
})


class _FailureTracker:
    """Track consecutive tool failures to trigger rethink injection.

    Scoreless tools (think, todolist, etc.) are excluded — their results
    are not meaningful signals for failure detection.
    """

    def __init__(self, threshold: int = _RETHINK_THRESHOLD, max_rethinks: int = _MAX_RETHINKS):
        self._results: list[bool] = []  # True = success, False = failure
        self._threshold = threshold
        self._max_rethinks = max_rethinks
        self._rethink_count = 0

    def record(self, success: bool, tool_name: str = "") -> None:
        if tool_name in _SCORELESS_TOOLS:
            return
        self._results.append(success)

    def record_batch(self, results: list[tuple[bool, str]]) -> None:
        for success, name in results:
            self.record(success, name)

    def should_rethink(self) -> bool:
        if self._rethink_count >= self._max_rethinks:
            return False
        if len(self._results) < self._threshold:
            return False
        return all(not s for s in self._results[-self._threshold:])

    def bump_rethink(self) -> int:
        self._rethink_count += 1
        self._results.clear()
        return self._rethink_count

    @property
    def consecutive_failures(self) -> int:
        count = 0
        for s in reversed(self._results):
            if not s:
                count += 1
            else:
                break
        return count
