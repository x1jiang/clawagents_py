"""Configurable tool loop detection (OpenClaw 2026.6.1 pattern)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

LoopLevel = Literal["warning", "critical"]
LoopDetector = Literal[
    "generic_repeat",
    "known_poll_no_progress",
    "ping_pong",
    "global_circuit_breaker",
]


DEFAULT_KNOWN_POLL_TOOLS: frozenset[str] = frozenset(
    {
        "execute",
        "task_status",
        "task_output",
        "read_file",
        "glob",
        "grep",
        "web_fetch",
        "web_search",
        "browser_snapshot",
    }
)


@dataclass
class LoopDetectionDetectors:
    generic_repeat: bool = True
    known_poll_no_progress: bool = True
    ping_pong: bool = True
    global_circuit_breaker: bool = True


@dataclass
class LoopDetectionConfig:
    """Loop detection policy. ClawAgents defaults to enabled (unlike OpenClaw)."""

    enabled: bool = True
    warning_threshold: int = 3
    critical_threshold: int = 6
    global_circuit_breaker_threshold: int = 30
    known_poll_tools: frozenset[str] = DEFAULT_KNOWN_POLL_TOOLS
    detectors: LoopDetectionDetectors = field(default_factory=LoopDetectionDetectors)


@dataclass
class LoopDetectionResult:
    stuck: bool
    level: LoopLevel | None = None
    detector: LoopDetector | None = None
    count: int = 0
    message: str = ""
    warning_key: str = ""


def resolve_loop_detection_config(config: LoopDetectionConfig | None) -> LoopDetectionConfig:
    return config or LoopDetectionConfig()


def hash_tool_call(tool_name: str, params: dict[str, Any]) -> str:
    try:
        return f"{tool_name}:{json.dumps(params, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        return f"{tool_name}:{params!r}"


def is_known_poll_tool_call(tool_name: str, params: dict[str, Any], config: LoopDetectionConfig) -> bool:
    if tool_name not in config.known_poll_tools:
        return False
    if tool_name == "execute":
        cmd = params.get("command") or params.get("cmd") or ""
        return bool(str(cmd).strip())
    if tool_name in {"read_file", "glob", "grep"}:
        return bool(params.get("path") or params.get("pattern") or params.get("glob_pattern"))
    return True


def get_no_progress_streak(
    history: list[tuple[str, str, str | None]],
    tool_name: str,
    call_hash: str,
) -> tuple[int, str | None]:
    """Return (streak, latest_result_hash) for identical call+result pairs."""
    streak = 0
    latest_result: str | None = None
    for name, ch, rh in reversed(history):
        if name == tool_name and ch == call_hash:
            streak += 1
            if latest_result is None:
                latest_result = rh
            elif rh != latest_result:
                break
        elif streak:
            break
    return streak, latest_result


def _read_range(params: dict[str, Any]) -> tuple[int, int] | None:
    """Return (start, end_exclusive) for read_file-style args, or None if unbounded."""
    try:
        offset = int(params.get("offset") or params.get("start_line") or 0)
    except (TypeError, ValueError):
        offset = 0
    offset = max(0, offset)
    limit_raw = params.get("limit") or params.get("end_line") or params.get("max_lines")
    if limit_raw is None or limit_raw == "":
        return None
    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    # end_line semantics: if end_line present without limit, treat as absolute end.
    if params.get("end_line") is not None and params.get("limit") is None:
        end = max(offset, int(params["end_line"]))
        return offset, end
    return offset, offset + limit


def ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def detect_overlapping_read(
    *,
    tool_name: str,
    params: dict[str, Any],
    prior_reads: list[tuple[str, dict[str, Any], str]],
) -> str | None:
    """If this read overlaps a prior successful read of the same path, return a stub."""
    if tool_name not in {"read_file", "hashline_read"}:
        return None
    path = str(params.get("path") or params.get("file_path") or "").strip()
    if not path:
        return None
    new_range = _read_range(params)
    for prior_name, prior_params, prior_out in reversed(prior_reads):
        if prior_name not in {"read_file", "hashline_read"}:
            continue
        prior_path = str(
            prior_params.get("path") or prior_params.get("file_path") or ""
        ).strip()
        if prior_path != path:
            continue
        # Exact args already handled by identical-call reuse; here catch paging overlap.
        prior_range = _read_range(prior_params)
        if new_range is None and prior_range is None:
            # Both unbounded full-file reads of the same path.
            return (
                f"[Reused prior {prior_name} of {path}] Same unbounded read already "
                f"ran this turn. Use that result; do not page the file again.\n"
                f"Prior excerpt ({min(400, len(prior_out))} chars):\n{prior_out[:400]}"
            )
        if new_range is not None and prior_range is not None and ranges_overlap(
            new_range, prior_range
        ):
            return (
                f"[Reused overlapping {prior_name} of {path}] "
                f"Requested lines ~{new_range[0]}–{new_range[1]} overlap prior "
                f"~{prior_range[0]}–{prior_range[1]}. Use the prior result or grep "
                f"for a symbol instead of sequential paging.\n"
                f"Prior excerpt ({min(400, len(prior_out))} chars):\n{prior_out[:400]}"
            )
        if new_range is None and prior_range is not None:
            return (
                f"[Reused prior {prior_name} of {path}] A partial read already "
                f"covered part of this file; avoid a full re-read. Grep for symbols "
                f"or widen a single bounded window once.\n"
                f"Prior excerpt ({min(400, len(prior_out))} chars):\n{prior_out[:400]}"
            )
    return None


def detect_known_poll_no_progress(
    *,
    tool_name: str,
    params: dict[str, Any],
    history: list[tuple[str, str, str | None]],
    config: LoopDetectionConfig | None = None,
) -> LoopDetectionResult | None:
    """OpenClaw-style early critical/warning for poll tools with no progress."""
    resolved = resolve_loop_detection_config(config)
    if not resolved.enabled or not resolved.detectors.known_poll_no_progress:
        return None
    if not is_known_poll_tool_call(tool_name, params, resolved):
        return None
    call_hash = hash_tool_call(tool_name, params)
    streak, result_hash = get_no_progress_streak(history, tool_name, call_hash)
    if streak >= resolved.critical_threshold:
        return LoopDetectionResult(
            stuck=True,
            level="critical",
            detector="known_poll_no_progress",
            count=streak,
            message=(
                f"CRITICAL: Called {tool_name} with identical arguments and no progress "
                f"{streak} times. This appears to be a stuck polling loop."
            ),
            warning_key=f"poll:{tool_name}:{call_hash}:{result_hash or 'none'}",
        )
    if streak >= resolved.warning_threshold:
        return LoopDetectionResult(
            stuck=True,
            level="warning",
            detector="known_poll_no_progress",
            count=streak,
            message=(
                f"WARNING: You have called {tool_name} {streak} times with identical "
                "arguments and no progress. Stop polling or report failure."
            ),
            warning_key=f"poll:{tool_name}:{call_hash}:{result_hash or 'none'}",
        )
    return None
