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
