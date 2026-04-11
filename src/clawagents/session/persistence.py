"""Session persistence for ClawAgents.

Saves and restores agent sessions as append-only JSONL files.
Each line is a typed event: turn_started, assistant_message,
tool_use, tool_result, turn_completed, usage, system_prompt.

Inspired by claw-code-main's session.rs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clawagents.providers.llm import LLMMessage

_SESSIONS_DIR = ".clawagents/sessions"


def _sessions_path() -> Path:
    return Path.cwd() / _SESSIONS_DIR


def _generate_session_id() -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    return f"session-{ts}"


@dataclass
class SessionInfo:
    """Summary of a saved session."""
    session_id: str
    path: Path
    created_ts: float
    turn_count: int
    task: str
    status: str


class SessionWriter:
    """Append-only JSONL writer for session events."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or _generate_session_id()
        self.dir = _sessions_path()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.session_id}.jsonl"
        self._turn_count = 0

    def append(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        event = {"type": event_type, "ts": time.time()}
        if data:
            event.update(data)
        from clawagents.utils.atomic_write import atomic_write_text
        existing = ""
        try:
            existing = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        atomic_write_text(self.path, existing + json.dumps(event, default=str) + "\n")

    def write_system_prompt(self, content: str) -> None:
        self.append("system_prompt", {"content": content})

    def write_turn_started(self, iteration: int) -> None:
        self._turn_count += 1
        self.append("turn_started", {"iteration": iteration})

    def write_assistant_message(
        self, content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        thinking: str | None = None,
    ) -> None:
        data: dict[str, Any] = {"content": content}
        if tool_calls:
            data["tool_calls"] = tool_calls
        if thinking:
            data["thinking"] = thinking
        self.append("assistant_message", data)

    def write_tool_result(
        self, tool_call_id: str, tool_name: str,
        success: bool, output: str, error: str | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "success": success,
            "output": output[:2000],  # cap for file size
        }
        if error:
            data["error"] = error[:500]
        self.append("tool_result", data)

    def write_usage(
        self, tokens_used: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        self.append("usage", {
            "tokens_used": tokens_used,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
        })

    def write_turn_completed(
        self, iteration: int, tool_calls: int, status: str,
    ) -> None:
        self.append("turn_completed", {
            "iteration": iteration,
            "tool_calls": tool_calls,
            "status": status,
        })


class SessionReader:
    """Read a JSONL session file and reconstruct messages."""

    def __init__(self, path: Path):
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.events.append(json.loads(line))

    def reconstruct_messages(self) -> list[LLMMessage]:
        """Rebuild LLMMessage list from session events."""
        messages: list[LLMMessage] = []

        for ev in self.events:
            ev_type = ev["type"]

            if ev_type == "system_prompt":
                messages.append(LLMMessage(role="system", content=ev["content"]))

            elif ev_type == "assistant_message":
                tool_calls_meta = None
                if ev.get("tool_calls"):
                    tool_calls_meta = [
                        {"id": tc["id"], "name": tc["name"], "args": tc.get("args", {})}
                        for tc in ev["tool_calls"]
                    ]
                messages.append(LLMMessage(
                    role="assistant",
                    content=ev.get("content", ""),
                    tool_calls_meta=tool_calls_meta,
                    thinking=ev.get("thinking"),
                ))

            elif ev_type == "tool_result":
                messages.append(LLMMessage(
                    role="tool",
                    content=ev.get("output", ""),
                    tool_call_id=ev.get("tool_call_id"),
                ))

        return messages

    def get_task(self) -> str:
        """Extract the original task from the first user message after system prompt."""
        for ev in self.events:
            if ev["type"] == "turn_started":
                # The task is the first user message
                break
        # Look for user messages in reconstructed
        msgs = self.reconstruct_messages()
        for m in msgs:
            if m.role == "user":
                return m.content if isinstance(m.content, str) else str(m.content)
        return ""

    def get_summary(self) -> SessionInfo:
        turn_count = sum(1 for ev in self.events if ev["type"] == "turn_completed")
        task = self.get_task()
        last_status = "unknown"
        for ev in reversed(self.events):
            if ev["type"] == "turn_completed":
                last_status = ev.get("status", "unknown")
                break
        created = self.events[0]["ts"] if self.events else 0

        return SessionInfo(
            session_id=self.path.stem,
            path=self.path,
            created_ts=created,
            turn_count=turn_count,
            task=task[:100],
            status=last_status,
        )


def list_sessions(limit: int = 20) -> list[SessionInfo]:
    """List saved sessions, most recent first."""
    sessions_dir = _sessions_path()
    if not sessions_dir.exists():
        return []

    infos: list[SessionInfo] = []
    files = sorted(sessions_dir.glob("session-*.jsonl"), reverse=True)
    for f in files[:limit]:
        try:
            reader = SessionReader(f)
            infos.append(reader.get_summary())
        except (json.JSONDecodeError, OSError):
            continue

    return infos
