from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clawagents.providers.llm import LLMMessage


def _message_to_dict(m: LLMMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_call_id:
        data["tool_call_id"] = m.tool_call_id
    if m.tool_calls_meta:
        data["tool_calls_meta"] = m.tool_calls_meta
    if m.thinking:
        data["thinking"] = m.thinking
    return data


def _dict_to_message(d: dict[str, Any]) -> LLMMessage:
    return LLMMessage(
        role=d["role"],
        content=d.get("content", ""),
        tool_call_id=d.get("tool_call_id"),
        tool_calls_meta=d.get("tool_calls_meta"),
        thinking=d.get("thinking"),
    )


@dataclass
class RunResult:
    messages: list[LLMMessage]
    task: str
    status: str
    final_output: Any
    result: str
    iterations: int
    max_iterations: int
    tool_calls: int
    trajectory_file: str = ""
    session_file: str = ""
    interruptions: list[dict[str, Any]] = field(default_factory=list)
    new_items: list[LLMMessage] = field(default_factory=list)

    @classmethod
    def from_agent_state(
        cls,
        state: Any,
        *,
        new_items: list[LLMMessage] | None = None,
        interruptions: list[dict[str, Any]] | None = None,
    ) -> "RunResult":
        return cls(
            messages=list(state.messages),
            task=str(state.current_task),
            status=str(state.status),
            final_output=state.final_output if state.final_output is not None else state.result,
            result=str(state.result),
            iterations=int(state.iterations),
            max_iterations=int(state.max_iterations),
            tool_calls=int(state.tool_calls),
            trajectory_file=str(getattr(state, "trajectory_file", "") or ""),
            session_file=str(getattr(state, "session_file", "") or ""),
            interruptions=list(interruptions or []),
            new_items=list(new_items or state.messages),
        )

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RunResult":
        return cls(
            messages=[_dict_to_message(m) for m in state.get("messages", [])],
            task=str(state.get("task", "")),
            status=str(state.get("status", "")),
            final_output=state.get("final_output"),
            result=str(state.get("result", "")),
            iterations=int(state.get("iterations", 0)),
            max_iterations=int(state.get("max_iterations", 0)),
            tool_calls=int(state.get("tool_calls", 0)),
            trajectory_file=str(state.get("trajectory_file") or ""),
            session_file=str(state.get("session_file") or ""),
            interruptions=list(state.get("interruptions") or []),
            new_items=[_dict_to_message(m) for m in state.get("new_items", [])],
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "messages": [_message_to_dict(m) for m in self.messages],
            "task": self.task,
            "status": self.status,
            "final_output": self.final_output,
            "result": self.result,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "tool_calls": self.tool_calls,
            "trajectory_file": self.trajectory_file,
            "session_file": self.session_file,
            "interruptions": list(self.interruptions),
            "new_items": [_message_to_dict(m) for m in self.new_items],
        }

    async def resume_into(self, session: Any) -> None:
        await session.add_items(self.messages)
