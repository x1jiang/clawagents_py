"""Prompt assembly helpers shared by ClawAgents runtimes."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from clawagents.providers.llm import LLMMessage

PROMPT_CACHE_BOUNDARY = "__CACHE_BOUNDARY__"

def build_system_prompt(
    base_prompt: str,
    tool_description: Optional[str] = "",
    lesson_preamble: Optional[str] = "",
    cache_boundary: str = PROMPT_CACHE_BOUNDARY,
) -> str:
    """Build the static system prompt prefix used before dynamic messages."""
    return f"{base_prompt}{lesson_preamble or ''}\n\n{tool_description or ''}\n{cache_boundary}"


def build_prompt_injection(
    memory_content: Optional[str] = None,
    skill_summaries: Optional[str] = None,
) -> Optional[str]:
    parts = [part for part in (memory_content, skill_summaries) if part]
    return "\n\n".join(parts) if parts else None


def append_prompt_injection(
    messages: Sequence[Any],
    injection: Optional[str],
) -> Sequence[Any]:
    if not injection:
        return messages

    result = list(messages)
    for index, message in enumerate(result):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "system":
            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
            result[index] = LLMMessage(
                role="system",
                content=f"{content}\n\n{injection}",
            )
            return result

    return messages
