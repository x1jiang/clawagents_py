"""Prompt assembly helpers shared by ClawAgents runtimes."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from clawagents.prompts.cache_align import normalize_stable_prefix
from clawagents.providers.llm import LLMMessage

PROMPT_CACHE_BOUNDARY = "__CACHE_BOUNDARY__"


def build_system_prompt(
    base_prompt: str,
    tool_description: Optional[str] = "",
    lesson_preamble: Optional[str] = "",
    cache_boundary: str = PROMPT_CACHE_BOUNDARY,
) -> str:
    """Build system prompt with a stable cacheable prefix.

    Layout::

        <normalized base + tools>   # static — provider KV-cache friendly
        __CACHE_BOUNDARY__
        <lesson preamble>           # dynamic — may change across runs

    Lessons sit *after* the boundary so PTRL updates do not bust the prefix cache.
    """
    static = normalize_stable_prefix(
        f"{base_prompt or ''}\n\n{tool_description or ''}".rstrip()
    )
    dynamic = (lesson_preamble or "").strip()
    if dynamic:
        return f"{static}\n{cache_boundary}\n{dynamic}\n"
    return f"{static}\n{cache_boundary}"


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
            # Idempotent: before_llm hooks run every loop round, so without
            # this check the injection accumulated one copy per round —
            # wasting tokens and churning the prompt-prefix cache.
            if isinstance(content, str) and injection in content:
                return messages
            # Prefer appending after the cache boundary when present so the
            # static prefix (instructions + tools) stays byte-stable.
            if isinstance(content, str) and PROMPT_CACHE_BOUNDARY in content:
                prefix, _, suffix = content.partition(PROMPT_CACHE_BOUNDARY)
                new_content = f"{prefix}{PROMPT_CACHE_BOUNDARY}\n{injection}\n{suffix}".rstrip() + "\n"
            else:
                new_content = f"{content}\n\n{injection}"
            result[index] = LLMMessage(
                role="system",
                content=new_content,
            )
            return result

    return messages
