"""Prompt assembly helpers shared by ClawAgents runtimes."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from clawagents.prompts.cache_align import normalize_stable_prefix
from clawagents.providers.llm import LLMMessage

PROMPT_CACHE_BOUNDARY = "__CACHE_BOUNDARY__"
INJECTION_BEGIN = "<!--clawagents:injection-->"
INJECTION_END = "<!--/clawagents:injection-->"

_INJECTION_BLOCK_RE = re.compile(
    re.escape(INJECTION_BEGIN) + r"[\s\S]*?" + re.escape(INJECTION_END) + r"\n?",
    re.MULTILINE,
)


def model_identity_section(
    provider: Optional[str],
    model: Optional[str],
) -> str:
    """Stable identity line keyed to the configured provider/model.

    Stops proxy/Gemini models from inventing \"I am Claude 3.5\" / \"trained by
    Google\" when the session is actually a different configured model.
    """
    model_id = (model or "").strip()
    if not model_id:
        return ""
    provider_id = (provider or "unknown").strip() or "unknown"
    return (
        "## Model identity\n"
        f"You are ClawAgent. The configured model for this session is "
        f"`{provider_id}/{model_id}`. Do not claim to be a different model, "
        "vendor, or training lineage."
    )


def append_model_identity(
    base_prompt: str,
    provider: Optional[str],
    model: Optional[str],
) -> str:
    """Append :func:`model_identity_section` unless already present."""
    base = base_prompt or ""
    block = model_identity_section(provider, model)
    if not block or "## Model identity" in base:
        return base
    return f"{base.rstrip()}\n\n{block}"


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


def strip_prompt_injection(content: str) -> str:
    """Remove a previously upserted clawagents injection block."""
    if not content:
        return content
    return _INJECTION_BLOCK_RE.sub("", content)


def append_prompt_injection(
    messages: Sequence[Any],
    injection: Optional[str],
) -> Sequence[Any]:
    """Upsert memory/skills injection into the system message.

    Replaces any prior ``<!--clawagents:injection-->`` block so per-turn skill
    ranking / reloaded rules do not accumulate copies.
    """
    if not injection:
        return messages

    block = f"{INJECTION_BEGIN}\n{injection}\n{INJECTION_END}"
    result = list(messages)
    for index, message in enumerate(result):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "system":
            continue
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content or "")
        content = strip_prompt_injection(content)
        if PROMPT_CACHE_BOUNDARY in content:
            prefix, _, suffix = content.partition(PROMPT_CACHE_BOUNDARY)
            new_content = f"{prefix}{PROMPT_CACHE_BOUNDARY}\n{block}\n{suffix.lstrip()}".rstrip() + "\n"
        else:
            new_content = f"{content.rstrip()}\n\n{block}"
        result[index] = LLMMessage(role="system", content=new_content)
        return result

    return messages
