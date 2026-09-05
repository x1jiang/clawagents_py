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

# User-pinned "always-on" context. Rides at the very END of the system message
# (after tools, rules and dynamic layers) so it is the last instruction the
# model reads before the conversation — the strongest position short of the
# user turn itself. Re-upserted every LLM round.
PINNED_BEGIN = "<!--clawagents:pinned-->"
PINNED_END = "<!--/clawagents:pinned-->"
PINNED_HEADING = "## Pinned context (always applies)"

_PINNED_BLOCK_RE = re.compile(
    r"\n*" + re.escape(PINNED_BEGIN) + r"[\s\S]*?" + re.escape(PINNED_END) + r"\n?",
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


def gemini_tool_use_section(
    provider: Optional[str],
    model: Optional[str],
) -> str:
    """Stop Gemini Flash from inventing tool results when it skipped the call."""
    blob = f"{provider or ''} {model or ''}".casefold()
    if "gemini" not in blob:
        return ""
    return (
        "## Tool use (Gemini)\n"
        "Native tools (`use_skill`, `list_skills`, `execute`, and the other "
        "declared functions) are available this turn. Never invent SQL counts, "
        "file contents, or command output. `use_skill` is instructions, not "
        "data — every count in a table must appear in this-turn `execute` "
        "output. If a daily total is all you have, say so; do not invent "
        "hourly or weekday cells. If you have not called `execute` this turn, "
        "say you have not run it — do not claim a query already executed."
    )


def append_model_identity(
    base_prompt: str,
    provider: Optional[str],
    model: Optional[str],
) -> str:
    """Append identity (and Gemini tool-honesty) unless already present."""
    base = base_prompt or ""
    for block in (
        model_identity_section(provider, model),
        gemini_tool_use_section(provider, model),
    ):
        if not block:
            continue
        heading = block.split("\n", 1)[0]
        if heading and heading in base:
            continue
        base = f"{base.rstrip()}\n\n{block}"
    return base


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


def build_pinned_block(text: Optional[str]) -> str:
    """Wrap user-pinned context in its tagged, precedence-framed block."""
    body = (text or "").strip()
    if not body:
        return ""
    return (
        f"{PINNED_BEGIN}\n"
        f"{PINNED_HEADING}\n"
        "The user pinned the following instructions for this workspace. They apply "
        "to every turn, including after context compaction, and take precedence "
        "over the project rules, tool notes, and defaults above when they conflict.\n\n"
        f"{body}\n"
        f"{PINNED_END}"
    )


def strip_pinned_context(content: str) -> str:
    """Remove a previously upserted pinned-context block."""
    if not content:
        return content
    return _PINNED_BLOCK_RE.sub("", content)


def append_pinned_context(
    messages: Sequence[Any],
    text: Optional[str],
) -> list:
    """Upsert pinned context as the LAST block of the system message.

    Empty ``text`` removes any existing block. Only the system message is
    touched; user turns are never rewritten.
    """
    result = list(messages)
    block = build_pinned_block(text)
    for index, message in enumerate(result):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "system":
            continue
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content or "")
        content = strip_pinned_context(content).rstrip()
        new_content = f"{content}\n\n{block}\n" if block else f"{content}\n"
        result[index] = LLMMessage(role="system", content=new_content)
        return result
    return result
