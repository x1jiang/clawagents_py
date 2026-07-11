"""Trim overly verbose assistant / tool narration before it re-enters context."""

from __future__ import annotations

from clawagents.providers.llm import LLMMessage

DEFAULT_ASSISTANT_CHARS = 12_000
DEFAULT_USER_CHARS = 16_000


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n... [{omitted} chars trimmed from verbose turn] ...\n{text[-tail:]}"


def trim_verbose_messages(
    messages: list[LLMMessage],
    *,
    assistant_chars: int = DEFAULT_ASSISTANT_CHARS,
    user_chars: int = DEFAULT_USER_CHARS,
) -> tuple[list[LLMMessage], int]:
    """Soft-cap long assistant/user string turns (skips tool role — crushers own those)."""
    changed = 0
    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "tool" or not isinstance(m.content, str):
            out.append(m)
            continue
        limit = assistant_chars if m.role == "assistant" else user_chars
        if m.role not in ("assistant", "user") or len(m.content) <= limit:
            out.append(m)
            continue
        # Don't trim compacted history markers aggressively
        if m.content.startswith("[System — Compacted History]"):
            out.append(m)
            continue
        out.append(
            LLMMessage(
                role=m.role,
                content=_trim(m.content, limit),
                tool_call_id=getattr(m, "tool_call_id", None),
                tool_calls_meta=getattr(m, "tool_calls_meta", None),
                gemini_parts=getattr(m, "gemini_parts", None),
                thinking=getattr(m, "thinking", None),
            )
        )
        changed += 1
    return out, changed
