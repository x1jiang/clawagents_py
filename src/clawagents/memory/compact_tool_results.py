"""Compact oversized tool results before summarization (DeepAgents 1.10.2)."""

from __future__ import annotations

from clawagents.memory.content_budgets import ContentBudgets, DEFAULT_CONTENT_BUDGETS
from clawagents.providers.llm import LLMMessage


def _content_chars(content: str | list) -> int:
    if isinstance(content, str):
        return len(content)
    return len(str(content))


def compact_tool_results(
    messages: list[LLMMessage],
    *,
    max_input_tokens: int,
    token_multiplier: float = 1.0,
    headroom_ratio: float = 0.7,
    budgets: ContentBudgets | None = None,
) -> tuple[list[LLMMessage], bool]:
    """Truncate individual tool messages when their collective size exceeds budget."""
    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    if not tool_indices:
        return messages, False

    b = budgets or DEFAULT_CONTENT_BUDGETS
    non_tool_chars = sum(_content_chars(m.content) for i, m in enumerate(messages) if i not in tool_indices)
    adjusted_max = max(int(max_input_tokens / max(token_multiplier, 0.1)), 1000)
    # Tools share of the window × harness headroom (char ≈ token*4).
    budget_for_tools = max(
        int(adjusted_max * 4 * b.tools * headroom_ratio) - non_tool_chars,
        4000,
    )
    per_tool_chars = max(budget_for_tools // len(tool_indices), 500)

    modified = False
    out = list(messages)
    for idx in tool_indices:
        m = messages[idx]
        content = m.content if isinstance(m.content, str) else str(m.content)
        if len(content) > per_tool_chars:
            out[idx] = LLMMessage(
                role=m.role,
                content=content[:per_tool_chars] + "\n...(result truncated before compaction)",
                tool_call_id=getattr(m, "tool_call_id", None),
                tool_calls_meta=getattr(m, "tool_calls_meta", None),
                thinking=getattr(m, "thinking", None),
            )
            modified = True
    return out, modified
