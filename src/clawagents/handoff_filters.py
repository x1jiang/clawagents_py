"""Built-in :data:`~clawagents.handoffs.InputFilter` helpers.

These transform a :class:`~clawagents.handoffs.HandoffInputData` before
the new agent takes over. They mirror the
``openai-agents-python/src/agents/extensions/handoff_filters.py`` API
but operate on :class:`~clawagents.providers.llm.LLMMessage` lists, the
shape clawagents uses internally.
"""

from __future__ import annotations

from typing import Iterable

from clawagents.handoffs import HandoffInputData
from clawagents.providers.llm import LLMMessage

__all__ = [
    "remove_all_tools",
    "nest_handoff_history",
]


def _is_tool_related(msg: LLMMessage) -> bool:
    """True for tool-call assistant messages and tool-result messages.

    Specifically:
      * ``role == "tool"`` — synthetic tool-result messages
      * ``role == "assistant"`` with native ``tool_calls_meta`` populated
      * ``role == "assistant"`` whose JSON content opens a single/multi
        tool call (text-mode fallback)
    """
    if msg.role == "tool":
        return True
    if msg.role == "assistant":
        if getattr(msg, "tool_calls_meta", None):
            return True
        content = msg.content if isinstance(msg.content, str) else ""
        stripped = content.strip()
        if stripped.startswith('{"tool":') or stripped.startswith("[{"):
            return True
    return False


def _is_tool_result_user(msg: LLMMessage) -> bool:
    """Detect text-mode synthetic tool-result user messages.

    The non-native dispatch path appends ``LLMMessage(role="user",
    content="[Tool Result] ...")`` after each tool call. Strip those too
    so the new agent doesn't see orphaned tool output.
    """
    if msg.role != "user":
        return False
    content = msg.content if isinstance(msg.content, str) else ""
    return content.startswith("[Tool Result]") or content.startswith("[Tool Results]")


def _filter_messages(messages: Iterable[LLMMessage]) -> list[LLMMessage]:
    return [m for m in messages if not (_is_tool_related(m) or _is_tool_result_user(m))]


def remove_all_tools(data: HandoffInputData) -> HandoffInputData:
    """Strip all tool-call / tool-result exchanges from the conversation.

    Useful when the new agent should reason from scratch on the user's
    intent without inheriting the previous agent's tool soup.
    """
    return HandoffInputData(
        input_history=_filter_messages(data.input_history),
        pre_handoff_items=list(data.pre_handoff_items),
        new_items=list(data.new_items),
        run_context=data.run_context,
    )


def nest_handoff_history(data: HandoffInputData) -> HandoffInputData:
    """Replace prior history with a single nested summary user message.

    The previous transcript is collapsed to a marker so the new agent
    starts on a fresh context window but still knows a handoff happened.
    The first system message (if any) and the most recent user message
    are preserved verbatim.
    """
    history = list(data.input_history)
    if not history:
        return data

    system_msg = next((m for m in history if m.role == "system"), None)
    last_user = next((m for m in reversed(history) if m.role == "user"), None)

    nested: list[LLMMessage] = []
    if system_msg is not None:
        nested.append(system_msg)
    nested.append(
        LLMMessage(
            role="user",
            content=(
                "[Handoff] The previous agent transferred this conversation. "
                "The full prior transcript has been summarised away. Continue "
                "from the most recent user request below."
            ),
        )
    )
    if last_user is not None and last_user is not system_msg:
        nested.append(last_user)

    return HandoffInputData(
        input_history=nested,
        pre_handoff_items=list(data.pre_handoff_items),
        new_items=list(data.new_items),
        run_context=data.run_context,
    )
