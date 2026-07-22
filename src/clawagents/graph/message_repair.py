"""Message repair utilities for the agent loop.

Sanitises assistant text (leaked model-control tokens) and patches
dangling / orphan tool-call messages so the next LLM call sees a valid
transcript.

Extracted from ``agent_loop.py`` for modularity.
"""

from __future__ import annotations

import re

from clawagents.providers.llm import LLMMessage


# ─── Model Control Token Sanitization ─────────────────────────────────────
_MODEL_CONTROL_TOKEN_RE = re.compile(r'<[｜|][^>]*?[｜|]>')

def _sanitize_assistant_text(text: str) -> str:
    """Strip leaked model control tokens from assistant text (GLM-5, DeepSeek, etc.)."""
    return _MODEL_CONTROL_TOKEN_RE.sub('', text).strip()


# ─── Dangling Tool Call Repair (learned from deepagents) ──────────────────
# When native function calling is used and the agent loop is interrupted mid-execution,
# the next LLM call sees tool_calls without matching tool results — most APIs reject this.
# This pass inserts synthetic "cancelled" responses for any dangling tool calls.
# It also drops orphan role="tool" messages whose tool_call_id was never declared
# by a preceding assistant tool_calls_meta (common after session preload limit
# cuts mid-pair → OpenAI 400: "messages with role 'tool' must be a response to
# a preceding message with 'tool_calls'").

def _patch_dangling_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    if not messages:
        return messages

    # Ids declared by assistant messages in this transcript.
    declared_ids: set[str] = set()
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls_meta:
            for tc in msg.tool_calls_meta:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    declared_ids.add(str(tc_id))

    # Drop orphan tool results first (no matching assistant tool_calls).
    filtered: list[LLMMessage] = []
    for msg in messages:
        if msg.role == "tool":
            tc_id = str(msg.tool_call_id) if msg.tool_call_id else ""
            if not tc_id or tc_id not in declared_ids:
                continue
        filtered.append(msg)

    # Build set of all tool_call_ids that have a matching role="tool" response
    responded_ids: set[str] = set()
    for msg in filtered:
        if msg.role == "tool" and msg.tool_call_id:
            responded_ids.add(str(msg.tool_call_id))

    patched: list[LLMMessage] = []
    for i, msg in enumerate(filtered):
        patched.append(msg)

        # Text-mode: look for assistant messages with JSON tool calls without a following [Tool Result]
        if msg.role == "assistant" and isinstance(msg.content, str) and msg.content.startswith('{"tool":'):
            _next_msg = filtered[i + 1] if i + 1 < len(filtered) else None
            _next_content = _next_msg.content if _next_msg is not None else None
            has_result = (
                _next_msg is not None
                and _next_msg.role == "user"
                and isinstance(_next_content, str)
                and _next_content.startswith("[Tool Result]")
            )
            if not has_result:
                patched.append(LLMMessage(
                    role="user",
                    content="[Tool Result] Tool call was cancelled — the agent was interrupted before it could complete.",
                ))

        # Native tool calls: inject synthetic role="tool" for any missing responses
        elif msg.role == "assistant" and msg.tool_calls_meta:
            for tc in msg.tool_calls_meta:
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id and str(tc_id) not in responded_ids:
                    patched.append(LLMMessage(
                        role="tool",
                        content="Tool call was cancelled — the agent was interrupted before it could complete.",
                        tool_call_id=str(tc_id),
                    ))
                    responded_ids.add(str(tc_id))

    return patched


def _drop_leading_orphan_tools(messages: list[LLMMessage]) -> list[LLMMessage]:
    """If a limited session preload starts mid tool-pair, drop leading orphans."""
    if not messages:
        return messages
    i = 0
    while i < len(messages) and messages[i].role == "tool":
        i += 1
    return messages[i:] if i else messages
