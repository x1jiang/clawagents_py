"""Provider-neutral message/response types, the thinking-stream filter, and the LLMProvider interface.

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Literal
# ─── Feature H: Thinking Token Preservation ────────────────────────────────

import re as _re

# ─── Public Types ──────────────────────────────────────────────────────────


class LLMMessage:
    def __init__(
        self,
        role: Literal["system", "user", "assistant", "tool"],
        content: str | list[dict[str, Any]],
        tool_call_id: str | None = None,
        tool_calls_meta: list[dict[str, Any]] | None = None,
        gemini_parts: list[dict[str, Any]] | None = None,
        thinking: str | None = None,
        added_tool_names: list[str] | None = None,
        anthropic_blocks: list[dict[str, Any]] | None = None,
    ):
        self.anthropic_blocks = deepcopy(anthropic_blocks)
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id          # For role="tool": the ID this result belongs to
        self.tool_calls_meta = tool_calls_meta    # For role="assistant": list of {id, name, args}
        self.gemini_parts = gemini_parts          # Preserved Gemini response parts (thought/thought_signature)
        self.thinking = thinking                  # Feature H: preserved <think> block content
        # For role="tool": tools this result activated. Lets the provider layer
        # reference them from the transcript instead of appending them to the
        # tool list, which would rewrite the cached prompt prefix.
        self.added_tool_names = added_tool_names


class NativeToolSchema:
    """Schema for a tool that can be passed to LLM native function calling."""
    __slots__ = ("name", "description", "parameters")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict[str, Any]],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters


class NativeToolCall:
    """A structured tool call returned by the LLM's native function calling."""
    __slots__ = ("tool_name", "args", "tool_call_id")

    def __init__(self, tool_name: str, args: dict[str, Any], tool_call_id: str = ""):
        self.tool_name = tool_name
        self.args = args
        self.tool_call_id = tool_call_id


class LLMResponse:
    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int,
        partial: bool = False,
        tool_calls: list[NativeToolCall] | None = None,
        gemini_parts: list[dict[str, Any]] | None = None,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        prompt_tokens: int = 0,
        thinking: str | None = None,
        finish_reason: str | None = None,
        reasoning_tokens: int = 0,
        anthropic_blocks: list[dict[str, Any]] | None = None,
    ):
        self.anthropic_blocks = deepcopy(anthropic_blocks)
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.partial = partial
        self.tool_calls = tool_calls
        self.gemini_parts = gemini_parts          # Preserved Gemini response parts (thought/thought_signature)
        # Prompt cache tracking (Claude Code pattern)
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens
        self.prompt_tokens = prompt_tokens
        # Separate reasoning channel (SGLang/vLLM ``reasoning_content``). Never
        # re-sent to the provider; the loop uses it for doom-loop detection and
        # the ``assistant_message`` event.
        self.thinking = thinking
        # Provider stop reason when known ("stop", "tool_calls", "length", …).
        # ``"length"`` lets the loop recover a turn that hit ``max_tokens``
        # mid-reasoning instead of treating the empty content as an answer.
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens


OnChunkCallback = (
    Callable[[str], Coroutine[Any, Any, None]] | Callable[[str], None] | None
)

# An unclosed block (the model hit max_tokens mid-thought) is still thinking,
# never answer text: match to end-of-string when ``</think>`` is missing.
_THINK_BLOCK_RE = _re.compile(r"<think>(.*?)(?:</think>|\Z)", _re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_thinking_tokens(content: str) -> tuple[str, str | None]:
    """Extract <think>...</think> blocks and return (clean_content, thinking).

    Handles models like Qwen3, DeepSeek that wrap chain-of-thought in <think> tags.
    Returns the content with thinking removed, and the thinking text separately.
    A block truncated before ``</think>`` counts as thinking to the end.
    """
    if not content or _THINK_OPEN not in content:
        return content, None
    thinking_parts: list[str] = []
    for m in _THINK_BLOCK_RE.finditer(content):
        thinking_parts.append(m.group(1).strip())
    clean = _THINK_BLOCK_RE.sub("", content).strip()
    thinking = "\n".join(thinking_parts) if thinking_parts else None
    return clean, thinking


class _ThinkStreamFilter:
    """Split a streamed text channel into visible text and inline ``<think>`` blocks.

    Tags can straddle chunk boundaries, so a suffix that could still become a
    tag is held back until the next chunk (or ``flush``). Text inside an
    unclosed block at end-of-stream is thinking, not answer.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    @staticmethod
    def _partial_suffix(buf: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(buf)), 0, -1):
            if tag.startswith(buf[-n:]):
                return n
        return 0

    def feed(self, text: str) -> tuple[str, str]:
        self._buf += text
        visible: list[str] = []
        thinking: list[str] = []
        while self._buf:
            tag = _THINK_CLOSE if self._inside else _THINK_OPEN
            sink = thinking if self._inside else visible
            i = self._buf.find(tag)
            if i >= 0:
                sink.append(self._buf[:i])
                self._buf = self._buf[i + len(tag):]
                self._inside = not self._inside
                continue
            keep = self._partial_suffix(self._buf, tag)
            cut = len(self._buf) - keep
            sink.append(self._buf[:cut])
            self._buf = self._buf[cut:]
            break
        return "".join(visible), "".join(thinking)

    def flush(self) -> tuple[str, str]:
        rest, self._buf = self._buf, ""
        if self._inside:
            return "", rest
        return rest, ""


def rebuild_thinking_content(content: str, thinking: str | None) -> str:
    """Re-attach thinking tokens for models that expect them in conversation history."""
    if not thinking:
        return content
    return f"<think>{thinking}</think>\n{content}"


class LLMProvider(ABC):
    name: str
    # Per-provider RetryPolicy override. When ``None``, the module-level
    # default is used. Set on an instance or subclass to customise behaviour
    # without touching the concrete provider class.
    retry_policy: Any = None

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        pass

    async def complete(
        self,
        messages: list[LLMMessage],
        *args: Any,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Non-tool convenience alias for ``chat``.

        Goal / memory-flush / dream helpers historically called ``complete``.
        Providers only implement ``chat``; this default forwards and accepts
        (then ignores) ``stream=`` so those call sites don't AttributeError.
        Non-streaming is the default when ``on_chunk`` is omitted.
        """
        return await self.chat(messages)
