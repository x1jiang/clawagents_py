"""Provider-agnostic transport interface.

Today, ``clawagents.providers.llm`` ships a single fat module that knows
about every concrete provider (OpenAI Responses, OpenAI Chat
Completions, Anthropic Messages, Google GenAI, Ollama, …). That works,
but it's hard to extend cleanly: every new provider has to be added
inside the same file with provider-specific branching.

This module introduces a thin :class:`Transport` abstraction so new
backends can be plugged in *without touching* ``llm.py``. The existing
provider code keeps working as-is — :class:`Transport` is purely
additive and lives alongside the legacy entrypoints.

Architecture
------------
- :class:`TransportRequest` — provider-agnostic chat request payload.
- :class:`TransportResponse` — provider-agnostic chat response payload.
- :class:`Transport` — abstract base class. Concrete subclasses (one per
  provider) implement ``chat`` and optionally ``stream`` / ``aclose``.
- :class:`TransportRegistry` — process-wide registry mapping a
  provider name to a :class:`Transport` instance.

Adapter for legacy providers
----------------------------
Use :class:`LegacyChatTransport` when you want to expose
``providers.llm.chat_with_provider`` (the existing function) as a
:class:`Transport` without rewriting it.

Mirrors ``clawagents/src/transport.ts``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class TransportRequest:
    """A provider-agnostic chat request.

    Keep this minimal: it should describe *what* the model is being
    asked to do, not the wire format used to ask. Provider-specific
    fields can ride along inside :attr:`extra`.

    Attributes:
        model: Model id (e.g. ``"gpt-5.4"``, ``"claude-4.5-sonnet"``).
        messages: Chat history as a list of dicts (``{"role", "content"}``).
            We use plain dicts here to avoid coupling the abstraction to
            ``providers.llm.LLMMessage``.
        tools: Optional list of native tool schemas, again as dicts.
        tool_choice: One of ``"auto"``, ``"required"``, ``"none"``,
            or a specific tool name.
        temperature: Optional sampling temperature.
        max_tokens: Optional response length cap.
        stream: When True, callers should use :meth:`Transport.stream`.
        extra: Free-form bag for provider-specific knobs that the
            transport implementation can interpret (e.g. Anthropic
            cache-control, Gemini thinking budgets).
    """

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportResponse:
    """A provider-agnostic chat response.

    Attributes:
        text: Final assistant text. ``None`` if the model only emitted
            tool calls.
        tool_calls: List of ``{"id", "name", "args"}`` dicts describing
            tool calls the model wants to perform. Empty list means
            none.
        usage: Optional usage payload (token counts, cost, latency,
            cache stats). The shape is provider-specific.
        finish_reason: One of ``"stop"``, ``"length"``, ``"tool_calls"``,
            ``"content_filter"``, etc. Mirrors OpenAI's spelling.
        raw: Optional raw provider response, kept for debugging.
            Avoid relying on this in production code.
    """

    text: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: Any | None = None


class Transport(ABC):
    """Abstract base class for a provider transport.

    A transport is a thin wrapper around one provider's chat API. It
    knows how to translate :class:`TransportRequest` into the
    provider's wire format and back into :class:`TransportResponse`.
    Authentication, base-url handling, retries, and rate-limiting are
    all internal concerns of the transport.
    """

    #: Stable identifier (``"openai"``, ``"anthropic"``, ``"gemini"``,
    #: ``"ollama"``, ``"openrouter"``, …). Used by
    #: :class:`TransportRegistry`.
    name: str = ""

    @abstractmethod
    async def chat(self, request: TransportRequest) -> TransportResponse:
        """Issue a single chat call and return the final response.

        The default for non-streaming providers; streaming providers
        should still implement this (consume their stream and return
        the accumulated result) so callers don't have to special-case.
        """

    async def stream(
        self,
        request: TransportRequest,
    ) -> AsyncIterator[TransportResponse]:
        """Yield incremental :class:`TransportResponse` chunks.

        The default implementation just calls :meth:`chat` once and
        yields the final result, so transports that don't support
        streaming still satisfy the interface.
        """
        yield await self.chat(request)

    async def aclose(self) -> None:
        """Release any underlying client resources. Called at shutdown."""


class TransportRegistry:
    """Process-wide transport map. Thread-safe by virtue of GIL."""

    _transports: dict[str, Transport] = {}

    @classmethod
    def register(cls, transport: Transport, *, name: str | None = None) -> None:
        """Register ``transport`` under ``name`` (defaults to ``transport.name``)."""
        key = name or transport.name
        if not key:
            raise ValueError("TransportRegistry.register: missing name")
        cls._transports[key] = transport

    @classmethod
    def get(cls, name: str) -> Transport:
        """Return the registered transport. Raises ``KeyError`` if missing."""
        try:
            return cls._transports[name]
        except KeyError:
            raise KeyError(
                f"No transport registered under {name!r}. "
                f"Known: {sorted(cls._transports)}"
            ) from None

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._transports

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._transports)

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._transports.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Drop all registered transports (test helper)."""
        cls._transports.clear()


class LegacyChatTransport(Transport):
    """Adapter that exposes a callable as a :class:`Transport`.

    Useful for wrapping the existing ``providers.llm.chat_with_provider``
    or any compatible coroutine without rewriting it. The callable must
    accept the :class:`TransportRequest` (or its keyword fields) and
    return a :class:`TransportResponse` directly.
    """

    def __init__(
        self,
        name: str,
        chat_fn: Any,
    ) -> None:
        self.name = name
        self._chat_fn = chat_fn

    async def chat(self, request: TransportRequest) -> TransportResponse:
        result = await self._chat_fn(request)
        if isinstance(result, TransportResponse):
            return result
        if isinstance(result, dict):
            return TransportResponse(**result)
        raise TypeError(
            f"LegacyChatTransport({self.name!r}): chat_fn returned "
            f"{type(result).__name__}, expected TransportResponse or dict"
        )


__all__ = [
    "TransportRequest",
    "TransportResponse",
    "Transport",
    "TransportRegistry",
    "LegacyChatTransport",
]
