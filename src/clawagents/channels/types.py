"""
Core channel abstraction for multi-platform messaging.

Each messaging platform (WhatsApp, Telegram, Signal, Slack, Discord, …)
implements the ChannelAdapter protocol. The ChannelRouter dispatches
inbound messages to agents and routes outbound replies through the
originating adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class ChannelMessage:
    """Normalized inbound message from any platform."""

    channel_id: str
    """Platform identifier, e.g. "telegram", "whatsapp", "signal"."""

    sender_id: str
    """Platform-specific sender identifier."""

    conversation_id: str
    """Group or chat identifier — combined with channel_id forms the session key."""

    body: str
    """Text body of the message."""

    timestamp: float
    """Epoch seconds when the message was sent."""

    sender_name: str | None = None
    media: list[dict[str, str]] = field(default_factory=list)
    reply_to_id: str | None = None
    raw: Any = None


@runtime_checkable
class ChannelAdapter(Protocol):
    """Protocol that every messaging platform adapter must implement."""

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    on_message: Callable[[ChannelMessage], None]

    async def start(self, config: dict[str, Any]) -> None: ...
    async def stop(self) -> None: ...
    async def send(
        self,
        conversation_id: str,
        content: str,
        media: list[dict[str, str]] | None = None,
    ) -> None: ...
