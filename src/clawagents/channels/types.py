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
class ChannelAttachment:
    """Normalized media/file attachment for channel messages."""

    url: str
    mime_type: str
    filename: str | None = None
    kind: str = "file"
    alt_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelCommand:
    """Parsed slash command from a channel message body."""

    name: str
    args: str = ""
    argv: list[str] = field(default_factory=list)
    raw: str = ""


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
    media: list[ChannelAttachment] = field(default_factory=list)
    command: ChannelCommand | None = None
    reply_to_id: str | None = None
    raw: Any = None

    def __post_init__(self) -> None:
        self.media = normalize_channel_attachments(self.media)
        if self.command is None:
            self.command = parse_channel_command(self.body)


def parse_channel_command(body: str, *, prefix: str = "/") -> ChannelCommand | None:
    """Parse a leading slash command without treating URLs as commands."""

    stripped = body.strip()
    if not stripped.startswith(prefix) or stripped.startswith(prefix * 2):
        return None
    first, _, rest = stripped[len(prefix):].partition(" ")
    name = first.strip().lower()
    if not name or any(ch.isspace() for ch in name):
        return None
    args = rest.strip()
    argv = args.split() if args else []
    return ChannelCommand(name=name, args=args, argv=argv, raw=stripped)


def normalize_channel_attachments(media: list[Any] | None) -> list[ChannelAttachment]:
    out: list[ChannelAttachment] = []
    for item in media or []:
        if isinstance(item, ChannelAttachment):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        mime_type = str(item.get("mime_type") or item.get("mimeType") or "")
        if not url or not mime_type:
            continue
        out.append(ChannelAttachment(
            url=url,
            mime_type=mime_type,
            filename=item.get("filename"),
            kind=str(item.get("kind") or "file"),
            alt_text=item.get("alt_text") or item.get("altText"),
            metadata=dict(item.get("metadata") or {}),
        ))
    return out


def channel_message_to_agent_input(msg: ChannelMessage) -> str:
    """Render a normalized channel message into the prompt sent to the agent."""

    parts: list[str] = []
    if msg.command:
        parts.append(f"[Channel Command: {msg.command.name}]")
        if msg.command.args:
            parts.append(f"Args: {msg.command.args}")
        parts.append("")
    parts.append(msg.body)
    if msg.media:
        parts.append("\n[Attachments]")
        for attachment in msg.media:
            name = attachment.filename or attachment.url
            parts.append(f"- {name} ({attachment.mime_type}): {attachment.url}")
    return "\n".join(parts).strip()


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
        media: list[ChannelAttachment | dict[str, str]] | None = None,
    ) -> None: ...
