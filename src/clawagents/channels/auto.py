"""
Auto-detect messaging channels from environment variables.

When `clawagents --serve` is run, this module checks for channel env vars
and automatically configures + starts the ChannelRouter alongside the gateway.

Environment variables:
    TELEGRAM_BOT_TOKEN     → starts Telegram adapter
    WHATSAPP_AUTH_DIR      → starts WhatsApp adapter (Baileys mode)
    WHATSAPP_API_URL       → starts WhatsApp adapter (Business API mode)
    SIGNAL_ACCOUNT         → starts Signal adapter
    CHANNEL_DEBOUNCE_MS    → debounce window for rapid messages (default: 500)

Sender allowlists (required — channels deny every sender by default):
    TELEGRAM_ALLOWED_SENDERS   → comma-separated user or chat ids
    WHATSAPP_ALLOWED_SENDERS   → comma-separated sender JIDs / phone ids
    SIGNAL_ALLOWED_SENDERS     → comma-separated phone numbers or group ids
    Set a list to ``*`` to accept anyone (only for a bot you intend to be public).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from clawagents.channels.router import ChannelRouter
from clawagents.channels.types import ChannelMessage

_ALLOWLIST_ENV = {
    "telegram": "TELEGRAM_ALLOWED_SENDERS",
    "whatsapp": "WHATSAPP_ALLOWED_SENDERS",
    "signal": "SIGNAL_ALLOWED_SENDERS",
}


def allowlist_env_name(channel_id: str) -> str:
    return _ALLOWLIST_ENV.get(channel_id, f"{channel_id.upper()}_ALLOWED_SENDERS")


def load_sender_allowlists(channel_ids: Any) -> dict[str, frozenset[str]]:
    """Read the per-channel sender allowlists from the environment.

    An unset or empty variable yields an empty set, i.e. deny everyone.
    """
    out: dict[str, frozenset[str]] = {}
    for ch in channel_ids:
        raw = os.getenv(allowlist_env_name(ch), "")
        out[ch] = frozenset(p.strip() for p in raw.split(",") if p.strip())
    return out


def make_sender_gate(
    allowlists: dict[str, frozenset[str]],
    *,
    log: Callable[[str], None] | None = None,
) -> Callable[[ChannelMessage], bool]:
    """Build the router's ``on_inbound`` gate: deny unless allowlisted.

    A channel message spawns an agent with shell and file tools on this
    host, so an unknown sender must never reach it. A message is accepted
    when its sender id *or* conversation id is listed (the latter lets an
    operator allow a whole group), or the list is ``*``. Each refused sender
    is logged once with its id so the operator can copy it into the env var.
    """
    emit = log or (lambda line: print(line, file=sys.stderr))
    reported: set[tuple[str, str]] = set()

    def gate(msg: ChannelMessage) -> bool:
        allowed = allowlists.get(msg.channel_id, frozenset())
        if "*" in allowed or msg.sender_id in allowed or msg.conversation_id in allowed:
            return True
        key = (msg.channel_id, msg.sender_id)
        if key not in reported and len(reported) < 1000:
            reported.add(key)
            emit(
                f"[Router] refused {msg.channel_id} message from sender "
                f"{msg.sender_id!r} (conversation {msg.conversation_id!r}): not in "
                f"{allowlist_env_name(msg.channel_id)}"
            )
        return False

    return gate


def detect_channels() -> dict[str, dict[str, Any]]:
    """Return a dict of channel_id -> config for all detected channels."""
    detected: dict[str, dict[str, Any]] = {}

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if token:
        detected["telegram"] = {"bot_token": token}

    wa_auth = os.getenv("WHATSAPP_AUTH_DIR", "")
    wa_api = os.getenv("WHATSAPP_API_URL", "")
    if wa_auth:
        detected["whatsapp"] = {
            "mode": "baileys",
            "auth_dir": wa_auth,
            "node_bin": os.getenv("WHATSAPP_NODE_BIN", "node"),
        }
    elif wa_api:
        detected["whatsapp"] = {
            "mode": "business_api",
            "api_url": wa_api,
            "api_token": os.getenv("WHATSAPP_API_TOKEN", ""),
            "phone_id": os.getenv("WHATSAPP_PHONE_ID", ""),
        }

    signal_acct = os.getenv("SIGNAL_ACCOUNT", "")
    if signal_acct:
        detected["signal"] = {
            "account": signal_acct,
            "signal_cli_bin": os.getenv("SIGNAL_CLI_BIN", "signal-cli"),
        }

    return detected


def describe_channels(channels: dict[str, dict[str, Any]]) -> list[str]:
    """Return human-readable descriptions for --doctor / banner."""
    lines = []
    for ch_id, cfg in channels.items():
        if ch_id == "telegram":
            tok = cfg["bot_token"]
            lines.append(f"telegram ({tok[:6]}...)")
        elif ch_id == "whatsapp":
            mode = cfg.get("mode", "baileys")
            lines.append(f"whatsapp ({mode})")
        elif ch_id == "signal":
            lines.append(f"signal ({cfg['account']})")
        else:
            lines.append(ch_id)
    return lines


async def start_channel_router(llm: Any) -> ChannelRouter | None:
    """
    Detect channels from env, create adapters, start the router.
    Returns None if no channels are configured.
    """
    channels = detect_channels()
    if not channels:
        return None

    from clawagents.agent import create_claw_agent

    async def agent_factory():
        return create_claw_agent(model=llm)

    debounce_ms = int(os.getenv("CHANNEL_DEBOUNCE_MS", "500"))
    allowlists = load_sender_allowlists(channels)
    for ch, allowed in allowlists.items():
        env_name = allowlist_env_name(ch)
        if not allowed:
            print(
                f"[Router] {env_name} is not set: every {ch} message will be refused. "
                f"Set it to a comma-separated list of sender ids (refused senders are "
                f"logged with their id), or to * to accept anyone.",
                file=sys.stderr,
            )
        elif "*" in allowed:
            print(
                f"[Router] WARNING: {env_name}=* — anyone who can message the {ch} "
                f"account can run an agent with shell and file tools on this host.",
                file=sys.stderr,
            )
    router = ChannelRouter(
        agent_factory,
        on_inbound=make_sender_gate(allowlists),
        debounce_ms=debounce_ms,
    )

    if "telegram" in channels:
        from clawagents.channels.telegram import TelegramAdapter
        router.register(TelegramAdapter())

    if "whatsapp" in channels:
        from clawagents.channels.whatsapp import WhatsAppAdapter
        router.register(WhatsAppAdapter())

    if "signal" in channels:
        from clawagents.channels.signal import SignalAdapter
        router.register(SignalAdapter())

    await router.start_all(channels)
    return router
