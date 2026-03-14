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
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from clawagents.channels.router import ChannelRouter


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
    router = ChannelRouter(agent_factory, debounce_ms=debounce_ms)

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
