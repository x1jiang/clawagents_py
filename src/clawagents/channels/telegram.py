"""
Telegram channel adapter using python-telegram-bot.

Requires: pip install python-telegram-bot
Config: {"bot_token": "123456:ABC-DEF..."}
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from clawagents.channels.types import ChannelAdapter, ChannelMessage


class TelegramAdapter:
    id = "telegram"
    name = "Telegram"
    on_message: Callable[[ChannelMessage], None] = lambda _: None

    def __init__(self) -> None:
        self._app: Any = None
        self._task: asyncio.Task | None = None

    async def start(self, config: dict[str, Any]) -> None:
        token = str(config.get("bot_token", ""))
        if not token:
            raise ValueError("TelegramAdapter: missing bot_token in config")

        try:
            from telegram import Update
            from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
        except ImportError:
            raise ImportError(
                "TelegramAdapter: 'python-telegram-bot' not installed. "
                "Run: pip install python-telegram-bot"
            )

        self._app = ApplicationBuilder().token(token).build()

        async def _on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            msg = update.message
            if msg is None or msg.text is None:
                return
            chat_id = str(msg.chat_id)
            sender_id = str(msg.from_user.id) if msg.from_user else chat_id
            sender_name = None
            if msg.from_user:
                parts = [msg.from_user.first_name, msg.from_user.last_name]
                sender_name = " ".join(p for p in parts if p) or None

            self.on_message(ChannelMessage(
                channel_id="telegram",
                sender_id=sender_id,
                sender_name=sender_name,
                conversation_id=chat_id,
                body=msg.text,
                timestamp=msg.date.timestamp(),
                raw=msg,
            ))

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        print(f"[Telegram] Bot started ({token[:6]}...)")

    async def send(
        self,
        conversation_id: str,
        content: str,
        media: list[dict[str, str]] | None = None,
    ) -> None:
        if not self._app:
            raise RuntimeError("TelegramAdapter: bot not started")
        await self._app.bot.send_message(chat_id=int(conversation_id), text=content)

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None
