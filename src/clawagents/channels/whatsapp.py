"""
WhatsApp channel adapter using whatsapp-web.js (via Node subprocess)
or neonize (pure-Python Baileys port).

For maximum compatibility, this adapter shells out to a small Node.js
helper script that uses Baileys. The helper communicates via stdin/stdout
JSON lines. This mirrors how OpenClaw's signal adapter works with signal-cli.

Alternatively, if you have a WhatsApp Business API endpoint, set
config = {"api_url": "https://...", "api_token": "...", "phone_id": "..."}
to use the official Cloud API instead.

Requires (Baileys mode): Node.js + npm install baileys
Requires (Business API mode): httpx (already a dep of openai)

Config (Baileys):
    {"mode": "baileys", "auth_dir": ".whatsapp-auth", "node_bin": "node"}

Config (Business API):
    {"mode": "business_api", "api_url": "...", "api_token": "...", "phone_id": "..."}
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from clawagents.channels.types import ChannelAdapter, ChannelMessage

_BAILEYS_HELPER = '''
const { default: makeWASocket, useMultiFileAuthState, DisconnectReason } = require("baileys");
const authDir = process.argv[2] || ".whatsapp-auth";

async function main() {
    const { state, saveCreds } = await useMultiFileAuthState(authDir);
    const sock = makeWASocket({ auth: state, printQRInTerminal: true });
    sock.ev.on("creds.update", saveCreds);
    sock.ev.on("connection.update", u => {
        if (u.connection === "open") process.stderr.write("[WhatsApp] Connected\\n");
        if (u.connection === "close") {
            const code = u.lastDisconnect?.error?.output?.statusCode;
            if (code !== DisconnectReason.loggedOut) {
                process.stderr.write("[WhatsApp] Reconnecting...\\n");
                main();
            }
        }
    });
    sock.ev.on("messages.upsert", ({ messages }) => {
        for (const m of messages) {
            if (m.key.fromMe) continue;
            const text = m.message?.conversation ?? m.message?.extendedTextMessage?.text ?? "";
            if (!text) continue;
            const out = JSON.stringify({
                type: "message",
                jid: m.key.remoteJid,
                sender: m.key.participant || m.key.remoteJid,
                pushName: m.pushName || "",
                text,
                ts: m.messageTimestamp,
            });
            process.stdout.write(out + "\\n");
        }
    });

    process.stdin.on("data", async buf => {
        try {
            const cmd = JSON.parse(buf.toString().trim());
            if (cmd.action === "send") {
                await sock.sendMessage(cmd.jid, { text: cmd.text });
                process.stdout.write(JSON.stringify({ type: "sent", jid: cmd.jid }) + "\\n");
            }
        } catch (e) {
            process.stderr.write("Error: " + e.message + "\\n");
        }
    });
}
main().catch(e => { process.stderr.write(e.stack + "\\n"); process.exit(1); });
'''


class WhatsAppAdapter:
    id = "whatsapp"
    name = "WhatsApp"
    on_message: Callable[[ChannelMessage], None] = lambda _: None

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._reader_task: asyncio.Task | None = None
        self._mode: str = "baileys"
        self._api_url: str = ""
        self._api_token: str = ""
        self._phone_id: str = ""

    async def start(self, config: dict[str, Any]) -> None:
        self._mode = str(config.get("mode", "baileys"))

        if self._mode == "business_api":
            self._api_url = str(config["api_url"])
            self._api_token = str(config["api_token"])
            self._phone_id = str(config["phone_id"])
            print("[WhatsApp] Business API mode — webhook receiver not included, "
                  "use your own webhook to call on_message().")
            return

        auth_dir = str(config.get("auth_dir", ".whatsapp-auth"))
        node_bin = str(config.get("node_bin", "node"))

        helper_path = Path(tempfile.gettempdir()) / "clawagents_wa_helper.js"
        helper_path.write_text(_BAILEYS_HELPER)

        self._proc = subprocess.Popen(
            [node_bin, str(helper_path), auth_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )

        self._reader_task = asyncio.create_task(self._read_loop())
        print(f"[WhatsApp] Baileys helper started (auth_dir={auth_dir})")

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, self._proc.stdout.readline)
            if not line:
                break
            try:
                data = json.loads(line.decode().strip())
            except json.JSONDecodeError:
                continue

            if data.get("type") == "message":
                self.on_message(ChannelMessage(
                    channel_id="whatsapp",
                    sender_id=data.get("sender", ""),
                    sender_name=data.get("pushName") or None,
                    conversation_id=data.get("jid", ""),
                    body=data.get("text", ""),
                    timestamp=float(data.get("ts", 0)) * 1000,
                    raw=data,
                ))

    async def send(
        self,
        conversation_id: str,
        content: str,
        media: list[dict[str, str]] | None = None,
    ) -> None:
        if self._mode == "business_api":
            await self._send_business_api(conversation_id, content)
            return

        if not self._proc or not self._proc.stdin:
            raise RuntimeError("WhatsAppAdapter: helper process not running")
        cmd = json.dumps({"action": "send", "jid": conversation_id, "text": content}) + "\n"
        self._proc.stdin.write(cmd.encode())
        self._proc.stdin.flush()

    async def _send_business_api(self, conversation_id: str, content: str) -> None:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self._api_url}/{self._phone_id}/messages",
                headers={"Authorization": f"Bearer {self._api_token}"},
                json={
                    "messaging_product": "whatsapp",
                    "to": conversation_id,
                    "type": "text",
                    "text": {"body": content},
                },
            )

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._proc:
            self._proc.terminate()
            self._proc = None
