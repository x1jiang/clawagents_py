"""
Signal channel adapter using signal-cli (subprocess).

Requires: signal-cli installed and registered
  - Install: https://github.com/AsamK/signal-cli
  - Register: signal-cli -a +1234567890 register
  - Verify:   signal-cli -a +1234567890 verify CODE

Config: {"account": "+1234567890", "signal_cli_bin": "signal-cli"}

The adapter runs `signal-cli -a <account> daemon --json` as a subprocess,
reading JSON lines from stdout (same pattern OpenClaw uses).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from typing import Any, Callable

from clawagents.channels.types import ChannelAdapter, ChannelMessage


class SignalAdapter:
    id = "signal"
    name = "Signal"
    on_message: Callable[[ChannelMessage], None] = lambda _: None

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._reader_task: asyncio.Task | None = None
        self._account: str = ""

    async def start(self, config: dict[str, Any]) -> None:
        self._account = str(config.get("account", ""))
        if not self._account:
            raise ValueError("SignalAdapter: missing 'account' (phone number)")

        bin_path = str(config.get("signal_cli_bin", "signal-cli"))

        self._proc = subprocess.Popen(
            [bin_path, "-a", self._account, "daemon", "--json"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )

        self._reader_task = asyncio.create_task(self._read_loop())
        print(f"[Signal] Daemon started for {self._account}")

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, self._proc.stdout.readline)
            if not line:
                break
            try:
                envelope = json.loads(line.decode().strip())
            except json.JSONDecodeError:
                continue
            self._handle_envelope(envelope)

    def _handle_envelope(self, envelope: dict) -> None:
        data = envelope.get("envelope")
        if not data:
            return

        data_message = data.get("dataMessage")
        if not data_message or not data_message.get("message"):
            return

        sender = data.get("source", "")
        sender_name = data.get("sourceName") or None
        group_id = (data_message.get("groupInfo") or {}).get("groupId", "")
        conversation_id = group_id or sender

        self.on_message(ChannelMessage(
            channel_id="signal",
            sender_id=sender,
            sender_name=sender_name,
            conversation_id=conversation_id,
            body=data_message["message"],
            timestamp=float(data.get("timestamp", time.time() * 1000)),
            raw=envelope,
        ))

    async def send(
        self,
        conversation_id: str,
        content: str,
        media: list[dict[str, str]] | None = None,
    ) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("SignalAdapter: daemon not running")

        is_group = not conversation_id.startswith("+")
        params: dict[str, Any]
        if is_group:
            params = {"groupId": conversation_id, "message": content}
        else:
            params = {"recipient": [conversation_id], "message": content}

        cmd = json.dumps({
            "jsonrpc": "2.0",
            "method": "send",
            "id": str(int(time.time() * 1000)),
            "params": params,
        }) + "\n"

        self._proc.stdin.write(cmd.encode())
        self._proc.stdin.flush()

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._proc:
            self._proc.terminate()
            self._proc = None
