"""
Multi-channel router that dispatches inbound messages to agents
and routes outbound replies through the originating adapter.

Features:
  - Per-session serialization via KeyedAsyncQueue (prevents race conditions)
  - Configurable agent factory (fresh agent per message, or shared)
  - Optional inbound debouncer (batches rapid messages)
  - Hooks: on_inbound, on_outbound, on_error for observability
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from clawagents.channels.types import ChannelAdapter, ChannelMessage
from clawagents.channels.keyed_queue import KeyedAsyncQueue

AgentFactory = Callable[[], Awaitable[Any]]


class ChannelRouter:
    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        on_inbound: Optional[Callable[[ChannelMessage], bool | Awaitable[bool]]] = None,
        on_outbound: Optional[Callable[[ChannelMessage, str], str | Awaitable[str]]] = None,
        on_error: Optional[Callable[[ChannelMessage, Exception], None]] = None,
        debounce_ms: int = 0,
    ) -> None:
        self._agent_factory = agent_factory
        self._adapters: dict[str, Any] = {}
        self._session_queue = KeyedAsyncQueue()
        self._on_inbound = on_inbound
        self._on_outbound = on_outbound
        self._on_error = on_error
        self._debounce_ms = debounce_ms
        self._debounce_tasks: dict[str, asyncio.TimerHandle] = {}
        self._debounce_batches: dict[str, list[ChannelMessage]] = {}

    def register(self, adapter: Any) -> "ChannelRouter":
        """Register a channel adapter. Sets the on_message callback."""
        adapter.on_message = lambda msg: self._handle_inbound(msg)
        self._adapters[adapter.id] = adapter
        return self

    async def start_all(self, configs: dict[str, dict[str, Any]]) -> None:
        """Start all registered adapters with their configs."""
        tasks = []
        for channel_id, config in configs.items():
            adapter = self._adapters.get(channel_id)
            if adapter:
                tasks.append(adapter.start(config))
            else:
                print(f'[Router] No adapter registered for channel "{channel_id}"')
        await asyncio.gather(*tasks)
        print(f"[Router] {len(self._adapters)} channel(s) started")

    async def stop_all(self) -> None:
        """Stop all adapters gracefully."""
        for handle in self._debounce_tasks.values():
            handle.cancel()
        self._debounce_tasks.clear()
        self._debounce_batches.clear()
        await asyncio.gather(*(a.stop() for a in self._adapters.values()))
        print("[Router] All channels stopped")

    @property
    def registered_channels(self) -> list[str]:
        return list(self._adapters.keys())

    @property
    def active_sessions(self) -> int:
        return self._session_queue.active_keys

    def _handle_inbound(self, msg: ChannelMessage) -> None:
        if self._debounce_ms <= 0:
            asyncio.ensure_future(self._dispatch(msg))
            return

        key = f"{msg.channel_id}:{msg.conversation_id}"
        batch = self._debounce_batches.setdefault(key, [])
        batch.append(msg)

        existing = self._debounce_tasks.get(key)
        if existing:
            existing.cancel()

        loop = asyncio.get_event_loop()
        handle = loop.call_later(
            self._debounce_ms / 1000.0,
            lambda k=key: asyncio.ensure_future(self._flush_debounce(k)),
        )
        self._debounce_tasks[key] = handle

    async def _flush_debounce(self, key: str) -> None:
        self._debounce_tasks.pop(key, None)
        messages = self._debounce_batches.pop(key, [])
        if not messages:
            return
        combined = ChannelMessage(
            channel_id=messages[-1].channel_id,
            sender_id=messages[-1].sender_id,
            sender_name=messages[-1].sender_name,
            conversation_id=messages[-1].conversation_id,
            body="\n".join(m.body for m in messages),
            timestamp=messages[-1].timestamp,
            raw=messages[-1].raw,
        )
        await self._dispatch(combined)

    async def _dispatch(self, msg: ChannelMessage) -> None:
        session_key = f"{msg.channel_id}:{msg.conversation_id}"

        async def _process() -> None:
            try:
                if self._on_inbound:
                    allow = self._on_inbound(msg)
                    if asyncio.iscoroutine(allow):
                        allow = await allow
                    if not allow:
                        return

                agent = await self._agent_factory()
                result = await agent.invoke(msg.body)
                reply = result.result or ""

                if self._on_outbound:
                    r = self._on_outbound(msg, reply)
                    if asyncio.iscoroutine(r):
                        r = await r
                    reply = r

                if not reply:
                    return

                adapter = self._adapters.get(msg.channel_id)
                if adapter:
                    await adapter.send(msg.conversation_id, reply)
            except Exception as e:
                if self._on_error:
                    self._on_error(msg, e)
                else:
                    print(f"[Router] Error processing {msg.channel_id}:{msg.conversation_id}: {e}")

        await self._session_queue.enqueue(session_key, _process)
