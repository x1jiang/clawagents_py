"""Tool discovery must not leave a transport pinned to a throwaway loop.

``create_claw_agent`` discovers MCP tools inside a temporary event loop so it
works from sync, Streamlit and Jupyter contexts. A transport left open keeps
that loop's reader tasks alive, and ``asyncio.run`` then blocks forever in
``_cancel_all_tasks`` — the agent is never constructed and a chat turn hangs
with no output. ``release_transports`` closes them; ``_ensure_session``
reconnects lazily on whichever loop first invokes a tool.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest

from clawagents.mcp.manager import MCPServerManager
from clawagents.mcp.server import MCPToolDescriptor
from clawagents.tools.registry import ToolRegistry


class _LingeringServer:
    """Stand-in whose reader task only unwinds when the transport closes.

    Plain cancellation is ignored, which is what makes a real anyio-backed
    MCP transport un-collectable from a loop that is shutting down.
    """

    name = "lingering"

    def __init__(self) -> None:
        self.connect_count = 0
        self.shutdown_count = 0
        self._stop: asyncio.Event | None = None
        self._reader: asyncio.Task | None = None

    async def connect(self) -> None:
        self._stop = asyncio.Event()
        self._reader = asyncio.get_running_loop().create_task(self._read_loop())
        self.connect_count += 1

    async def _read_loop(self) -> None:
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=0.05)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

    async def list_tools(self):
        return [
            MCPToolDescriptor(
                name="ping",
                description="Ping the lingering server.",
                input_schema={"type": "object", "properties": {}},
                server_name=self.name,
            )
        ]

    async def shutdown(self) -> None:
        self.shutdown_count += 1
        if self._stop is not None:
            self._stop.set()
        if self._reader is not None:
            await self._reader
            self._reader = None


def test_release_transports_keeps_tools_registered():
    server = _LingeringServer()
    manager = MCPServerManager([server])
    registry = ToolRegistry()

    async def flow():
        await manager.start(registry)
        await manager.release_transports()

    asyncio.run(flow())

    assert server.shutdown_count == 1
    # The tool survives the transport so the agent can still advertise it.
    assert registry.get("ping") is not None
    assert manager.started is True


def test_discovery_loop_closes_when_called_from_a_running_loop():
    """The exact shape of create_claw_agent's MCP block must not deadlock."""
    server = _LingeringServer()
    manager = MCPServerManager([server])
    registry = ToolRegistry()

    async def discover_and_release():
        await manager.start(registry)
        await manager.release_transports()

    async def outer():
        # A running loop here forces the thread + asyncio.run branch.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, discover_and_release())
            try:
                future.result(timeout=20)
            except concurrent.futures.TimeoutError:
                pytest.fail(
                    "MCP discovery loop never shut down — transport left pinned "
                    "to the throwaway loop (agent construction would hang)"
                )

    asyncio.run(outer())
    assert server.shutdown_count == 1
