"""MCP sessions must survive event-loop changes via lazy reconnect.

Regression for the ClosedResourceError seen when ``create_claw_agent``
registers MCP tools in a temporary event loop (asyncio.run in a helper
thread) and the agent later invokes them from its own run loop. Uses a real
stdio server (FastMCP echo) — task/loop affinity cannot be faked.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest

pytest.importorskip("mcp", reason="optional MCP SDK not installed")

from clawagents import MCPServerStdio  # noqa: E402

SERVER_SCRIPT = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("echo-test")

    @mcp.tool()
    def echo(text: str) -> str:
        \"\"\"Echo text back.\"\"\"
        return "echo:" + text

    mcp.run()
    """
)


@pytest.fixture()
def echo_server(tmp_path):
    script = tmp_path / "echo_server.py"
    script.write_text(SERVER_SCRIPT, encoding="utf-8")
    return MCPServerStdio(
        {"command": sys.executable, "args": [str(script)]},
        name="echo-test",
        client_session_timeout_seconds=30.0,
    )


def _text_of(result) -> str:
    out = ""
    for block in getattr(result, "content", None) or []:
        out += getattr(block, "text", "")
    return out


def test_invoke_from_fresh_loop_reconnects(echo_server):
    """connect() in loop 1, invoke_tool() in loop 2 → auto-reconnect, no ClosedResourceError."""

    async def register():
        await echo_server.connect()
        tools = await echo_server.list_tools()
        return [t.name for t in tools]

    names = asyncio.run(register())  # loop 1 — closed after this line
    assert "echo" in names

    async def use():
        result = await echo_server.invoke_tool("echo", {"text": "hi"})
        return _text_of(result)

    out = asyncio.run(use())  # loop 2 — must not see loop-1 streams
    assert out == "echo:hi"

    async def cleanup():
        await echo_server.shutdown()

    asyncio.run(cleanup())


def test_same_loop_invocations_reuse_session(echo_server):
    """Within one loop the session is reused (no reconnect churn)."""

    async def flow():
        await echo_server.connect()
        first_session = echo_server._session
        r1 = await echo_server.invoke_tool("echo", {"text": "a"})
        r2 = await echo_server.invoke_tool("echo", {"text": "b"})
        assert echo_server._session is first_session
        await echo_server.shutdown()
        return _text_of(r1), _text_of(r2)

    a, b = asyncio.run(flow())
    assert (a, b) == ("echo:a", "echo:b")
