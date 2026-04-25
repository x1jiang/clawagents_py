"""Tests for the MCP (Model Context Protocol) client integration.

The whole subpackage must import cleanly even when the optional ``mcp`` SDK
is not installed. Live-server tests only run when ``mcp`` is available.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest


# ``asyncio_mode = "auto"`` in ``pyproject.toml`` already runs ``async def``
# tests under asyncio; we only mark the live-server tests explicitly so that
# sync helpers (schema normalisation, surface re-export checks) don't get the
# spurious ``PytestWarning`` about asyncio markers.
MCP_SDK_AVAILABLE = importlib.util.find_spec("mcp") is not None


# ─── Always-on tests (run with or without the SDK) ───────────────────────


def test_public_surface_re_exported_from_top_level():
    """The v6.4 public API must expose every MCP class at the top of the package."""
    import clawagents

    for name in (
        "MCPServer",
        "MCPServerStdio",
        "MCPServerSse",
        "MCPServerStreamableHttp",
        "MCPServerManager",
        "MCPLifecyclePhase",
        "MCPToolDescriptor",
        "MCPBridgedTool",
        "is_mcp_sdk_available",
        "require_mcp_sdk",
        "mcp_tool_to_clawagents_tool",
    ):
        assert hasattr(clawagents, name), f"clawagents.{name} not exported"


def test_mcp_subpackage_imports_without_optional_sdk():
    """``import clawagents.mcp`` works whether or not ``mcp`` is installed."""
    import clawagents.mcp as mcp_module

    assert mcp_module.MCPServerStdio is not None
    assert mcp_module.MCPServerManager is not None


def test_lifecycle_phase_enum_values():
    """Phase enum must include every documented state from the brief."""
    from clawagents.mcp import MCPLifecyclePhase

    expected = {
        "idle", "connecting", "initializing", "discovering_tools",
        "ready", "invoking", "errored", "shutdown",
    }
    actual = {p.value for p in MCPLifecyclePhase}
    assert expected == actual


def test_tool_bridge_normalizes_jsonschema_to_clawagents_params():
    """An MCP tool's ``inputSchema`` (JSON Schema) is flattened to the clawagents shape."""
    from clawagents.mcp.tool_bridge import _normalize_input_schema

    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Where to read"},
            "limit": {"type": "integer", "description": "Max bytes"},
            "verbose": {"type": "boolean"},
            "weird": {"type": ["string", "null"]},
        },
        "required": ["path"],
    }
    out = _normalize_input_schema(schema)
    assert out["path"] == {"type": "string", "description": "Where to read", "required": True}
    assert out["limit"] == {"type": "integer", "description": "Max bytes", "required": False}
    assert out["verbose"]["type"] == "boolean"
    # ``["string", "null"]`` collapses to ``"string"``
    assert out["weird"]["type"] == "string"


def test_tool_bridge_handles_empty_or_missing_schema():
    from clawagents.mcp.tool_bridge import _normalize_input_schema

    assert _normalize_input_schema({}) == {}
    # JSON-Schema "type" might be omitted; properties might be missing.
    assert _normalize_input_schema({"type": "object"}) == {}
    assert _normalize_input_schema({"type": "object", "properties": "not-a-dict"}) == {}  # type: ignore[arg-type]


def test_call_result_stringifier_concatenates_text_blocks():
    from clawagents.mcp.tool_bridge import _stringify_call_result

    class _Block:
        def __init__(self, text: str | None, type_: str = "text") -> None:
            self.text = text
            self.type = type_

    class _Result:
        def __init__(self, content: list[_Block], is_error: bool = False) -> None:
            self.content = content
            self.isError = is_error

    success, output, error = _stringify_call_result(
        _Result([_Block("hello"), _Block("world")])
    )
    assert success is True
    assert output == "hello\nworld"
    assert error is None

    success, output, error = _stringify_call_result(
        _Result([_Block("boom")], is_error=True)
    )
    assert success is False
    assert "boom" in output
    assert error is not None


def test_call_result_stringifier_summarises_non_text_blocks():
    from clawagents.mcp.tool_bridge import _stringify_call_result

    class _Block:
        def __init__(self) -> None:
            self.type = "image"
            self.text = None  # not a string

    class _Result:
        content = [_Block()]
        isError = False

    success, output, error = _stringify_call_result(_Result())
    assert success is True
    assert "[image block]" in output


async def test_create_claw_agent_raises_when_mcp_servers_passed_without_sdk(monkeypatch):
    """If the ``mcp`` SDK is missing, ``create_claw_agent(mcp_servers=[...])`` must
    raise a clear ``ImportError`` at construction time, not silently."""
    import clawagents.agent as agent_module

    def fake_is_available() -> bool:
        return False

    # We monkey-patch the helper that ``create_claw_agent`` consults so we can
    # exercise the import-error path even on machines where ``mcp`` is installed.
    import clawagents.mcp as mcp_pkg
    monkeypatch.setattr(mcp_pkg, "is_mcp_sdk_available", fake_is_available)
    # The agent factory imports ``is_mcp_sdk_available`` from clawagents.mcp, so
    # patching the package re-export is sufficient.

    # We need *some* dummy server to trigger the path. A bare object suffices —
    # the import-error check runs before ``MCPServerManager`` is constructed.
    sentinel = object()
    with pytest.raises(ImportError, match="clawagents\\[mcp\\]"):
        # Use a stub LLM provider via the ``model`` kwarg path with a real
        # provider would require API keys. Instead we go through `create_claw_agent`
        # but supply an LLMProvider directly.
        from clawagents.providers.llm import LLMProvider, LLMMessage, LLMResponse

        class _StubLLM(LLMProvider):  # type: ignore[misc]
            def __init__(self) -> None:
                pass

            async def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                return LLMResponse(content="", finish_reason="stop")

            async def chat_stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                yield ""

        agent_module.create_claw_agent(model=_StubLLM(), mcp_servers=[sentinel])


# ─── Live-server tests (only when ``mcp`` SDK is installed) ───────────────


_FIXTURE_SOURCE = textwrap.dedent(
    '''
    """Tiny MCP stdio server used as a test fixture.

    Lists one tool ``echo`` that returns its ``text`` argument prefixed with ``echo: ``.
    """

    import asyncio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent


    server = Server("clawagents-test-fixture")


    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="echo",
                description="Echo the provided text.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to echo"},
                    },
                    "required": ["text"],
                },
            )
        ]


    @server.call_tool()
    async def _call_tool(name, arguments):
        if name == "echo":
            text = (arguments or {}).get("text", "")
            return [TextContent(type="text", text=f"echo: {text}")]
        raise ValueError(f"Unknown tool: {name}")


    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


    if __name__ == "__main__":
        asyncio.run(main())
    '''
).strip()


@pytest.fixture
def stdio_server_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "echo_mcp_server.py"
    p.write_text(_FIXTURE_SOURCE + "\n", encoding="utf-8")
    return p


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="mcp SDK not installed")
async def test_stdio_server_lists_and_invokes_echo_tool(stdio_server_fixture: Path) -> None:
    from clawagents.mcp import MCPServerStdio

    server = MCPServerStdio(
        params={
            "command": sys.executable,
            "args": [str(stdio_server_fixture)],
        },
        name="echo-fixture",
    )
    async with server:
        tools = await server.list_tools()
        assert [t.name for t in tools] == ["echo"]
        result = await server.invoke_tool("echo", {"text": "hi"})
        # Result has .content with at least one text block.
        text_blocks = [getattr(b, "text", "") for b in (result.content or [])]
        assert any("echo: hi" in t for t in text_blocks)


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="mcp SDK not installed")
async def test_manager_bridges_mcp_tool_into_registry(stdio_server_fixture: Path) -> None:
    from clawagents.mcp import MCPServerManager, MCPServerStdio
    from clawagents.tools.registry import ToolRegistry

    server = MCPServerStdio(
        params={"command": sys.executable, "args": [str(stdio_server_fixture)]},
        name="echo-fixture",
    )
    registry = ToolRegistry()
    manager = MCPServerManager([server])
    try:
        registered = await manager.start(registry)
        assert "echo" in registered
        bridged = registry.get("echo")
        assert bridged is not None
        assert "text" in bridged.parameters
        result = await bridged.execute({"text": "hello"})
        assert result.success is True
        assert "echo: hello" in result.output
    finally:
        await manager.shutdown()


@pytest.mark.skipif(not MCP_SDK_AVAILABLE, reason="mcp SDK not installed")
async def test_lifecycle_phase_progresses_through_states(stdio_server_fixture: Path) -> None:
    from clawagents.mcp import MCPLifecyclePhase, MCPServerStdio

    server = MCPServerStdio(
        params={"command": sys.executable, "args": [str(stdio_server_fixture)]},
        name="echo-fixture",
    )
    assert server.phase == MCPLifecyclePhase.IDLE
    async with server:
        assert server.phase == MCPLifecyclePhase.READY
        await server.list_tools()
        # list_tools transitions through DISCOVERING_TOOLS and back to READY.
        assert server.phase == MCPLifecyclePhase.READY
    assert server.phase == MCPLifecyclePhase.SHUTDOWN
