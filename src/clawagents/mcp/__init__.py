"""MCP (Model Context Protocol) client integration for clawagents.

Bridges external MCP servers into the clawagents :class:`ToolRegistry` so the
agent loop can call them like any other tool.

Public surface (mirrors the TypeScript port):

  - :class:`MCPServer`              — abstract base
  - :class:`MCPServerStdio`         — spawns a subprocess and speaks MCP over stdio
  - :class:`MCPServerSse`           — connects over HTTP+SSE
  - :class:`MCPServerStreamableHttp` — connects over Streamable HTTP
  - :class:`MCPServerManager`       — lifecycles a list of servers
  - :class:`MCPLifecyclePhase`      — hardened lifecycle state machine
  - :func:`mcp_tool_to_clawagents_tool` — adapts an MCP tool description into
    a clawagents :class:`Tool`

The optional ``mcp`` extra installs the Python SDK (``pip install
clawagents[mcp]``). Importing this subpackage works without the SDK; only
*starting* a server raises :class:`ImportError` when the SDK is missing.
"""

from clawagents.mcp.server import (
    MCPServer,
    MCPServerStdio,
    MCPServerSse,
    MCPServerStreamableHttp,
    MCPLifecyclePhase,
    MCPToolDescriptor,
    is_mcp_sdk_available,
    require_mcp_sdk,
    scrub_env_for_stdio,
)
from clawagents.mcp.manager import MCPServerManager
from clawagents.mcp.tool_bridge import mcp_tool_to_clawagents_tool, MCPBridgedTool

__all__ = [
    "MCPServer",
    "MCPServerStdio",
    "MCPServerSse",
    "MCPServerStreamableHttp",
    "MCPServerManager",
    "MCPLifecyclePhase",
    "MCPToolDescriptor",
    "is_mcp_sdk_available",
    "require_mcp_sdk",
    "scrub_env_for_stdio",
    "mcp_tool_to_clawagents_tool",
    "MCPBridgedTool",
]
