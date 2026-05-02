"""Manager for a list of :class:`MCPServer` instances.

The agent factory feeds its ``mcp_servers=`` list into a :class:`MCPServerManager`,
which:

  1. ``connect()``s every server.
  2. Calls ``list_tools()`` on each, bridges every tool into the supplied
     ``ToolRegistry`` via :class:`MCPBridgedTool`.
  3. Registers a finalizer so ``shutdown()`` is invoked when the agent run ends.

Shutdown is best-effort and concurrent: an error in one server's cleanup
never prevents the others from closing.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, TYPE_CHECKING

from clawagents.tracing import custom_span

if TYPE_CHECKING:
    from clawagents.mcp.server import MCPServer
    from clawagents.tools.registry import ToolRegistry


class MCPServerManager:
    """Lifecycles a collection of :class:`MCPServer` instances.

    Typical use::

        manager = MCPServerManager([MCPServerStdio(...)])
        await manager.start(registry)
        try:
            ...  # agent run
        finally:
            await manager.shutdown()
    """

    def __init__(
        self,
        servers: Iterable["MCPServer"],
        *,
        name_prefix_with_server: bool = False,
    ) -> None:
        self.servers: list["MCPServer"] = list(servers)
        self.name_prefix_with_server = name_prefix_with_server
        self._started = False
        self._registered_tool_names: list[str] = []

    @property
    def started(self) -> bool:
        return self._started

    async def start(self, registry: "ToolRegistry") -> list[str]:
        """Connect every server and bridge their tools into ``registry``.

        Returns the list of tool names registered. Idempotent: calling twice
        does nothing on the second call.
        """
        from clawagents.mcp.tool_bridge import mcp_tool_to_clawagents_tool

        if self._started:
            return list(self._registered_tool_names)

        with custom_span("mcp.manager.start", server_count=len(self.servers)):
            for server in self.servers:
                await server.connect()
                tools = await server.list_tools()
                prefix = server.name if self.name_prefix_with_server else None
                for descriptor in tools:
                    bridged = mcp_tool_to_clawagents_tool(descriptor, server, name_prefix=prefix)
                    registry.register(bridged)
                    self._registered_tool_names.append(bridged.name)

        self._started = True
        return list(self._registered_tool_names)

    async def shutdown(self) -> None:
        """Shut every server down sequentially, swallowing per-server errors.

        We deliberately avoid ``asyncio.gather`` here: the underlying MCP
        transports rely on AnyIO cancel scopes which require shutdown to run
        in the same task that opened them. Sequential shutdown keeps task
        affinity intact.
        """
        if not self.servers:
            self._started = False
            return
        with custom_span("mcp.manager.shutdown", server_count=len(self.servers)):
            for server in self.servers:
                try:
                    await server.shutdown()
                except Exception as exc:  # pragma: no cover — best-effort
                    with custom_span(
                        "mcp.manager.shutdown_error",
                        server=server.name,
                        error=str(exc),
                    ):
                        pass
        self._started = False

    def get_server_config(self, server_name: str) -> Any:
        """Return mutable transport params for a server when available."""
        for server in self.servers:
            if server.name == server_name:
                return getattr(server, "params", None)
        raise KeyError(f"No MCP server registered with name '{server_name}'")

    def update_server_config(self, server_name: str, config: Any) -> None:
        """Replace mutable transport params for a server when supported."""
        for server in self.servers:
            if server.name == server_name:
                if not hasattr(server, "params"):
                    raise TypeError(f"MCP server '{server_name}' does not expose mutable params")
                setattr(server, "params", config)
                return
        raise KeyError(f"No MCP server registered with name '{server_name}'")

    async def reconnect_all(self) -> None:
        """Shutdown and reconnect all servers without re-registering tools."""
        await self.shutdown()
        for server in self.servers:
            await server.connect()
        self._started = True

    async def update_server_auth(
        self,
        server_name: str,
        *,
        mode: str,
        value: str,
        key: str | None = None,
        reconnect: bool = True,
    ) -> Any:
        """Apply auth material to stdio/env or http header params."""
        config = self.get_server_config(server_name)
        if not isinstance(config, dict):
            raise TypeError(f"MCP server '{server_name}' params are not mutable dict config")
        updated = dict(config)
        if mode in {"env"}:
            env_key = key or "MCP_AUTH_TOKEN"
            env = dict(updated.get("env") or {})
            env[env_key] = value
            updated["env"] = env
        elif mode in {"bearer", "header"}:
            header_key = key or "Authorization"
            headers = dict(updated.get("headers") or {})
            headers[header_key] = f"Bearer {value}" if mode == "bearer" and header_key.lower() == "authorization" else value
            updated["headers"] = headers
        else:
            raise ValueError("mode must be bearer, header, or env")
        self.update_server_config(server_name, updated)
        if reconnect:
            await self.reconnect_all()
        return updated

    async def __aenter__(self) -> "MCPServerManager":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.shutdown()

    # ── Convenience pass-throughs ──

    async def list_all_tools(self) -> dict[str, list[Any]]:
        """Return ``{server_name: [MCPToolDescriptor, ...]}`` for every server."""
        out: dict[str, list[Any]] = {}
        for server in self.servers:
            out[server.name] = await server.list_tools()
        return out

    async def invoke_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Locate ``server_name`` and invoke ``tool_name`` on it."""
        for server in self.servers:
            if server.name == server_name:
                return await server.invoke_tool(tool_name, arguments)
        raise KeyError(f"No MCP server registered with name '{server_name}'")
