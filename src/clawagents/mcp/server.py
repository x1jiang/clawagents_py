"""MCP server abstractions: ``MCPServer`` ABC and stdio/SSE/Streamable-HTTP impls.

These wrap the official ``mcp`` SDK's :class:`mcp.client.session.ClientSession`
plus the transport-specific async context managers
(``stdio_client`` / ``sse_client`` / ``streamablehttp_client``). We do **not**
implement JSON-RPC framing ourselves.

Each server tracks an :class:`MCPLifecyclePhase` (Idle → Connecting →
Initializing → DiscoveringTools → Ready → Invoking → Errored / Shutdown),
emitting tracing spans on every transition.

The optional ``mcp`` SDK is imported lazily. ``import clawagents.mcp`` always
works; only ``connect()`` requires the SDK.
"""

from __future__ import annotations

import abc
import enum
import importlib.util
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional, TypedDict, Union

from clawagents.redact import is_secret_name
from clawagents.tracing import custom_span, tool_span

logger = logging.getLogger(__name__)


# ─── Environment scrubbing for stdio MCP servers ──────────────────────────


# Variables that are almost always required for a child process to work
# (locales, terminal type, shell, home directory, language) but are not
# secrets. Anything not on this list is dropped from the inherited parent
# environment by default to avoid leaking, e.g., ``OPENAI_API_KEY`` to an
# untrusted MCP server.
_SAFE_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "TZ",
    "TMPDIR",
    "PWD",
)


def _is_safe_passthrough(name: str) -> bool:
    if name in _SAFE_PASSTHROUGH_KEYS:
        return True
    # LC_ALL / LC_CTYPE / LC_TIME / etc.
    if name.startswith("LC_"):
        return True
    return False


def scrub_env_for_stdio(
    user_env: Optional[dict[str, str]],
    *,
    inherit_safe: bool = True,
    allowlist: Optional[Iterable[str]] = None,
    parent_env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Build a minimal, secret-free environment for an MCP stdio child.

    Policy:

    * Start with an empty dict.
    * If ``inherit_safe`` (default), copy the parent's locale / shell / path
      keys (see :data:`_SAFE_PASSTHROUGH_KEYS`).
    * Copy any explicit ``allowlist`` keys from the parent env. These are
      passed through verbatim — use this for variables the MCP server
      genuinely needs (``GITHUB_TOKEN``, ``DATABASE_URL``).
    * Apply ``user_env`` on top, overriding anything above.

    Anything *not* in the safe passthrough or allowlist is **dropped** —
    including ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``AWS_*``, etc.

    A diagnostic log is emitted listing the secret-named keys that were
    dropped, so operators can spot accidental leaks early.
    """
    src = parent_env if parent_env is not None else dict(os.environ)
    out: dict[str, str] = {}

    if inherit_safe:
        for k, v in src.items():
            if _is_safe_passthrough(k):
                out[k] = v

    if allowlist:
        for k in allowlist:
            if k in src:
                out[k] = src[k]

    if user_env:
        out.update(user_env)

    # Diagnostic: which secret-shaped parent vars did we drop?
    dropped_secrets = [
        k for k in src.keys()
        if is_secret_name(k) and k not in out
    ]
    if dropped_secrets:
        logger.debug(
            "clawagents.mcp: dropped %d secret-named env vars from stdio child "
            "(use env_allowlist to inherit explicitly): %s",
            len(dropped_secrets),
            ", ".join(sorted(dropped_secrets)[:8])
            + (" …" if len(dropped_secrets) > 8 else ""),
        )

    return out


# ─── SDK probe ────────────────────────────────────────────────────────────


_MCP_INSTALL_HINT = (
    "The Model Context Protocol SDK is required to use MCPServerStdio / "
    "MCPServerSse / MCPServerStreamableHttp. "
    "Install it with: pip install 'clawagents[mcp]' "
    "(or directly: pip install mcp)."
)


def is_mcp_sdk_available() -> bool:
    """Return ``True`` when the optional ``mcp`` package is importable."""
    return importlib.util.find_spec("mcp") is not None


def require_mcp_sdk() -> None:
    """Raise :class:`ImportError` with a helpful message if ``mcp`` is missing."""
    if not is_mcp_sdk_available():
        raise ImportError(_MCP_INSTALL_HINT)


# ─── Lifecycle phases (port from claw-code's hardened model) ──────────────


class MCPLifecyclePhase(str, enum.Enum):
    """Lifecycle phases for an :class:`MCPServer`.

    Mirrors the hardened state machine in
    ``claw-code-main/rust/crates/runtime/src/mcp_lifecycle_hardened.rs``,
    collapsed to the minimum set the clawagents loop cares about.
    """

    IDLE = "idle"
    CONNECTING = "connecting"
    INITIALIZING = "initializing"
    DISCOVERING_TOOLS = "discovering_tools"
    READY = "ready"
    INVOKING = "invoking"
    ERRORED = "errored"
    SHUTDOWN = "shutdown"


# ─── Param types (mirror openai-agents-python's MCPServerStdioParams shape) ──


class MCPServerStdioParams(TypedDict, total=False):
    """Stdio transport parameters.

    Security note: by default the parent process's environment is **not**
    forwarded to the child wholesale. Only locale / path / shell variables
    are inherited (see :func:`scrub_env_for_stdio`). Use ``env_allowlist`` to
    forward specific keys (e.g. ``["GITHUB_TOKEN", "DATABASE_URL"]``), or
    set ``inherit_safe_env=False`` and pass everything explicitly via ``env``.

    Operators that want the legacy "inherit everything" behaviour can set the
    ``CLAW_MCP_INHERIT_ALL_ENV=1`` env var.
    """

    command: str
    args: list[str]
    env: dict[str, str]
    env_allowlist: list[str]
    inherit_safe_env: bool
    cwd: Union[str, Path]
    encoding: str


class MCPServerSseParams(TypedDict, total=False):
    """HTTP + SSE transport parameters."""

    url: str
    headers: dict[str, str]
    timeout: float
    sse_read_timeout: float


class MCPServerStreamableHttpParams(TypedDict, total=False):
    """Streamable-HTTP transport parameters."""

    url: str
    headers: dict[str, str]
    timeout: float
    sse_read_timeout: float
    terminate_on_close: bool


# ─── Tool descriptor (transport-agnostic view of an MCP tool) ─────────────


@dataclass
class MCPToolDescriptor:
    """A normalized description of an MCP tool, decoupled from the SDK type."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""

    @classmethod
    def from_sdk_tool(cls, tool: Any, server_name: str) -> "MCPToolDescriptor":
        """Build a descriptor from an ``mcp.types.Tool`` (or duck-equivalent)."""
        input_schema: dict[str, Any] = {}
        raw_schema = getattr(tool, "inputSchema", None)
        if isinstance(raw_schema, dict):
            input_schema = raw_schema
        return cls(
            name=getattr(tool, "name", "") or "",
            description=getattr(tool, "description", "") or "",
            input_schema=input_schema,
            server_name=server_name,
        )


ToolFilter = Optional[Callable[[MCPToolDescriptor], Union[bool, Awaitable[bool]]]]


# ─── MCPServer ABC ────────────────────────────────────────────────────────


class MCPServer(abc.ABC):
    """Abstract base class for MCP servers.

    Subclasses implement :meth:`_create_streams` to yield the transport's
    ``(read_stream, write_stream)`` (or ``(read, write, get_session_id)``)
    pair. Everything else — the ``ClientSession`` lifecycle, lifecycle
    tracing, error surfacing — is shared.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        tool_filter: ToolFilter = None,
        cache_tools_list: bool = False,
        client_session_timeout_seconds: Optional[float] = 5.0,
    ) -> None:
        self._name_override = name
        self.tool_filter = tool_filter
        self.cache_tools_list = cache_tools_list
        self.client_session_timeout_seconds = client_session_timeout_seconds

        self._phase: MCPLifecyclePhase = MCPLifecyclePhase.IDLE
        self._session: Any = None  # mcp.client.session.ClientSession when connected
        self._exit_stack: Optional[AsyncExitStack] = None
        self._tools_cache: Optional[list[MCPToolDescriptor]] = None
        self._last_error: Optional[str] = None

    # ── Sub-class hooks ──

    @abc.abstractmethod
    def _default_name(self) -> str:
        """Return a default human-readable name for this transport."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_streams(self) -> Any:
        """Return an async context manager that yields ``(read, write[, ...])``.

        Implementations call into the SDK's transport helpers
        (``stdio_client``, ``sse_client``, ``streamablehttp_client``).
        """
        raise NotImplementedError

    # ── Public API ──

    @property
    def name(self) -> str:
        return self._name_override or self._default_name()

    @property
    def phase(self) -> MCPLifecyclePhase:
        return self._phase

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _transition(self, phase: MCPLifecyclePhase, error: Optional[str] = None) -> None:
        """Move to ``phase`` and emit a tracing span recording the transition."""
        prev = self._phase
        self._phase = phase
        if error is not None:
            self._last_error = error
        with custom_span(
            f"mcp.lifecycle.{phase.value}",
            server=self.name,
            from_phase=prev.value,
            to_phase=phase.value,
            error=error,
        ):
            pass

    async def connect(self) -> None:
        """Spawn the transport, open a ClientSession, run the MCP handshake."""
        require_mcp_sdk()
        if self._phase not in (MCPLifecyclePhase.IDLE, MCPLifecyclePhase.SHUTDOWN, MCPLifecyclePhase.ERRORED):
            return  # Already connecting / ready
        from mcp import ClientSession

        self._transition(MCPLifecyclePhase.CONNECTING)
        self._exit_stack = AsyncExitStack()
        try:
            transport = await self._exit_stack.enter_async_context(self._create_streams())
            # stdio_client / sse_client return (read, write); streamablehttp_client returns (read, write, get_session_id)
            read, write, *_rest = transport

            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            self._transition(MCPLifecyclePhase.INITIALIZING)
            await session.initialize()
            self._session = session
            self._transition(MCPLifecyclePhase.READY)
        except BaseException as exc:
            self._transition(MCPLifecyclePhase.ERRORED, error=str(exc))
            try:
                if self._exit_stack is not None:
                    await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None
            self._session = None
            raise

    async def list_tools(self, *, force_refresh: bool = False) -> list[MCPToolDescriptor]:
        """Return the tools advertised by the server (filtered + optionally cached)."""
        if self._session is None:
            raise RuntimeError(
                f"MCP server '{self.name}' is not connected. Call connect() first."
            )
        if self.cache_tools_list and not force_refresh and self._tools_cache is not None:
            return self._tools_cache

        prev_phase = self._phase
        self._transition(MCPLifecyclePhase.DISCOVERING_TOOLS)
        try:
            result = await self._session.list_tools()
        except BaseException as exc:
            self._transition(MCPLifecyclePhase.ERRORED, error=str(exc))
            raise
        finally:
            if self._phase == MCPLifecyclePhase.DISCOVERING_TOOLS:
                # Restore Ready unless we erred above.
                self._transition(MCPLifecyclePhase.READY)
            elif prev_phase == MCPLifecyclePhase.READY and self._phase == MCPLifecyclePhase.READY:
                pass

        descriptors = [
            MCPToolDescriptor.from_sdk_tool(t, server_name=self.name)
            for t in getattr(result, "tools", []) or []
        ]
        if self.tool_filter is not None:
            descriptors = await _apply_tool_filter(descriptors, self.tool_filter)
        if self.cache_tools_list:
            self._tools_cache = descriptors
        return descriptors

    async def invoke_tool(
        self, tool_name: str, arguments: Optional[dict[str, Any]] = None
    ) -> Any:
        """Call ``tool_name`` on the server. Returns the raw ``CallToolResult``."""
        if self._session is None:
            raise RuntimeError(
                f"MCP server '{self.name}' is not connected. Call connect() first."
            )
        with tool_span(f"mcp.{self.name}.{tool_name}", server=self.name, tool=tool_name):
            self._transition(MCPLifecyclePhase.INVOKING)
            try:
                result = await self._session.call_tool(tool_name, arguments or {})
            except BaseException as exc:
                self._transition(MCPLifecyclePhase.ERRORED, error=str(exc))
                raise
            self._transition(MCPLifecyclePhase.READY)
            return result

    async def shutdown(self) -> None:
        """Close the session and underlying transport."""
        if self._exit_stack is None:
            return
        try:
            await self._exit_stack.aclose()
        except Exception as exc:  # pragma: no cover — cleanup best-effort
            self._last_error = str(exc)
        finally:
            self._exit_stack = None
            self._session = None
            self._tools_cache = None
            self._transition(MCPLifecyclePhase.SHUTDOWN)

    async def __aenter__(self) -> "MCPServer":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.shutdown()


async def _apply_tool_filter(
    tools: list[MCPToolDescriptor],
    tool_filter: Callable[[MCPToolDescriptor], Union[bool, Awaitable[bool]]],
) -> list[MCPToolDescriptor]:
    import inspect
    out: list[MCPToolDescriptor] = []
    for t in tools:
        verdict = tool_filter(t)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        if verdict:
            out.append(t)
    return out


# ─── MCPServerStdio ───────────────────────────────────────────────────────


class MCPServerStdio(MCPServer):
    """MCP server speaking JSON-RPC over a child process's stdio.

    Mirrors openai-agents-python's
    :class:`agents.mcp.MCPServerStdio` constructor shape.
    """

    def __init__(
        self,
        params: MCPServerStdioParams,
        *,
        name: Optional[str] = None,
        tool_filter: ToolFilter = None,
        cache_tools_list: bool = False,
        client_session_timeout_seconds: Optional[float] = 5.0,
    ) -> None:
        super().__init__(
            name=name,
            tool_filter=tool_filter,
            cache_tools_list=cache_tools_list,
            client_session_timeout_seconds=client_session_timeout_seconds,
        )
        self.params: MCPServerStdioParams = params

    def _default_name(self) -> str:
        return f"stdio: {self.params.get('command', '?')}"

    def _create_streams(self) -> Any:
        require_mcp_sdk()
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        # Compute the scrubbed env *eagerly* so the parent's secrets never
        # reach the SDK call. The escape hatch ``CLAW_MCP_INHERIT_ALL_ENV=1``
        # restores the old "inherit everything when env is None" semantics
        # for operators that explicitly opt in.
        if os.environ.get("CLAW_MCP_INHERIT_ALL_ENV", "").lower() in {"1", "true", "yes"}:
            scrubbed_env: Optional[dict[str, str]] = self.params.get("env")
        else:
            scrubbed_env = scrub_env_for_stdio(
                self.params.get("env"),
                inherit_safe=self.params.get("inherit_safe_env", True),
                allowlist=self.params.get("env_allowlist"),
            )

        sdk_params = StdioServerParameters(
            command=self.params["command"],
            args=list(self.params.get("args") or []),
            env=scrubbed_env,
            cwd=self.params.get("cwd"),
            encoding=self.params.get("encoding", "utf-8"),
        )
        return stdio_client(sdk_params)


# ─── MCPServerSse ─────────────────────────────────────────────────────────


class MCPServerSse(MCPServer):
    """MCP server over HTTP + SSE."""

    def __init__(
        self,
        params: MCPServerSseParams,
        *,
        name: Optional[str] = None,
        tool_filter: ToolFilter = None,
        cache_tools_list: bool = False,
        client_session_timeout_seconds: Optional[float] = 5.0,
    ) -> None:
        super().__init__(
            name=name,
            tool_filter=tool_filter,
            cache_tools_list=cache_tools_list,
            client_session_timeout_seconds=client_session_timeout_seconds,
        )
        self.params: MCPServerSseParams = params

    def _default_name(self) -> str:
        return f"sse: {self.params.get('url', '?')}"

    def _create_streams(self) -> Any:
        require_mcp_sdk()
        from mcp.client.sse import sse_client

        kwargs: dict[str, Any] = {"url": self.params["url"]}
        for key in ("headers", "timeout", "sse_read_timeout"):
            if key in self.params:
                kwargs[key] = self.params[key]  # type: ignore[literal-required]
        return sse_client(**kwargs)


# ─── MCPServerStreamableHttp ──────────────────────────────────────────────


class MCPServerStreamableHttp(MCPServer):
    """MCP server over the Streamable HTTP transport."""

    def __init__(
        self,
        params: MCPServerStreamableHttpParams,
        *,
        name: Optional[str] = None,
        tool_filter: ToolFilter = None,
        cache_tools_list: bool = False,
        client_session_timeout_seconds: Optional[float] = 5.0,
    ) -> None:
        super().__init__(
            name=name,
            tool_filter=tool_filter,
            cache_tools_list=cache_tools_list,
            client_session_timeout_seconds=client_session_timeout_seconds,
        )
        self.params: MCPServerStreamableHttpParams = params

    def _default_name(self) -> str:
        return f"streamable_http: {self.params.get('url', '?')}"

    def _create_streams(self) -> Any:
        require_mcp_sdk()
        from mcp.client.streamable_http import streamablehttp_client

        kwargs: dict[str, Any] = {"url": self.params["url"]}
        for key in ("headers", "timeout", "sse_read_timeout", "terminate_on_close"):
            if key in self.params:
                kwargs[key] = self.params[key]  # type: ignore[literal-required]
        return streamablehttp_client(**kwargs)
