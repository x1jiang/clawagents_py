"""Stdio JSON-RPC ACP server wrapper around a ClawAgents agent.

The heavy lifting of the JSON-RPC framing is delegated to the official
``agent-client-protocol`` package — we only contribute the bridge
between *our* agent loop and *its* ``Agent`` interface.

Importing this module never fails: the optional ``acp`` package is
loaded lazily inside :meth:`AcpServer.serve`. If it is missing we raise
:class:`MissingAcpDependencyError` with a clear install hint.

Typical usage::

    from clawagents import create_claw_agent
    from clawagents.acp import AcpServer

    agent = create_claw_agent(name="claw")
    AcpServer(agent).serve()  # blocks on stdin/stdout

For tests, :func:`run_prompt` exposes the inner translation pipeline
without requiring the optional package or stdio plumbing.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from clawagents.acp.errors import MissingAcpDependencyError
from clawagents.acp.messages import PromptRequest, StopReason
from clawagents.acp.session import AgentSession, PermissionRequester


logger = logging.getLogger(__name__)


# Whether the optional `acp` package is currently importable. Probed
# lazily so that simply importing :mod:`clawagents.acp` is cheap and
# doesn't trigger filesystem walks on every call.
def _probe_acp() -> bool:
    try:
        importlib.import_module("acp")
        return True
    except Exception:
        return False


ACP_AVAILABLE: bool = _probe_acp()


# Type alias for "anything that looks like a ClawAgent" — we keep this
# loose so the server can wrap mock agents in tests.
AgentLike = Any

# Async runner: takes a ClawAgents agent + a prompt + an event sink and
# drives one prompt cycle. Used so tests can swap in a mock without
# pulling the full agent loop.
PromptRunner = Callable[
    [AgentLike, PromptRequest, AgentSession], Awaitable[StopReason]
]


# ──────────────────────────────────────────────────────────────────────
# Default agent runner — wires PromptRequest into ClawAgent.run()
# ──────────────────────────────────────────────────────────────────────


async def _default_runner(
    agent: AgentLike, prompt: PromptRequest, session: AgentSession
) -> StopReason:
    """Drive a single ACP prompt → ClawAgent run cycle.

    The agent's event sink is replaced with a sync forwarder that pushes
    onto an asyncio queue; a drain coroutine consumes the queue and
    awaits the (async) ACP sink. This avoids the classic
    ``ensure_future`` ordering bug where the agent finishes before its
    streamed events have been flushed.
    """

    saved_on_event = getattr(agent, "on_event", None)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def push(kind: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        # If we're on the event-loop thread, put_nowait() is correct
        # and ordering-preserving. From a worker thread we must use
        # call_soon_threadsafe; we detect that case so the same `push`
        # works both ways.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            queue.put_nowait((kind, dict(payload or {})))
        else:
            loop.call_soon_threadsafe(
                queue.put_nowait, (kind, dict(payload or {}))
            )

    if hasattr(agent, "on_event"):
        try:
            agent.on_event = push
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not attach on_event to agent", exc_info=True)

    drained_done = asyncio.Event()

    async def drainer() -> None:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                kind, payload = item
                await session.adispatch(kind, payload)
        finally:
            drained_done.set()

    drain_task = asyncio.create_task(drainer())
    try:
        run_fn = (
            getattr(agent, "arun", None)
            or getattr(agent, "run", None)
            or getattr(agent, "invoke", None)
        )
        if run_fn is None:
            raise RuntimeError("agent has no run() / arun() / invoke() method")

        result = run_fn(prompt.text)
        if asyncio.iscoroutine(result):
            output = await result
        else:
            output = result
        if not isinstance(output, str) and hasattr(output, "result"):
            output = output.result

        await queue.put(None)
        await drained_done.wait()

        if not session.emitted and isinstance(output, str) and output:
            await session.adispatch("message_text", {"text": output})
        return session.stop_reason or StopReason.END_TURN
    except asyncio.CancelledError:
        await queue.put(None)
        return StopReason.CANCELLED
    except Exception as exc:
        logger.exception("ACP prompt runner failed: %s", exc)
        await queue.put(None)
        await drained_done.wait()
        await session.adispatch("error", {"error": str(exc)})
        return StopReason.ERROR
    finally:
        if not drain_task.done():
            drain_task.cancel()
        if saved_on_event is not None:
            try:
                agent.on_event = saved_on_event
            except Exception:  # pragma: no cover - defensive
                pass


# ──────────────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────────────


@dataclass
class AcpServer:
    """ACP server fronting a ClawAgents agent.

    Parameters
    ----------
    agent:
        A ClawAgents ``ClawAgent`` (or anything with a compatible
        ``run`` / ``arun`` method).
    prompt_runner:
        Override the default runner — useful for tests.
    permission_requester:
        Async callback invoked when the agent wants to ask for a
        sensitive action's approval. ``None`` means auto-allow.
    """

    agent: AgentLike
    prompt_runner: PromptRunner = field(default=_default_runner)
    permission_requester: Optional[PermissionRequester] = None

    def serve(self) -> None:
        """Run the server, blocking on stdin/stdout until EOF.

        Raises :class:`MissingAcpDependencyError` if the ``acp`` package
        is not installed.
        """

        try:
            acp = importlib.import_module("acp")
        except Exception as exc:  # pragma: no cover - covered by mock test
            raise MissingAcpDependencyError(exc) from exc

        asyncio.run(self._serve_async(acp))

    async def run_prompt(
        self, prompt: PromptRequest, sink: Callable[[Dict[str, Any]], Any]
    ) -> StopReason:
        """Drive one prompt cycle with a custom sink (test entry point)."""

        session = AgentSession(
            session_id=prompt.session_id,
            sink=sink,
            permission_requester=self.permission_requester,
        )
        return await self.prompt_runner(self.agent, prompt, session)

    async def _serve_async(self, acp: Any) -> None:  # pragma: no cover - I/O
        """Internal ACP server loop; only entered when acp is installed."""

        # The official package's surface evolves quickly; we only rely
        # on a small subset documented as stable: stdio_streams() and
        # AgentSideConnection. Anything richer (UI elements, slash
        # commands) lives in user code that subclasses ``Agent``.
        try:
            stdio_streams = getattr(acp, "stdio_streams")
            agent_side = getattr(acp, "AgentSideConnection")
        except AttributeError as exc:
            raise MissingAcpDependencyError(exc) from exc

        reader, writer = await stdio_streams()
        ClawAcpAgent = self._make_agent_class(acp)
        connection = agent_side(lambda conn: ClawAcpAgent(conn, self), reader, writer)
        # Block until either side closes.
        await getattr(connection, "wait_closed", asyncio.Event().wait)()

    def _make_agent_class(self, acp: Any) -> type:  # pragma: no cover - I/O
        Agent = getattr(acp, "Agent")
        server_self = self

        class ClawAcpAgent(Agent):  # type: ignore[misc, valid-type]
            def __init__(self, conn: Any, srv: "AcpServer") -> None:
                super().__init__()
                self._conn = conn
                self._srv = srv

            async def initialize(self, params: Any) -> Any:
                return {
                    "agentCapabilities": {"loadSession": False},
                    "protocolVersion": getattr(params, "protocol_version", 1),
                }

            async def newSession(self, params: Any) -> Any:
                import uuid

                return {"sessionId": f"sess_{uuid.uuid4().hex[:12]}"}

            async def prompt(self, params: Any) -> Any:
                # Translate the SDK params into our PromptRequest.
                payload = (
                    params.model_dump() if hasattr(params, "model_dump") else dict(params)
                )
                req = PromptRequest.from_dict(payload)

                async def sink(raw: Dict[str, Any]) -> None:
                    await self._conn.session_update(req.session_id, raw)

                stop = await server_self.run_prompt(req, sink)
                return {"stopReason": stop.value}

            async def cancel(self, params: Any) -> Any:
                # Cancellation surfacing: just return; in a richer
                # implementation we'd cancel the running task.
                return None

        return ClawAcpAgent


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────


def serve(
    agent: AgentLike,
    *,
    prompt_runner: Optional[PromptRunner] = None,
    permission_requester: Optional[PermissionRequester] = None,
) -> None:
    """Shortcut: ``serve(agent)`` is sugar for ``AcpServer(agent).serve()``."""

    srv = AcpServer(
        agent=agent,
        prompt_runner=prompt_runner or _default_runner,
        permission_requester=permission_requester,
    )
    srv.serve()
