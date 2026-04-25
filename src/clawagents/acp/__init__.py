"""Agent Client Protocol (ACP) adapter.

ACP is the JSON-RPC protocol used by IDEs like Zed to drive an external
agent. This module wraps a :class:`clawagents.agent.ClawAgent` so it can
be served over the protocol — letting a ClawAgents agent appear as a
first-class Zed agent.

The wire schema defined in :mod:`clawagents.acp.messages` is hermetic:
no optional dependencies needed. The :class:`AcpServer` in
:mod:`clawagents.acp.server` lazily imports the official
``agent-client-protocol`` package when you actually call ``serve()`` —
so importing this module never fails, even without ``acp`` installed.

Install the optional dependency::

    pip install "clawagents[acp]"

Public surface
--------------
* :class:`AcpError`, :class:`MissingAcpDependencyError`
* Message dataclasses: :class:`PromptRequest`, :class:`SessionUpdate`,
  :class:`AgentMessageChunk`, :class:`AgentThoughtChunk`,
  :class:`ToolCallStart`, :class:`ToolCallComplete`,
  :class:`PermissionRequest`
* :class:`AgentSession` — translates ClawAgents events to ACP updates
* :class:`AcpServer` — stdio JSON-RPC server entry point
"""

from clawagents.acp.errors import AcpError, MissingAcpDependencyError
from clawagents.acp.messages import (
    PromptRequest,
    SessionUpdate,
    AgentMessageChunk,
    AgentThoughtChunk,
    ToolCallStart,
    ToolCallComplete,
    PermissionRequest,
    PermissionDecision,
    StopReason,
    encode_update,
    decode_update,
)
from clawagents.acp.session import AgentSession, SessionEventSink
from clawagents.acp.server import AcpServer, ACP_AVAILABLE, serve

__all__ = [
    "AcpError",
    "MissingAcpDependencyError",
    "PromptRequest",
    "SessionUpdate",
    "AgentMessageChunk",
    "AgentThoughtChunk",
    "ToolCallStart",
    "ToolCallComplete",
    "PermissionRequest",
    "PermissionDecision",
    "StopReason",
    "encode_update",
    "decode_update",
    "AgentSession",
    "SessionEventSink",
    "AcpServer",
    "ACP_AVAILABLE",
    "serve",
]
