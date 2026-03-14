"""
JSON-RPC-style WebSocket protocol for ClawAgents Gateway.

Follows a similar pattern to OpenClaw's gateway protocol:
  - Inbound:  { type: "req", id, method, params }
  - Outbound: { type: "res", id, ok, payload?, error? }
  - Events:   { type: "event", event, payload, seq }
"""

from __future__ import annotations
from typing import Any


def is_valid_request(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    return (
        msg.get("type") == "req"
        and isinstance(msg.get("id"), str)
        and isinstance(msg.get("method"), str)
        and isinstance(msg.get("params"), dict)
    )


def make_response(req_id: str, ok: bool, payload_or_error: Any) -> dict:
    if ok:
        return {"type": "res", "id": req_id, "ok": True, "payload": payload_or_error}
    return {"type": "res", "id": req_id, "ok": False, "error": str(payload_or_error)}


def make_event(event: str, payload: dict, seq: int) -> dict:
    return {"type": "event", "event": event, "payload": payload, "seq": seq}
