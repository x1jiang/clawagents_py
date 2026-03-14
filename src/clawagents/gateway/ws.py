"""
WebSocket handler for the ClawAgents gateway (FastAPI native).

Supports:
  - chat.send    — run an agent task with real-time streaming events
  - chat.history — retrieve session history
  - chat.inject  — inject an assistant note without triggering a run
  - ping         — keepalive
"""

from __future__ import annotations

import json
import time
import math
import secrets
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from clawagents.agent import create_claw_agent
from clawagents.process.command_queue import enqueue_command_in_lane, get_queue_size
from clawagents.process.lanes import CommandLane
from clawagents.gateway.protocol import is_valid_request, make_response, make_event

VALID_LANES = {"main", "cron", "subagent", "nested"}

_sessions: dict[str, dict] = {}


def _resolve_lane(raw: str | None) -> str:
    lane = (raw or "").strip().lower() or CommandLane.Main.value
    return lane if lane in VALID_LANES else CommandLane.Main.value


def _resolve_session(raw: str | None) -> str:
    if raw and raw.strip():
        return raw.strip()
    return f"ws-{int(time.time())}-{secrets.token_hex(4)}"


def _get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": []}
    return _sessions[session_id]


def attach_websocket(app: FastAPI, llm: Any, gateway_api_key: str):
    """Register the /ws WebSocket endpoint on the FastAPI app."""

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket, token: str = Query(default="")):
        if gateway_api_key and token != gateway_api_key:
            await ws.close(code=4001, reason="Unauthorized")
            return

        await ws.accept()

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json(make_response("?", False, "Invalid JSON"))
                    continue

                if not is_valid_request(msg):
                    await ws.send_json(make_response("?", False, "Invalid frame"))
                    continue

                method = msg["method"]
                if method == "ping":
                    await ws.send_json(make_response(msg["id"], True, {"pong": int(time.time() * 1000)}))

                elif method == "chat.send":
                    await _handle_chat_send(ws, msg, llm)

                elif method == "chat.history":
                    _handle_chat_history_sync(ws, msg)
                    await ws.send_json(_chat_history_response(msg))

                elif method == "chat.inject":
                    resp = _handle_chat_inject(msg)
                    await ws.send_json(resp)

                else:
                    await ws.send_json(make_response(msg["id"], False, f"Unknown method: {method}"))

        except WebSocketDisconnect:
            pass

    print("   WebSocket: enabled on ws:// /ws")


async def _handle_chat_send(ws: WebSocket, msg: dict, llm: Any):
    params = msg["params"]
    task = str(params.get("task", ""))
    if not task:
        await ws.send_json(make_response(msg["id"], False, "Missing 'task' parameter"))
        return

    lane = _resolve_lane(params.get("lane"))
    session_id = _resolve_session(params.get("sessionId"))
    session = _get_or_create_session(session_id)

    seq = 0

    async def send_event(event: str, payload: dict):
        nonlocal seq
        await ws.send_json(make_event(event, {**payload, "sessionId": session_id}, seq))
        seq += 1

    await send_event("queued", {"lane": lane, "position": get_queue_size(lane)})

    try:
        async def _execute():
            await send_event("started", {"lane": lane})
            agent = create_claw_agent(model=llm)

            async def on_event(kind, data):
                await send_event("agent", {"kind": kind, **(data if isinstance(data, dict) else {"data": data})})

            return await agent.invoke(task, on_event=on_event)

        result = await enqueue_command_in_lane(lane, _execute)

        now_ms = int(time.time() * 1000)
        session["messages"].append({"role": "user", "content": task, "timestamp": now_ms})
        session["messages"].append({"role": "assistant", "content": result.result or "", "timestamp": now_ms})

        await ws.send_json(make_response(msg["id"], True, {
            "sessionId": session_id,
            "lane": lane,
            "status": result.status,
            "result": result.result,
            "iterations": result.iterations,
        }))
    except Exception as e:
        await ws.send_json(make_response(msg["id"], False, str(e)))


def _chat_history_response(msg: dict) -> dict:
    session_id = _resolve_session(msg["params"].get("sessionId"))
    session = _sessions.get(session_id)
    return make_response(msg["id"], True, {
        "sessionId": session_id,
        "messages": session["messages"] if session else [],
    })


def _handle_chat_history_sync(ws: WebSocket, msg: dict):
    pass  # response built by _chat_history_response


def _handle_chat_inject(msg: dict) -> dict:
    params = msg["params"]
    session_id = _resolve_session(params.get("sessionId"))
    content = str(params.get("content", ""))
    if not content:
        return make_response(msg["id"], False, "Missing 'content' parameter")
    session = _get_or_create_session(session_id)
    session["messages"].append({"role": "assistant", "content": content, "timestamp": int(time.time() * 1000)})
    return make_response(msg["id"], True, {"sessionId": session_id, "injected": True})
