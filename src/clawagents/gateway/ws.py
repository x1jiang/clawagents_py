"""
WebSocket handler for the ClawAgents gateway (FastAPI native).

Supports:
  - chat.send    — run an agent task with real-time streaming events
  - chat.history — retrieve session history
  - chat.inject  — inject an assistant note without triggering a run
  - ping         — keepalive
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import math
import secrets
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

from clawagents.agent import create_claw_agent
from clawagents.process.command_queue import enqueue_command_in_lane, get_queue_size
from clawagents.process.lanes import CommandLane
from clawagents.gateway.protocol import is_valid_request, make_response, make_event

# Upper bound on buffered outbound WS events. Observatory payloads run ~1.5 MB
# per LLM call, so an unbounded queue behind a slow client retains hundreds of
# MB over a long run. Shed past this and report the count once at the end.
_WS_EVENT_QUEUE_MAX = 10_000

VALID_LANES = {"main", "cron", "subagent", "nested"}

# Bounded LRU: every sessionId-less message used to mint a fresh entry that
# was retained forever — a memory leak plus indefinite transcript retention.
_MAX_SESSIONS = 500
_MAX_MESSAGES_PER_SESSION = 200

_sessions: dict[str, dict] = {}


def _resolve_lane(raw: str | None) -> str:
    lane = (raw or "").strip().lower() or CommandLane.Main.value
    return lane if lane in VALID_LANES else CommandLane.Main.value


def _resolve_session(raw: str | None) -> str:
    if raw and raw.strip():
        return raw.strip()
    return f"ws-{int(time.time())}-{secrets.token_hex(4)}"


def _get_or_create_session(session_id: str) -> dict:
    # Refresh recency (dicts preserve insertion order → oldest first).
    session = _sessions.pop(session_id, None)
    if session is None:
        session = {"messages": []}
    _sessions[session_id] = session
    while len(_sessions) > _MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
    return session


def _push_messages(session: dict, *msgs: dict) -> None:
    session["messages"].extend(msgs)
    overflow = len(session["messages"]) - _MAX_MESSAGES_PER_SESSION
    if overflow > 0:
        del session["messages"][:overflow]


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

            # One ordered queue for BOTH agent and observatory events.
            #
            # ``on_event`` must be sync: the OnEvent contract is
            # ``Callable[[EventKind, dict], None]`` and RunEvents.emit discards
            # the return value, so an ``async def`` here produced a coroutine
            # per event that was never awaited — every agent stream event was
            # silently dropped and the WS client saw only queued/started/final.
            # Sharing one queue with the observatory pump also keeps the two
            # streams in their true emission order.
            event_queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue(
                maxsize=_WS_EVENT_QUEUE_MAX
            )
            dropped = 0

            def _enqueue(event: str, payload: dict) -> None:
                nonlocal dropped
                try:
                    event_queue.put_nowait((event, payload))
                except asyncio.QueueFull:
                    # A client too slow to drain must not grow the queue without
                    # bound; shed instead and report the count once at the end.
                    dropped += 1

            def on_event(kind, data):
                _enqueue(
                    "agent",
                    {"kind": kind, **(data if isinstance(data, dict) else {"data": data})},
                )

            async def _pump() -> None:
                while True:
                    item = await event_queue.get()
                    if item is None:
                        break
                    event, payload = item
                    await send_event(event, payload)

            pump = asyncio.create_task(_pump())

            async def _drain() -> None:
                await event_queue.put(None)
                try:
                    await pump
                except Exception:
                    pass
                if dropped:
                    with contextlib.suppress(Exception):
                        await send_event(
                            "warning",
                            {
                                "message": (
                                    f"dropped {dropped} event(s): client could not "
                                    "keep up with the stream"
                                )
                            },
                        )

            # Check if Context Observatory recording is enabled by params setting or env
            enable_obs = bool(
                params.get("enable_context_observatory")
                or params.get("context_observatory")
                or os.environ.get("CLAWAGENTS_ENABLE_CONTEXT_OBSERVATORY") == "1"
            )

            if enable_obs:
                from clawagents.context_observatory.hooks import ContextObserverHooks
                from clawagents.context_observatory.store import EventStore
                from clawagents.graph.model_profiles import resolve_model_profile

                model_name = getattr(llm, "model", None) or getattr(llm, "name", None) or str(llm)
                profile = resolve_model_profile(str(model_name))
                context_window = int(
                    params.get("context_window")
                    or (profile["max_input_tokens"] if profile else 128_000)
                )
                store = EventStore()
                store.set_session_meta(
                    model=str(model_name),
                    context_window=context_window,
                    started_at=time.time(),
                )
                # Observatory publishes ride the same ordered queue as agent
                # events, so bursts cannot reorder relative to the stream.
                def _obs_sink(event: Any) -> None:
                    _enqueue("observatory", event.to_dict())

                observer = ContextObserverHooks(
                    store=store,
                    model=str(model_name),
                    context_window=context_window,
                    event_sink=_obs_sink,
                )

                try:
                    result = await agent.invoke(task, on_event=on_event, hooks=observer)
                    store.set_session_meta(completed_at=time.time(), status=result.status)
                    store.auto_save(chat_id=session_id)
                    return result
                except Exception:
                    store.set_session_meta(completed_at=time.time(), status="failed")
                    store.auto_save(chat_id=session_id)
                    raise
                finally:
                    await _drain()
            else:
                try:
                    return await agent.invoke(task, on_event=on_event)
                finally:
                    await _drain()

        result = await enqueue_command_in_lane(lane, _execute)

        now_ms = int(time.time() * 1000)
        _push_messages(
            session,
            {"role": "user", "content": task, "timestamp": now_ms},
            {"role": "assistant", "content": result.result or "", "timestamp": now_ms},
        )

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
    _push_messages(session, {"role": "assistant", "content": content, "timestamp": int(time.time() * 1000)})
    return make_response(msg["id"], True, {"sessionId": session_id, "injected": True})
