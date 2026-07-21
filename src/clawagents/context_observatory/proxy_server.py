"""Observatory Proxy Server — standalone FastAPI server for context observability.

Runs on a separate port (default 3002) from the main ClawAgents gateway.
Provides the same ``/chat/stream`` SSE interface that the Streamlit UI expects,
but with observatory hooks baked in for full context inspection.

**Zero modifications to the core gateway.**  This server independently creates
an agent, wraps it with observation hooks, and streams events to the UI.

Usage::

    # Start as part of the observatory (auto-launched by __main__.py)
    python -m clawagents.context_observatory

    # Or start standalone
    python -m clawagents.context_observatory.proxy_server --port 3002
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

_PREVIEW = 2000
_SYS_FULL = 50_000


# ── Observatory hooks (self-contained) ───────────────────────────────────


def _make_context_hooks(sse):
    """Create RunHooks that emit ``llm_context`` events via the SSE callback."""
    from clawagents.lifecycle import RunHooks

    class _ContextEmittingHooks(RunHooks):
        async def on_llm_start(self, context, model, messages):
            try:
                msg_snapshots: list[dict[str, Any]] = []
                total_tokens = 0
                tokens_by_role: dict[str, int] = {}

                for m in messages:
                    role = getattr(m, "role", "unknown")
                    content = getattr(m, "content", "")
                    content_str = (
                        content if isinstance(content, str)
                        else str(content) if content is not None
                        else ""
                    )
                    content_len = len(content_str)
                    tok_est = max(content_len // 4, 1)
                    total_tokens += tok_est
                    tokens_by_role[role] = tokens_by_role.get(role, 0) + tok_est

                    limit = _SYS_FULL if role == "system" else _PREVIEW
                    msg_snapshots.append({
                        "role": role,
                        "content_preview": content_str[:_PREVIEW],
                        "content_length": content_len,
                        "token_count": tok_est,
                        "has_tool_calls": bool(getattr(m, "tool_calls_meta", None)),
                        "tool_call_id": getattr(m, "tool_call_id", None),
                        "full_content": content_str[:limit],
                    })

                sse("agent", {
                    "kind": "llm_context",
                    "data": {
                        "model": model or "",
                        "messages": msg_snapshots,
                        "total_input_tokens": total_tokens,
                        "tokens_by_role": tokens_by_role,
                    },
                })
            except Exception:
                logger.debug("Observatory context hook error", exc_info=True)

    return _ContextEmittingHooks()


async def _invoke_with_hooks(agent, task, sse, payload=None):
    """Invoke an agent with observatory hooks applied.

    Applies permission bypass, dummy ask_user, and context-emitting hooks.
    """
    if payload is None:
        payload = {}

    # ── Permission bypass ────────────────────────────────────────────
    from clawagents.permissions.mode import PermissionMode
    agent._default_permission_mode = PermissionMode.BYPASS
    agent.approval_handler = None
    agent.require_approval_tools = []

    # ── Prevent blocking on CLI input ────────────────────────────────
    def _dummy_ask_user(question: str) -> str:
        return (
            "Auto-reply: the user is not available to answer questions "
            "in the Context Observatory. Please proceed without asking."
        )

    from clawagents.tools.interactive import AskUserTool
    agent.tools.register(AskUserTool(ask_fn=_dummy_ask_user))

    # ── on_event: old-style emit() callback ──────────────────────────
    _STREAM_COVERED = frozenset({
        "approval_required",
        "tool_started", "tool_completed", "tool_skipped",
        "tool_call", "tool_result",
    })

    def on_event(kind: str, data: Any) -> None:
        if kind in _STREAM_COVERED:
            return
        sse("agent", {"kind": kind, "data": data})

    # ── on_stream_event: typed StreamEvent callback ──────────────────
    from clawagents.stream_events import StreamEvent

    def on_stream_event(ev: StreamEvent) -> None:
        kind = ev.kind
        if kind == "approval_required":
            return
        ev_payload = dict(ev.data) if ev.data else {}
        for attr in ev.__dataclass_fields__:
            if attr not in ("kind", "data"):
                val = getattr(ev, attr, None)
                if val is not None:
                    ev_payload[attr] = val
        if kind == "usage":
            sse("usage", ev_payload)
        else:
            sse("agent", {"kind": kind, "data": ev_payload})

    # ── RunHooks for context snapshots ───────────────────────────────
    hooks = _make_context_hooks(sse)

    # ── Session mounting ─────────────────────────────────────────────
    from clawagents.paths import get_sessions_dir
    from clawagents.session.backends import JsonlFileSession

    chat_id = payload.get("chat_id") or payload.get("session_id")
    session = None
    if chat_id:
        session = JsonlFileSession(chat_id, dir_path=get_sessions_dir())

    return await agent.invoke(
        task,
        on_event=on_event,
        on_stream_event=on_stream_event,
        hooks=hooks,
        session=session,
        features={"session_persistence": True} if session else None,
    )


# ── FastAPI application ──────────────────────────────────────────────────


def create_observatory_app() -> FastAPI:
    """Create the observatory proxy FastAPI application."""
    from clawagents.config.config import load_config, get_default_model
    from clawagents.providers.llm import create_provider
    from clawagents.agent import create_claw_agent
    from clawagents.process.command_queue import (
        enqueue_command_in_lane,
        get_queue_size,
    )
    from clawagents.process.lanes import CommandLane

    config = load_config()
    active_model = get_default_model(config)
    llm = create_provider(active_model, config)

    VALID_LANES = {"main", "cron", "subagent", "nested"}

    def _resolve_lane(raw: str | None) -> str:
        lane = (raw or "").strip().lower() or CommandLane.Main.value
        return lane if lane in VALID_LANES else CommandLane.Main.value

    app = FastAPI(title="ClawAgents Observatory Proxy")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "observatory-proxy",
            "provider": llm.name,
            "model": active_model,
        }

    @app.post("/chat/stream")
    async def chat_stream(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return Response(
                content=json.dumps({"error": "Invalid JSON"}),
                status_code=400,
                media_type="application/json",
            )

        task = payload.get("task", "Unknown task")
        lane = _resolve_lane(payload.get("lane"))

        event_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def sse(event: str, data: Any):
            event_queue.put_nowait(
                f"event: {event}\ndata: {json.dumps(data)}\n\n"
            )

        async def _run():
            sse("queued", {"lane": lane, "position": get_queue_size(lane)})
            try:
                result = await enqueue_command_in_lane(lane, _execute)
                sse("done", {
                    "lane": lane,
                    "status": result.status,
                    "result": result.result,
                    "iterations": result.iterations,
                })
            except Exception as e:
                sse("error", {"lane": lane, "error": str(e)})
            finally:
                event_queue.put_nowait(None)

        async def _execute():
            api_key = payload.get("api_key") or payload.get("openai_api_key")
            agent = create_claw_agent(model=llm, api_key=api_key)
            return await _invoke_with_hooks(agent, task, sse, payload)

        run_task = asyncio.create_task(_run())

        async def _stream():
            try:
                while True:
                    msg = await event_queue.get()
                    if msg is None:
                        break
                    yield msg
            finally:
                if not run_task.done():
                    run_task.cancel()

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


# ── Standalone entry point ───────────────────────────────────────────────


def start_proxy(port: int = 3002, host: str = "127.0.0.1") -> None:
    """Start the observatory proxy server."""
    import uvicorn

    app = create_observatory_app()
    print(f"\n🔭 Observatory Proxy running on http://{host}:{port}")
    print(f"   Endpoint: POST /chat/stream | GET /health\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Observatory proxy server")
    parser.add_argument("--port", type=int, default=3002)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    start_proxy(port=args.port, host=args.host)
