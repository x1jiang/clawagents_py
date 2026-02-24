import json
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

from clawagents.config.config import load_config, get_default_model
from clawagents.providers.llm import create_provider
from clawagents.process.command_queue import (
    enqueue_command_in_lane,
    get_queue_size,
    get_total_queue_size,
    get_active_task_count,
)
from clawagents.process.lanes import CommandLane
from clawagents.agent import create_claw_agent

VALID_LANES = {"main", "cron", "subagent", "nested"}


def _resolve_lane(raw: str | None) -> str:
    lane = (raw or "").strip().lower() or CommandLane.Main.value
    return lane if lane in VALID_LANES else CommandLane.Main.value


def create_app() -> tuple:
    config = load_config()
    active_model = get_default_model(config)
    llm = create_provider(active_model, config)

    app = FastAPI(title="ClawAgents Gateway")

    @app.get("/health")
    async def health():
        return {"status": "ok", "provider": llm.name, "model": active_model}

    @app.get("/queue")
    async def queue_status():
        lanes = {lane: get_queue_size(lane) for lane in VALID_LANES}
        return {
            "lanes": lanes,
            "total": get_total_queue_size(),
            "active": get_active_task_count(),
        }

    @app.post("/chat")
    async def chat(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return Response(
                content=json.dumps({"error": 'Invalid JSON. Send { "task": "...", "lane": "main|cron|subagent" }'}),
                status_code=400,
                media_type="application/json",
            )

        task = payload.get("task", "Unknown task")
        lane = _resolve_lane(payload.get("lane"))

        async def execute_graph():
            print(f"[Gateway] lane={lane} task: {task}")
            agent = create_claw_agent(model=llm)
            return await agent.invoke(task)

        try:
            result = await enqueue_command_in_lane(lane, execute_graph)
            return {
                "success": True,
                "lane": lane,
                "status": result.status,
                "result": result.result,
                "iterations": result.iterations,
            }
        except Exception as e:
            return Response(
                content=json.dumps({"success": False, "lane": lane, "error": str(e)}),
                status_code=500,
                media_type="application/json",
            )

    @app.post("/chat/stream")
    async def chat_stream(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return Response(
                content=json.dumps({"error": 'Invalid JSON. Send { "task": "...", "lane": "main|cron|subagent" }'}),
                status_code=400,
                media_type="application/json",
            )

        task = payload.get("task", "Unknown task")
        lane = _resolve_lane(payload.get("lane"))

        import asyncio

        event_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def sse(event: str, data: Any):
            event_queue.put_nowait(f"event: {event}\ndata: {json.dumps(data)}\n\n")

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
            sse("started", {"lane": lane})
            agent = create_claw_agent(model=llm)

            def on_event(ev: Any):
                sse("agent", ev if isinstance(ev, dict) else {"type": str(ev)})

            return await agent.invoke(task, on_event=on_event)

        asyncio.create_task(_run())

        async def _stream():
            while True:
                msg = await event_queue.get()
                if msg is None:
                    break
                yield msg

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app, llm, active_model


def start_gateway(port: int = 3000):
    app, llm, active_model = create_app()
    print(f"\n🦞 ClawAgents Gateway running on http://localhost:{port}")
    print(f"   Provider: {llm.name}")
    print(f"   Model: {active_model}")
    print("   Endpoints: POST /chat | POST /chat/stream | GET /queue | GET /health\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
