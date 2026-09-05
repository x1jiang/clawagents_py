"""SSE client for the ClawAgents sidecar gateway.

Python port of the VSCode extension's ``gatewayClient.ts``. Communicates with
the sidecar's ``POST /chat/stream`` SSE endpoint and translates the raw event
stream into typed ``HostToWebview``-equivalent dicts that the Streamlit chat
panel can consume directly.

The event types mirror the TypeScript ``HostToWebview`` union in
``clawagents-vscode/src/protocol.ts``:

    assistant_delta, assistant_message, tool_started, tool_completed,
    permission_required, ask_user_required, file_changed, usage,
    compact_progress, checkpoint, done, error, cancelled, status

Usage::

    client = SseClient(host="127.0.0.1", port=3000, token="...")
    async for event in client.stream_chat("hello", mode="auto"):
        print(event)  # {"type": "assistant_delta", "delta": "Hi"}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE parsing (equivalent to parseSseChunk in gatewayClient.ts L66-88)
# ---------------------------------------------------------------------------

def parse_sse_chunk(
    buffer: str,
) -> tuple[list[dict[str, str]], str]:
    """Parse an SSE text buffer into discrete events.

    Returns ``(events, remaining_buffer)`` where each event is
    ``{"event": ..., "data": ...}``.
    """
    events: list[dict[str, str]] = []
    parts = buffer.split("\n\n")
    rest = parts.pop()  # incomplete trailing chunk

    for part in parts:
        if not part.strip():
            continue
        event = "message"
        data_lines: list[str] = []
        for line in part.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        events.append({"event": event, "data": "\n".join(data_lines)})

    return events, rest


# ---------------------------------------------------------------------------
# Agent event mapper (equivalent to mapAgentEvent in gatewayClient.ts L100-205)
# ---------------------------------------------------------------------------

def _extract_file_path(data: dict[str, Any]) -> str | None:
    """Extract a file path from tool args or top-level data."""
    args = data.get("args") if isinstance(data.get("args"), dict) else data
    raw = (
        args.get("path")
        or args.get("file_path")
        or args.get("target_path")
        or data.get("file_path")
        or data.get("path")
    )
    return str(raw) if raw is not None else None


def _num(v: Any) -> int | float | None:
    return v if isinstance(v, (int, float)) else None


def map_agent_event(kind: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """Map a sidecar ``agent`` event into a HostToWebview-equivalent dict."""
    if kind == "assistant_delta":
        return {
            "type": "assistant_delta",
            "delta": str(data.get("delta") or data.get("text") or ""),
        }

    if kind == "assistant_message":
        return {
            "type": "assistant_message",
            "text": str(
                data.get("content") or data.get("text") or data.get("message") or ""
            ),
        }

    if kind in ("tool_started", "tool_call"):
        call_id = str(
            data.get("call_id")
            or data.get("id")
            or data.get("name")
            or ""
        )
        return {
            "type": "tool_started",
            "id": call_id,
            "name": str(data.get("name") or data.get("tool_name") or "tool"),
            "args": data.get("args") or data.get("input") or data.get("rawInput"),
            "filePath": _extract_file_path(data),
        }

    if kind in ("tool_completed", "tool_result", "tool_skipped"):
        raw_out = data.get("output") or data.get("preview") or data.get("content")
        err = data.get("error") or data.get("reason")
        out_text = (
            (str(raw_out).strip() if isinstance(raw_out, str) and raw_out.strip() else "")
            or (str(err).strip() if isinstance(err, str) and err.strip() else "")
            or ""
        )
        skipped = kind == "tool_skipped"
        return {
            "type": "tool_completed",
            "id": str(data.get("call_id") or data.get("id") or data.get("name") or ""),
            "name": str(data.get("name") or data.get("tool_name") or "tool"),
            "success": False if skipped else (data.get("success") is not False),
            "output": out_text[:8000],
            "filePath": _extract_file_path(data),
        }

    if kind == "permission_required":
        return {
            "type": "permission_required",
            "requestId": str(
                data.get("request_id") or data.get("requestId") or ""
            ),
            "tool": str(data.get("tool") or data.get("name") or "tool"),
            "filePath": str(data["file_path"]) if data.get("file_path") else None,
            "command": str(data["command"]) if data.get("command") else None,
            "reason": str(data["reason"]) if data.get("reason") else None,
        }

    if kind == "ask_user_required":
        return {
            "type": "ask_user_required",
            "requestId": str(
                data.get("request_id") or data.get("requestId") or ""
            ),
            "question": str(data.get("question") or data.get("prompt") or ""),
        }

    if kind == "usage":
        # Single mapper for both usage shapes: the agent-loop `usage` event and
        # the `on_stream_event` forward. A second `kind == "usage"` block used to
        # sit further down carrying the cache/reasoning fields, but this branch
        # returns unconditionally so that one was unreachable — prompt-cache
        # benefit never reached the dashboard. Keep every field in one place.
        return {
            "type": "usage",
            "promptTokens": _num(
                data.get("prompt_tokens") or data.get("promptTokens") or data.get("input_tokens")
            ),
            "completionTokens": _num(
                data.get("completion_tokens")
                or data.get("completionTokens")
                or data.get("output_tokens")
            ),
            "totalTokens": _num(
                data.get("total_tokens") or data.get("totalTokens") or data.get("tokens_used")
            ),
            "runCostUsd": _num(data.get("run_cost_usd") or data.get("runCostUsd")),
            "sessionCostUsd": _num(
                data.get("session_cost_usd") or data.get("sessionCostUsd")
            ),
            "lastInputTokens": _num(
                data.get("last_input_tokens")
                or data.get("prompt_tokens")
                or data.get("input_tokens")
            ),
            "cachedInputTokens": _num(
                data.get("cached_input_tokens") or data.get("cachedInputTokens")
            ),
            "cacheCreationTokens": _num(
                data.get("cache_creation_tokens") or data.get("cacheCreationTokens")
            ),
            "reasoningTokens": _num(
                data.get("reasoning_tokens") or data.get("reasoningTokens")
            ),
            "model": data.get("model"),
        }

    if kind == "compact_progress":
        return {
            "type": "compact_progress",
            "phase": str(data.get("phase") or ""),
            "message": str(data["message"]) if data.get("message") else None,
        }

    if kind == "llm_context":
        # Full LLM context snapshot from RunHooks.on_llm_start
        return {
            "type": "llm_context",
            "turn": data.get("turn"),
            "model": data.get("model", ""),
            "messages": data.get("messages", []),
            "system_prompt_breakdown": data.get("system_prompt_breakdown", {}),
            "total_input_tokens": _num(data.get("total_input_tokens")),
            "tokens_by_role": data.get("tokens_by_role", {}),
        }

    if kind == "observatory":
        return {"type": "observatory_event", "event": data}

    if kind == "context":
        return {"type": "status", "message": str(data.get("message") or "context")}

    if kind == "checkpoint":
        return {
            "type": "checkpoint",
            "sha": str(data.get("sha") or ""),
            "tool": str(data["tool"]) if data.get("tool") else None,
            "phase": str(data["phase"]) if data.get("phase") else None,
            "label": str(data["label"]) if data.get("label") else None,
            "messageCount": _num(
                data.get("message_count") or data.get("messageCount")
            ),
        }

    if kind == "approval_required":
        return {
            "type": "permission_required",
            "requestId": str(
                data.get("call_id")
                or data.get("id")
                or data.get("request_id")
                or ""
            ),
            "tool": str(
                data.get("tool_name")
                or data.get("name")
                or data.get("tool")
                or "tool"
            ),
            "reason": "Library require_approval",
        }

    if kind == "warn":
        return {
            "type": "status",
            "message": f"⚠ {data.get('message') or 'warning'}",
        }

    if kind == "error":
        return {
            "type": "error",
            "message": str(data.get("message") or data.get("error") or "Unknown error"),
        }

    # Unknown kind
    return None


# ---------------------------------------------------------------------------
# SseClient
# ---------------------------------------------------------------------------


@dataclass
class SseClient:
    """Async HTTP client for the ClawAgents sidecar gateway.

    Mirrors the ``GatewayClient`` class from the VSCode extension.
    """

    host: str = "127.0.0.1"
    port: int = 3000
    token: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        h: dict[str, str] = {"Accept": accept}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    # -- JSON helpers -------------------------------------------------------

    async def _request_json(
        self,
        method: str,
        path: str,
        body: Any = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            kwargs: dict[str, Any] = {"headers": self._headers()}
            if body is not None:
                kwargs["json"] = body
            resp = await client.request(
                method, f"{self.base_url}{path}", **kwargs
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"{method} {path} HTTP {resp.status_code}: {resp.text}"
                )
            text = resp.text
            return json.loads(text) if text.strip() else {}

    # -- Health / settings --------------------------------------------------

    async def fetch_health(self) -> dict[str, Any] | None:
        try:
            return await self._request_json("GET", "/health")
        except Exception:
            return None

    async def get_settings(self) -> dict[str, Any]:
        return await self._request_json("GET", "/settings")

    # -- Chat management ----------------------------------------------------

    async def list_chats(self, query: str | None = None) -> list[dict[str, Any]]:
        path = f"/chats?q={query}" if query else "/chats"
        result = await self._request_json("GET", path)
        return result if isinstance(result, list) else []

    async def create_chat(self, mode: str = "auto") -> dict[str, Any]:
        return await self._request_json("POST", "/chats", {"mode": mode})

    async def get_chat(self, chat_id: str) -> dict[str, Any]:
        return await self._request_json("GET", f"/chats/{chat_id}")

    async def delete_chat(self, chat_id: str) -> dict[str, Any]:
        return await self._request_json("DELETE", f"/chats/{chat_id}")

    # -- Permission / ask-user resolution -----------------------------------

    async def resolve_permission(
        self, request_id: str, decision: str
    ) -> None:
        """Resolve a permission request (allow_once / allow_always / deny)."""
        await self._request_json(
            "POST", f"/permissions/{request_id}", {"decision": decision}
        )

    async def resolve_ask_user(
        self,
        request_id: str,
        *,
        answer: str | None = None,
        skip: bool = False,
    ) -> None:
        await self._request_json(
            "POST",
            f"/ask_user/{request_id}",
            {"answer": answer, "skip": skip},
        )

    # -- Cancel -------------------------------------------------------------

    async def cancel(self) -> None:
        try:
            await self._request_json("POST", "/cancel")
        except Exception:
            pass

    # -- Streaming chat (POST /chat/stream) ---------------------------------

    async def stream_chat(
        self,
        task: str,
        *,
        chat_id: str | None = None,
        mode: str = "auto",
        model: str | None = None,
        reasoning_effort: str | None = None,
        interaction: str = "interactive",
        api_key: str | None = None,
        enable_context_observatory: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat events via SSE.

        Yields ``HostToWebview``-equivalent dicts. The caller should iterate::

            async for event in client.stream_chat("hello"):
                handle(event)
        """
        resolved_api_key = api_key
        if resolved_api_key is None and self.token and self.token.startswith("sk-"):
            resolved_api_key = self.token

        body = {
            "task": task,
            "chat_id": chat_id,
            "session_id": chat_id,
            "lane": "main",
            "mode": mode,
            "model": model or None,
            "reasoning_effort": reasoning_effort or None,
            "interaction": interaction,
            "api_key": resolved_api_key,
            "enable_context_observatory": enable_context_observatory,
        }

        headers = self._headers(accept="text/event-stream")
        headers["Content-Type"] = "application/json"

        # No total timeout for streaming; 60s read timeout for keepalives.
        timeout = httpx.Timeout(timeout=None, read=60.0)

        resolved_chat_id: str | None = chat_id
        saw_terminal = False

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/stream",
                json=body,
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    yield {"type": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}
                    return

                buffer = ""
                async for chunk in resp.aiter_bytes():
                    buffer += chunk.decode("utf-8", errors="replace")
                    events, buffer = parse_sse_chunk(buffer)

                    for ev in events:
                        data: dict[str, Any] = {}
                        try:
                            data = json.loads(ev["data"]) if ev["data"] else {}
                        except json.JSONDecodeError:
                            data = {"raw": ev["data"]}

                        # Track chat_id
                        if data.get("chat_id"):
                            resolved_chat_id = str(data["chat_id"])

                        mapped: dict[str, Any] | None = None

                        if ev["event"] == "agent":
                            kind = str(data.get("kind", ""))
                            payload = (
                                data["data"]
                                if isinstance(data.get("data"), dict)
                                else data
                            )
                            mapped = map_agent_event(kind, payload)

                        elif ev["event"] == "permission_required":
                            mapped = {
                                "type": "permission_required",
                                "requestId": str(data.get("request_id", "")),
                                "tool": str(data.get("tool", "tool")),
                                "filePath": str(data["file_path"]) if data.get("file_path") else None,
                                "command": str(data["command"]) if data.get("command") else None,
                                "reason": str(data["reason"]) if data.get("reason") else None,
                            }

                        elif ev["event"] == "ask_user_required":
                            mapped = {
                                "type": "ask_user_required",
                                "requestId": str(data.get("request_id", "")),
                                "question": str(data.get("question", "")),
                            }

                        elif ev["event"] == "file_changed":
                            mapped = {
                                "type": "file_changed",
                                "path": str(data.get("path", "")),
                                "snapshotId": str(data["snapshot_id"]) if data.get("snapshot_id") else None,
                                "snapshotRel": str(data["snapshot_rel"]) if data.get("snapshot_rel") else None,
                            }

                        elif ev["event"] == "usage":
                            mapped = {
                                "type": "usage",
                                "promptTokens": _num(data.get("prompt_tokens") or data.get("input_tokens")),
                                "completionTokens": _num(data.get("completion_tokens") or data.get("output_tokens")),
                                "totalTokens": _num(data.get("total_tokens")),
                                "lastInputTokens": _num(
                                    data.get("last_input_tokens")
                                    or data.get("prompt_tokens")
                                    or data.get("input_tokens")
                                ),
                                "runCostUsd": _num(data.get("run_cost_usd")),
                                "sessionCostUsd": _num(data.get("session_cost_usd")),
                            }

                        elif ev["event"] == "observatory":
                            mapped = {"type": "observatory_event", "event": data}

                        elif ev["event"] == "done":
                            usage_obj = (
                                data["usage"]
                                if isinstance(data.get("usage"), dict)
                                else {}
                            )
                            mapped = {
                                "type": "done",
                                "status": str(data.get("status", "done")),
                                "result": str(data["result"]) if data.get("result") is not None else None,
                                "iterations": data.get("iterations") if isinstance(data.get("iterations"), int) else None,
                                "usage": data.get("usage"),
                                "runCostUsd": _num(usage_obj.get("run_cost_usd")),
                                "sessionCostUsd": _num(usage_obj.get("session_cost_usd")),
                                "chatId": resolved_chat_id,
                            }

                        elif ev["event"] == "error":
                            mapped = {
                                "type": "error",
                                "message": str(
                                    data.get("error")
                                    or data.get("message")
                                    or "Stream error"
                                ),
                            }

                        elif ev["event"] in ("started", "queued"):
                            mapped = {
                                "type": "status",
                                "message": "Queued…" if ev["event"] == "queued" else "Running…",
                            }

                        if mapped is not None:
                            if mapped["type"] in ("done", "error", "cancelled"):
                                saw_terminal = True
                            yield mapped

                # Stream ended — ensure we emit a terminal event
                if not saw_terminal:
                    yield {
                        "type": "error",
                        "message": (
                            "Stream ended without completion — "
                            "the sidecar may have crashed or restarted."
                        ),
                    }
