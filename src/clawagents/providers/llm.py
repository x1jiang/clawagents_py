from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, TypeVar

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError
from google import genai
from google.genai import types

from clawagents.config.config import EngineConfig

logger = logging.getLogger(__name__)

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

T = TypeVar("T")

# ─── Public Types ──────────────────────────────────────────────────────────


class LLMMessage:
    def __init__(
        self,
        role: Literal["system", "user", "assistant", "tool"],
        content: str | list[dict[str, Any]],
        tool_call_id: str | None = None,
        tool_calls_meta: list[dict[str, Any]] | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id          # For role="tool": the ID this result belongs to
        self.tool_calls_meta = tool_calls_meta    # For role="assistant": list of {id, name, args}


class NativeToolSchema:
    """Schema for a tool that can be passed to LLM native function calling."""
    __slots__ = ("name", "description", "parameters")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict[str, Any]],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters


class NativeToolCall:
    """A structured tool call returned by the LLM's native function calling."""
    __slots__ = ("tool_name", "args", "tool_call_id")

    def __init__(self, tool_name: str, args: dict[str, Any], tool_call_id: str = ""):
        self.tool_name = tool_name
        self.args = args
        self.tool_call_id = tool_call_id


class LLMResponse:
    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int,
        partial: bool = False,
        tool_calls: list[NativeToolCall] | None = None,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.partial = partial
        self.tool_calls = tool_calls


OnChunkCallback = (
    Callable[[str], Coroutine[Any, Any, None]] | Callable[[str], None] | None
)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        pass


# ─── Streaming Robustness Internals ───────────────────────────────────────

_MAX_RETRIES = 3
_INITIAL_DELAY_S = 1.0
_MAX_DELAY_S = 16.0
_CHUNK_STALL_S = 60.0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_retryable(err: BaseException) -> bool:
    if isinstance(err, APIStatusError):
        return err.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(err, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(err, Exception):
        msg = str(err).lower()
        return any(
            tok in msg
            for tok in (
                "econnreset", "network", "timeout", "stream stalled",
                "rate limit", "too many requests", "service unavailable",
                "429", "500", "502", "503", "504",
            )
        )
    return False


def _jittered_delay(attempt: int) -> float:
    base = _INITIAL_DELAY_S * (2 ** attempt)
    return min(base + random.random() * base * 0.1, _MAX_DELAY_S)


async def _stall_guarded_stream(
    aiter: AsyncIterator[T],
    timeout_s: float,
) -> AsyncIterator[T]:
    """Yield items from *aiter*, raising TimeoutError if no item arrives
    within *timeout_s* seconds (stall detection)."""
    ait = aiter.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(ait.__anext__(), timeout=timeout_s)
            yield chunk
        except StopAsyncIteration:
            return


async def _invoke_callback(
    cb: OnChunkCallback,
    text: str,
) -> None:
    """Call *cb* with *text*, isolating errors so a broken callback
    never kills the stream."""
    if cb is None:
        return
    try:
        if asyncio.iscoroutinefunction(cb):
            await cb(text)
        else:
            cb(text)
    except Exception:
        logger.debug("onChunk callback raised — isolated", exc_info=True)


async def _with_retry(
    tag: str,
    fn: Callable[[], Coroutine[Any, Any, T]],
) -> T:
    last_error: BaseException | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _jittered_delay(attempt - 1)
            logger.warning(
                "  [%s] Retry %d/%d after %.1fs", tag, attempt, _MAX_RETRIES, delay,
            )
            await asyncio.sleep(delay)
        try:
            return await fn()
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc):
                break
    raise last_error  # type: ignore[misc]


# ─── Native Tool Schema Converters ────────────────────────────────────────


def _to_openai_tools(schemas: list[NativeToolSchema]) -> list[dict[str, Any]]:
    """Convert NativeToolSchema list → OpenAI Chat Completions `tools` param."""
    result = []
    for s in schemas:
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for k, v in s.parameters.items():
            properties[k] = {"type": v.get("type", "string"), "description": v.get("description", "")}
            if "items" in v:
                properties[k]["items"] = v["items"]
            if v.get("required"):
                required.append(k)
        fn_def: dict[str, Any] = {
            "name": s.name,
            "description": s.description,
            "parameters": {"type": "object", "properties": properties},
        }
        if required:
            fn_def["parameters"]["required"] = required
        result.append({"type": "function", "function": fn_def})
    return result


def _to_gemini_tools(schemas: list[NativeToolSchema]) -> list[dict[str, Any]]:
    """Convert NativeToolSchema list → Gemini FunctionDeclaration format."""
    declarations = []
    for s in schemas:
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for k, v in s.parameters.items():
            properties[k] = {"type": v.get("type", "string").upper(), "description": v.get("description", "")}
            if "items" in v:
                properties[k]["items"] = {"type": v["items"].get("type", "string").upper()}
            if v.get("required"):
                required.append(k)
        decl: dict[str, Any] = {
            "name": s.name,
            "description": s.description,
            "parameters": {"type": "OBJECT", "properties": properties},
        }
        if required:
            decl["parameters"]["required"] = required
        declarations.append(decl)
    return [{"function_declarations": declarations}]


def _parse_openai_tool_calls(
    tool_calls: Any,
) -> list[NativeToolCall] | None:
    """Extract NativeToolCall list from OpenAI response tool_calls (handles function vs custom union)."""
    if not tool_calls:
        return None
    import json as _json
    result: list[NativeToolCall] = []
    for tc in tool_calls:
        if getattr(tc, "type", None) == "function":
            fn = tc.function
            result.append(NativeToolCall(
                tool_name=fn.name,
                args=_json.loads(fn.arguments or "{}"),
                tool_call_id=getattr(tc, "id", "") or "",
            ))
        # Skip custom tool calls — we only generate function tools
    return result if result else None


# ─── OpenAI Provider ──────────────────────────────────────────────────────
#
# Uses the Chat Completions API (chat.completions.create). Supports native
# function calling via the `tools` parameter for models like GPT-4o, GPT-5,
# GPT-5-nano, GPT-5.1, and GPT-5.2 (non-Codex).
#
# NOTE: GPT-5.2-Codex and similar models use the Responses API
# (client.responses.create) which has a different tool-calling interface.
# Those would need a separate ResponsesAPIProvider.


# Models that only accept a specific temperature value
_FIXED_TEMPERATURE_MODELS: dict[str, float] = {
    "gpt-5-nano": 1.0,
    "gpt-5-mini": 1.0,
    "gpt-5": 1.0,
    "gpt-5.1": 1.0,
    "gpt-5.2": 1.0,
    "o1": 1.0,
    "o1-mini": 1.0,
    "o1-preview": 1.0,
    "o3": 1.0,
    "o3-mini": 1.0,
    "o4-mini": 1.0,
}


def _resolve_temperature(model: str, requested: float) -> float:
    """Return the fixed temperature if the model requires it, else the requested value."""
    for prefix, fixed in _FIXED_TEMPERATURE_MODELS.items():
        if model == prefix or model.startswith(prefix + "-"):
            return fixed
    return requested


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, config: EngineConfig):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.openai_model
        self._max_tokens = config.max_tokens
        self._temperature = _resolve_temperature(config.openai_model, config.temperature)

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        formatted = []
        for m in messages:
            if m.role == "tool" and m.tool_call_id:
                formatted.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls_meta:
                formatted.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                        for tc in m.tool_calls_meta
                    ],
                })
            else:
                formatted.append({"role": m.role, "content": m.content})
        oai_tools = _to_openai_tools(tools) if tools else None

        if not on_chunk:
            return await _with_retry("openai", lambda: self._request_once(formatted, oai_tools))
        return await self._stream_with_retry(formatted, on_chunk, cancel_event, oai_tools)

    async def _request_once(
        self, messages: list[dict[str, str]],
        oai_tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
        resp = await self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        native_calls = _parse_openai_tool_calls(getattr(msg, "tool_calls", None))
        return LLMResponse(
            content=msg.content or "",
            model=self.model,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
            tool_calls=native_calls,
        )

    async def _stream_with_retry(
        self,
        messages: list[dict[str, str]],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        oai_tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [openai] Stream retry %d/%d after %.1fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            chunks: list[str] = []
            final_tokens = 0
            tools_accumulation: dict[int, dict[str, Any]] = {}

            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": self._max_tokens,
                    "temperature": self._temperature,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if oai_tools:
                    kwargs["tools"] = oai_tools
                stream = await self.client.chat.completions.create(**kwargs)

                async for chunk in _stall_guarded_stream(stream, _CHUNK_STALL_S):
                    if cancel_event and cancel_event.is_set():
                        await stream.close()
                        return LLMResponse(
                            content="".join(chunks),
                            model=self.model,
                            tokens_used=final_tokens,
                            partial=True,
                        )

                    try:
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                text = delta.content
                                chunks.append(text)
                                await _invoke_callback(on_chunk, text)
                            
                            if getattr(delta, "tool_calls", None):
                                for tc in delta.tool_calls:
                                    idx = tc.index
                                    if idx not in tools_accumulation:
                                        tools_accumulation[idx] = {"id": "", "name": "", "arguments": ""}
                                    if getattr(tc, "id", None):
                                        tools_accumulation[idx]["id"] = tc.id
                                    if getattr(tc, "function", None):
                                        if tc.function.name:
                                            tools_accumulation[idx]["name"] += tc.function.name
                                        if tc.function.arguments:
                                            tools_accumulation[idx]["arguments"] += tc.function.arguments

                        if chunk.usage:
                            final_tokens = chunk.usage.total_tokens
                    except Exception:
                        pass  # malformed chunk — skip

                native_calls = None
                if tools_accumulation:
                    import json as _json
                    native_calls = []
                    for idx in sorted(tools_accumulation.keys()):
                        fn = tools_accumulation[idx]
                        native_calls.append(NativeToolCall(
                            tool_name=fn["name"],
                            args=_json.loads(fn["arguments"] or "{}"),
                            tool_call_id=fn.get("id", ""),
                        ))

                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    tool_calls=native_calls,
                )

            except Exception as exc:
                last_error = exc
                if chunks:
                    partial = "".join(chunks)
                    logger.warning(
                        "  [openai] Stream interrupted after %d chars — returning partial",
                        len(partial),
                    )
                    return LLMResponse(
                        content=partial,
                        model=self.model,
                        tokens_used=final_tokens,
                        partial=True,
                    )
                if not _is_retryable(exc):
                    break

        raise last_error  # type: ignore[misc]


# ─── Gemini Provider ──────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: EngineConfig):
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.gemini_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        # Build a toolCallId → toolName lookup from all assistant messages
        tc_id_to_name: dict[str, str] = {}
        for m in messages:
            if m.role == "assistant" and m.tool_calls_meta:
                for tc in m.tool_calls_meta:
                    tc_id_to_name[tc["id"]] = tc["name"]

        system_parts = []
        user_contents = []

        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                elif isinstance(m.content, list):
                    system_parts.extend([p.get("text", "") for p in m.content if p.get("type") == "text"])
            elif m.role == "tool" and m.tool_call_id:
                tool_name = tc_id_to_name.get(m.tool_call_id, "unknown")
                user_contents.append({"role": "user", "parts": [{"function_response": {
                    "name": tool_name,
                    "response": {"result": m.content},
                }}]})
            elif m.role == "assistant" and m.tool_calls_meta:
                # Convert tool_calls_meta to functionCall parts
                parts = []
                if m.content:
                    parts.append({"text": m.content})
                for tc in m.tool_calls_meta:
                    parts.append({"function_call": {"name": tc["name"], "args": tc["args"]}})
                user_contents.append({"role": "model", "parts": parts})
            else:
                role_name = "model" if m.role == "assistant" else "user"
                if isinstance(m.content, str):
                    user_contents.append({"role": role_name, "parts": [{"text": m.content}]})
                elif isinstance(m.content, list):
                    parts = []
                    for part in m.content:
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            import base64
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                mime_b64 = url[5:]
                                mime, b64_str = mime_b64.split(";base64,")
                                parts.append({"inline_data": {"mime_type": mime, "data": base64.b64decode(b64_str)}})
                    user_contents.append({"role": role_name, "parts": parts})

        system_instruction = "\n".join(system_parts)

        config_opts: dict[str, Any] = {
            "max_output_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if system_instruction:
            config_opts["system_instruction"] = system_instruction
        if tools:
            config_opts["tools"] = _to_gemini_tools(tools)
        gemini_config = types.GenerateContentConfig(**config_opts)

        if not on_chunk:
            return await _with_retry("gemini", lambda: self._request_once(
                user_contents, gemini_config,
            ))
        return await self._stream_with_retry(
            user_contents, gemini_config, on_chunk, cancel_event,
        )

    async def _request_once(
        self,
        user_contents: list[dict[str, Any]],
        gemini_config: types.GenerateContentConfig,
    ) -> LLMResponse:
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_contents,
            config=gemini_config,
        )
        # Extract native function calls from Gemini response parts
        fn_calls: list[NativeToolCall] | None = None
        candidates = getattr(resp, "candidates", None)
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) if candidates[0].content else None
            if parts:
                fn_calls = []
                for p in parts:
                    fc = getattr(p, "function_call", None)
                    if fc:
                        import uuid
                        fn_calls.append(NativeToolCall(
                            tool_name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                            tool_call_id=f"gemini_{uuid.uuid4().hex[:8]}",
                        ))
                if not fn_calls:
                    fn_calls = None
        extracted_text = ""
        if candidates and parts:
            extracted_text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
            
        return LLMResponse(
            content=extracted_text,
            model=self.model,
            tokens_used=(
                resp.usage_metadata.candidates_token_count or 0
                if resp.usage_metadata
                else 0
            ),
            tool_calls=fn_calls,
        )

    async def _stream_with_retry(
        self,
        user_contents: list[dict[str, Any]],
        gemini_config: types.GenerateContentConfig,
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
    ) -> LLMResponse:
        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning(
                    "  [gemini] Stream retry %d/%d after %.1fs",
                    attempt, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            chunks: list[str] = []
            final_tokens = 0
            fn_calls: list[NativeToolCall] = []

            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=user_contents,
                    config=gemini_config,
                )

                async for chunk in _stall_guarded_stream(stream, _CHUNK_STALL_S):
                    if cancel_event and cancel_event.is_set():
                        return LLMResponse(
                            content="".join(chunks),
                            model=self.model,
                            tokens_used=final_tokens,
                            partial=True,
                            tool_calls=fn_calls if fn_calls else None,
                        )

                    try:
                        # Extract text content
                        if hasattr(chunk, "text") and chunk.text:
                            chunks.append(chunk.text)
                            await _invoke_callback(on_chunk, chunk.text)
                        # Extract function calls from streaming parts
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for candidate in chunk.candidates:
                                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                                    for p in candidate.content.parts:
                                        fc = getattr(p, "function_call", None)
                                        if fc:
                                            import uuid
                                            fn_calls.append(NativeToolCall(
                                                tool_name=fc.name,
                                                args=dict(fc.args) if fc.args else {},
                                                tool_call_id=f"gemini_{uuid.uuid4().hex[:8]}",
                                            ))
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            final_tokens = chunk.usage_metadata.candidates_token_count or 0
                    except Exception:
                        pass  # malformed chunk — skip

                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    tool_calls=fn_calls if fn_calls else None,
                )

            except Exception as exc:
                last_error = exc
                if chunks:
                    partial = "".join(chunks)
                    logger.warning(
                        "  [gemini] Stream interrupted after %d chars — returning partial",
                        len(partial),
                    )
                    return LLMResponse(
                        content=partial,
                        model=self.model,
                        tokens_used=final_tokens,
                        partial=True,
                    )
                if not _is_retryable(exc):
                    break

        raise last_error  # type: ignore[misc]


# ─── Fallback Provider ────────────────────────────────────────────────────


# ─── Factory ──────────────────────────────────────────────────────────────


def create_provider(model_name: str, config: EngineConfig) -> LLMProvider:
    """
    Create a single LLM provider. The provider is inferred from the model name:
    names starting with "gemini" → GeminiProvider, everything else → OpenAIProvider.
    """
    if model_name.lower().startswith("gemini"):
        config.gemini_model = model_name
        return GeminiProvider(config)
    config.openai_model = model_name
    return OpenAIProvider(config)
