from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, TypeVar

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError
try:
    from google import genai
    from google.genai import types
    _HAS_GEMINI = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    _HAS_GEMINI = False

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
        gemini_parts: list[dict[str, Any]] | None = None,
        thinking: str | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id          # For role="tool": the ID this result belongs to
        self.tool_calls_meta = tool_calls_meta    # For role="assistant": list of {id, name, args}
        self.gemini_parts = gemini_parts          # Preserved Gemini response parts (thought/thought_signature)
        self.thinking = thinking                  # Feature H: preserved <think> block content


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
        gemini_parts: list[dict[str, Any]] | None = None,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.partial = partial
        self.tool_calls = tool_calls
        self.gemini_parts = gemini_parts          # Preserved Gemini response parts (thought/thought_signature)


OnChunkCallback = (
    Callable[[str], Coroutine[Any, Any, None]] | Callable[[str], None] | None
)


# ─── Feature H: Thinking Token Preservation ────────────────────────────────

import re as _re

_THINK_BLOCK_RE = _re.compile(r"<think>(.*?)</think>", _re.DOTALL)


def strip_thinking_tokens(content: str) -> tuple[str, str | None]:
    """Extract <think>...</think> blocks and return (clean_content, thinking).

    Handles models like Qwen3, DeepSeek that wrap chain-of-thought in <think> tags.
    Returns the content with thinking removed, and the thinking text separately.
    """
    if not content or "<think>" not in content:
        return content, None
    thinking_parts: list[str] = []
    for m in _THINK_BLOCK_RE.finditer(content):
        thinking_parts.append(m.group(1).strip())
    clean = _THINK_BLOCK_RE.sub("", content).strip()
    thinking = "\n".join(thinking_parts) if thinking_parts else None
    return clean, thinking


def rebuild_thinking_content(content: str, thinking: str | None) -> str:
    """Re-attach thinking tokens for models that expect them in conversation history."""
    if not thinking:
        return content
    return f"<think>{thinking}</think>\n{content}"


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


# ─── Truncated JSON Repair ─────────────────────────────────────────────────


def _repair_json(text: str) -> Any:
    """Best-effort parse of possibly-truncated JSON from an LLM tool call.

    Strategy:
      1. Try normal json.loads.
      2. Try closing open braces/brackets from the end.
      3. Fall back to empty dict.
    """
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to close unclosed braces/brackets
    closers = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in closers:
            stack.append(closers[ch])
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    repaired = text + "".join(reversed(stack))
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract a partial object up to the last complete key-value
    try:
        # Truncate to last comma or colon and close
        for i in range(len(text) - 1, 0, -1):
            if text[i] in (",", ":"):
                candidate = text[:i].rstrip(",: \t\n") + "".join(reversed(stack))
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    logger.warning("JSON repair failed for tool call arguments (input: %s) — using empty args", text[:200])
    return {}


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
    result: list[NativeToolCall] = []
    for tc in tool_calls:
        if getattr(tc, "type", None) == "function":
            fn = tc.function
            result.append(NativeToolCall(
                tool_name=fn.name,
                args=_repair_json(fn.arguments or "{}"),
                tool_call_id=getattr(tc, "id", "") or "",
            ))
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


# o-series reasoning models require temperature=1 (API restriction).
# GPT-5 models accept any temperature — do NOT include them here.
_FIXED_TEMPERATURE_MODELS: dict[str, float] = {
    "o1": 1.0,
    "o1-mini": 1.0,
    "o1-preview": 1.0,
    "o3": 1.0,
    "o3-mini": 1.0,
    "o4-mini": 1.0,
    "gpt-5-nano": 1.0,
    "gpt-5-mini": 1.0,
    "gpt-5-turbo": 1.0,
}

_NON_REASONING_MODELS: set[str] = {
    "gpt-5-micro", "gpt-4o", "gpt-4o-mini",
}


def _resolve_temperature(model: str, requested: float) -> float:
    """Return the fixed temperature if the model requires it, else the requested value."""
    if model in _NON_REASONING_MODELS:
        return requested
    for prefix, fixed in _FIXED_TEMPERATURE_MODELS.items():
        if model == prefix or model.startswith(prefix + "-"):
            return fixed
    if model == "gpt-5" or model.startswith("gpt-5-2") or model.startswith("gpt-5."):
        return 1.0
    return requested


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, config: EngineConfig):
        base_url = config.openai_base_url or None
        api_version = config.openai_api_version or None
        api_key = config.openai_api_key or ("not-needed" if base_url else "")

        api_type = (config.openai_api_type or "").lower()
        is_azure = api_type == "azure" or (api_version and base_url and "azure" in base_url.lower())
        if is_azure and api_version and base_url:
            try:
                from openai import AsyncAzureOpenAI
                self.client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=api_version,
                )
            except ImportError:
                self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = AsyncOpenAI(**client_kwargs)

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
                    native_calls = []
                    for idx in sorted(tools_accumulation.keys()):
                        fn = tools_accumulation[idx]
                        native_calls.append(NativeToolCall(
                            tool_name=fn["name"],
                            args=_repair_json(fn["arguments"] or "{}"),
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


def _serialize_gemini_parts(parts: Any) -> list[dict[str, Any]] | None:
    """Serialize Gemini Part objects to dicts, preserving thought/thought_signature."""
    if not parts:
        return None
    serialized = []
    for p in parts:
        d: dict[str, Any] = {}
        if getattr(p, "text", None) is not None:
            d["text"] = p.text
        if getattr(p, "thought", None):
            d["thought"] = True
        if getattr(p, "thought_signature", None):
            d["thought_signature"] = p.thought_signature
        fc = getattr(p, "function_call", None)
        if fc:
            d["function_call"] = {"name": fc.name, "args": dict(fc.args) if fc.args else {}}
            if getattr(p, "thought_signature", None):
                d["thought_signature"] = p.thought_signature
        if d:
            serialized.append(d)
    return serialized if serialized else None


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: EngineConfig):
        if not _HAS_GEMINI:
            raise ImportError("google-genai not installed. Install with: pip install clawagents[gemini]")
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
                if m.gemini_parts:
                    user_contents.append({"role": "model", "parts": m.gemini_parts})
                else:
                    parts = []
                    if m.content:
                        parts.append({"text": m.content})
                    for tc in m.tool_calls_meta:
                        parts.append({"function_call": {"name": tc["name"], "args": tc["args"]}})
                    user_contents.append({"role": "model", "parts": parts})
            elif m.role == "assistant" and m.gemini_parts:
                user_contents.append({"role": "model", "parts": m.gemini_parts})
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
        *,
        _malformed_retry: bool = False,
    ) -> LLMResponse:
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_contents,
            config=gemini_config,
        )
        fn_calls: list[NativeToolCall] | None = None
        raw_parts = None
        finish_reason = None
        candidates = getattr(resp, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            parts = getattr(candidates[0].content, "parts", None) if candidates[0].content else None
            if parts:
                raw_parts = _serialize_gemini_parts(parts)
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
            extracted_text = "".join(
                getattr(p, "text", "") for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            )

        fr_str = str(finish_reason) if finish_reason else ""
        if not _malformed_retry and "MALFORMED_FUNCTION_CALL" in fr_str and not fn_calls:
            logger.warning("  [gemini] MALFORMED_FUNCTION_CALL detected — retrying with mode=ANY")
            retry_opts: dict[str, Any] = {}
            for attr in ("max_output_tokens", "temperature", "system_instruction", "tools"):
                val = getattr(gemini_config, attr, None)
                if val is not None:
                    retry_opts[attr] = val
            retry_opts["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
            retry_config = types.GenerateContentConfig(**retry_opts)
            return await self._request_once(user_contents, retry_config, _malformed_retry=True)

        return LLMResponse(
            content=extracted_text,
            model=self.model,
            tokens_used=(
                resp.usage_metadata.candidates_token_count or 0
                if resp.usage_metadata
                else 0
            ),
            tool_calls=fn_calls,
            gemini_parts=raw_parts,
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
            all_stream_parts: list[Any] = []
            last_finish_reason: Any = None

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
                            gemini_parts=_serialize_gemini_parts(all_stream_parts),
                        )

                    try:
                        _chunk_text_parts: list[str] = []
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for _cand in chunk.candidates:
                                if hasattr(_cand, "content") and _cand.content and hasattr(_cand.content, "parts"):
                                    for _p in _cand.content.parts:
                                        if getattr(_p, "text", None) and not getattr(_p, "thought", False):
                                            _chunk_text_parts.append(_p.text)
                        if _chunk_text_parts:
                            _joined = "".join(_chunk_text_parts)
                            chunks.append(_joined)
                            await _invoke_callback(on_chunk, _joined)
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for candidate in chunk.candidates:
                                fr = getattr(candidate, "finish_reason", None)
                                if fr is not None:
                                    last_finish_reason = fr
                                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                                    for p in candidate.content.parts:
                                        all_stream_parts.append(p)
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

                fr_str = str(last_finish_reason) if last_finish_reason else ""
                if "MALFORMED_FUNCTION_CALL" in fr_str and not fn_calls:
                    logger.warning("  [gemini] MALFORMED_FUNCTION_CALL in stream — retrying with mode=ANY (non-stream)")
                    retry_opts: dict[str, Any] = {}
                    for attr in ("max_output_tokens", "temperature", "system_instruction", "tools"):
                        val = getattr(gemini_config, attr, None)
                        if val is not None:
                            retry_opts[attr] = val
                    retry_opts["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
                    retry_config = types.GenerateContentConfig(**retry_opts)
                    return await self._request_once(user_contents, retry_config, _malformed_retry=True)

                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    tool_calls=fn_calls if fn_calls else None,
                    gemini_parts=_serialize_gemini_parts(all_stream_parts),
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
                        gemini_parts=_serialize_gemini_parts(all_stream_parts),
                    )
                if not _is_retryable(exc):
                    break

        raise last_error  # type: ignore[misc]


# ─── Anthropic Provider ───────────────────────────────────────────────────

try:
    import anthropic as _anthropic_mod
    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic_mod = None  # type: ignore
    _HAS_ANTHROPIC = False


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: EngineConfig):
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Install with: pip install clawagents[anthropic]"
            )
        self.client = _anthropic_mod.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.model = config.anthropic_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        system_parts = []
        api_messages = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content if isinstance(m.content, str) else str(m.content))
            elif m.role == "tool" and m.tool_call_id:
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}],
                })
            elif m.role == "assistant" and m.tool_calls_meta:
                content_blocks = []
                if m.content:
                    content_blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls_meta:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
            else:
                role = "assistant" if m.role == "assistant" else "user"
                api_messages.append({"role": role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)
        if self._temperature > 0:
            kwargs["temperature"] = self._temperature
        if tools:
            kwargs["tools"] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            k: {"type": v.get("type", "string"), "description": v.get("description", "")}
                            for k, v in s.parameters.items()
                        },
                        "required": [k for k, v in s.parameters.items() if v.get("required")],
                    },
                }
                for s in tools
            ]

        if not on_chunk:
            return await _with_retry("anthropic", lambda: self._request_once(kwargs))
        return await self._stream_with_retry(kwargs, on_chunk, cancel_event)

    async def _request_once(self, kwargs: dict[str, Any]) -> LLMResponse:
        resp = await self.client.messages.create(**kwargs)
        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(NativeToolCall(
                    tool_name=block.name,
                    args=dict(block.input) if block.input else {},
                    tool_call_id=block.id,
                ))

        return LLMResponse(
            content="".join(text_parts),
            model=self.model,
            tokens_used=(resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
    ) -> LLMResponse:
        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning("  [anthropic] Retry %d/%d after %.1fs", attempt, _MAX_RETRIES, delay)
                await asyncio.sleep(delay)

            chunks: list[str] = []
            tool_calls: list[NativeToolCall] = []
            current_tool: dict[str, Any] | None = None
            final_tokens = 0

            try:
                async with self.client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if cancel_event and cancel_event.is_set():
                            return LLMResponse(
                                content="".join(chunks), model=self.model,
                                tokens_used=final_tokens, partial=True,
                            )

                        if event.type == "content_block_start":
                            if hasattr(event.content_block, "type"):
                                if event.content_block.type == "tool_use":
                                    current_tool = {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "input_json": "",
                                    }
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                chunks.append(event.delta.text)
                                await _invoke_callback(on_chunk, event.delta.text)
                            elif hasattr(event.delta, "partial_json") and current_tool:
                                current_tool["input_json"] += event.delta.partial_json
                        elif event.type == "content_block_stop":
                            if current_tool:
                                tool_calls.append(NativeToolCall(
                                    tool_name=current_tool["name"],
                                    args=_repair_json(current_tool["input_json"] or "{}"),
                                    tool_call_id=current_tool["id"],
                                ))
                                current_tool = None
                        elif event.type == "message_delta":
                            if hasattr(event.usage, "output_tokens"):
                                final_tokens = event.usage.output_tokens

                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    tokens_used=final_tokens,
                    tool_calls=tool_calls if tool_calls else None,
                )

            except Exception as exc:
                last_error = exc
                if chunks:
                    return LLMResponse(
                        content="".join(chunks), model=self.model,
                        tokens_used=final_tokens, partial=True,
                    )
                if not _is_retryable(exc):
                    break

        raise last_error  # type: ignore[misc]


# ─── Factory ──────────────────────────────────────────────────────────────


def create_provider(model_name: str, config: EngineConfig) -> LLMProvider:
    """Create a single LLM provider inferred from model name."""
    lower = model_name.lower()
    if lower.startswith("gemini"):
        if not _HAS_GEMINI:
            raise ImportError(
                "google-genai package not installed. Install with: pip install clawagents[gemini]"
            )
        config.gemini_model = model_name
        return GeminiProvider(config)
    if lower.startswith("claude") or lower.startswith("anthropic"):
        config.anthropic_model = model_name
        return AnthropicProvider(config)
    config.openai_model = model_name
    return OpenAIProvider(config)
