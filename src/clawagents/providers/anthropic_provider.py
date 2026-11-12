"""Anthropic provider (direct API).

Part of the ``clawagents.providers.llm`` implementation; import from that module.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
from typing import Any
from clawagents.config.config import EngineConfig
logger = logging.getLogger("clawagents.providers.llm")
# ─── Feature H: Thinking Token Preservation ────────────────────────────────

import re as _re
# ─── Anthropic Provider ───────────────────────────────────────────────────

try:
    import anthropic as _anthropic_mod
    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic_mod = None  # type: ignore
    _HAS_ANTHROPIC = False
from clawagents.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    NativeToolCall,
    NativeToolSchema,
    OnChunkCallback,
)
from clawagents.providers.stream_retry import (
    _MAX_RETRIES,
    _StreamReplayGuard,
    _admit_stream_breaker,
    _get_stream_breaker,
    _invoke_callback,
    _is_retryable,
    _jittered_delay,
    _record_stream_breaker,
    _with_retry,
)
from clawagents.providers.tool_convert import (
    _fire_first_token,
    _repair_json,
)
from clawagents.providers.openai_provider import (
    anthropic_model_rejects_sampling_params,
    normalize_reasoning_effort,
)

def _serialize_anthropic_blocks(blocks: Any) -> list[dict[str, Any]] | None:
    """Preserve response blocks verbatim, including empty signed/redacted thought."""
    output = []
    for block in blocks or []:
        if isinstance(block, dict):
            output.append(deepcopy(block))
            continue
        dump = getattr(block, "model_dump", None)
        data = dump(mode="json", exclude_none=True) if callable(dump) else None
        if isinstance(data, dict):
            output.append(data)
            continue
        # Small protocol-compatible providers/test doubles without SDK models.
        kind = getattr(block, "type", None)
        fields = {"text": ("text",), "thinking": ("thinking", "signature"),
                  "redacted_thinking": ("data",), "tool_use": ("id", "name", "input")}.get(kind if isinstance(kind, str) else "")
        if fields:
            data = {"type": kind}
            for key in fields:
                value = getattr(block, key, None)
                if isinstance(value, (str, dict)):
                    data[key] = deepcopy(value)
            output.append(data)
    return output or None


def _anthropic_message_content(content: Any) -> Any:
    """Shape a user/assistant message's content for Anthropic's Messages API.

    Anthropic has no ``image_url`` content type — convert the canonical
    OpenAI-style image blocks (used for user-message images) into
    ``image``/``source`` blocks. Without this an attached image reaches Claude
    as an invalid block and the request 400s. Non-list content and non-image
    blocks pass through unchanged; malformed image_url parts are dropped.
    """
    if not isinstance(content, list):
        return content
    if not any(
        isinstance(p, dict) and p.get("type") in ("image_url", "file") for p in content
    ):
        return content
    from clawagents.media.documents import file_part_to_anthropic_block
    from clawagents.media.images import image_url_to_anthropic_block

    converted: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            block = image_url_to_anthropic_block(part)
            if block is not None:
                converted.append(block)
        elif isinstance(part, dict) and part.get("type") == "file":
            block = file_part_to_anthropic_block(part)
            if block is not None:
                converted.append(block)
        else:
            converted.append(part)
    return converted


def _apply_conversation_cache_breakpoints(api_messages: list[dict[str, Any]]) -> None:
    """Mark the stable conversation prefix for Anthropic ephemeral prompt caching.

    Places a ``cache_control`` breakpoint on the last content block immediately
    before the final substantive user turn (typically the volatile tail).
    """
    if len(api_messages) < 2:
        return

    last_user_idx: int | None = None
    for i in range(len(api_messages) - 1, -1, -1):
        msg = api_messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            last_user_idx = i
            break
        if isinstance(content, list):
            has_text = any(
                isinstance(block, dict)
                and block.get("type") == "text"
                and str(block.get("text", "")).strip()
                for block in content
            )
            if has_text:
                last_user_idx = i
                break

    if last_user_idx is None or last_user_idx <= 0:
        return

    boundary_idx = last_user_idx - 1
    msg = api_messages[boundary_idx]
    content = msg.get("content")
    marker = {"cache_control": {"type": "ephemeral"}}

    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, **marker}]
    elif isinstance(content, list) and content:
        blocks = [dict(block) if isinstance(block, dict) else block for block in content]
        last = blocks[-1]
        if isinstance(last, dict):
            last = dict(last)
            last["cache_control"] = {"type": "ephemeral"}
            blocks[-1] = last
        msg["content"] = blocks


class _DeferredToolsMixin:
    """Opt-in cache-preserving tool activation with a self-healing fallback.

    The ``defer_loading`` / ``tool_reference`` wire shape cannot be verified
    without a live API call, so a rejection must never cost the user a turn:
    the first time the provider rejects it, deferral is switched off for this
    provider instance and the request is retried with the ordinary full tool
    list. That makes enabling the feature safe to try in production.
    """

    _deferred_disabled: bool = False

    def _deferred_tools_enabled(self) -> bool:
        if self._deferred_disabled:
            return False
        try:
            from clawagents.config.features import is_enabled

            if not is_enabled("deferred_tool_loading"):
                return False
        except Exception:
            return False
        from clawagents.providers.deferred_tools import model_supports_tool_references

        return model_supports_tool_references(getattr(self, "model", ""))

    def _disable_deferred_tools(self, exc: BaseException) -> bool:
        """Turn deferral off after a shape rejection. True when caller retries."""
        from clawagents.providers.deferred_tools import is_deferred_tool_rejection

        if self._deferred_disabled or not is_deferred_tool_rejection(exc):
            return False
        self._deferred_disabled = True
        logger.warning(
            "  [anthropic] deferred tool loading rejected (%s) — falling back to "
            "the full tool list for the rest of this session",
            type(exc).__name__,
        )
        return True


_ALWAYS_ADAPTIVE_ANTHROPIC_MODELS = frozenset({
    "fable-5-1", "fable-5", "opus-5-5", "sonnet-5-5",
})


def _current_anthropic_model(model: str) -> str | None:
    # Accept direct, Mantle and regional Bedrock IDs, including dated snapshots.
    normalized = _re.sub(r"((?:opus|sonnet|haiku|fable)-\d+)\.(\d+)", r"\1-\2", (model or "").strip().lower())
    match = _re.search(
        r"(?:^|[./])claude-(fable-5-1|fable-5|opus-5-5|opus-5|sonnet-5-5|sonnet-5|haiku-5-5|haiku-5)"
        r"(?:$|-(?:\d{8}(?:-v\d+)?|v\d+)(?::\d+)?$)",
        normalized,
    )
    return match.group(1) if match else None


def _apply_current_anthropic_effort(kwargs: dict[str, Any], model: str, effort: str | None) -> None:
    """Current Claude 5 models default to adaptive thinking; older models do not."""
    current = _current_anthropic_model(model)
    if not current or not effort:
        return
    if effort == "none" and current not in _ALWAYS_ADAPTIVE_ANTHROPIC_MODELS:
        kwargs["thinking"] = {"type": "disabled"}
    # Fable 5.x and Opus/Sonnet 5.5 thinking is always adaptive; never send
    # their rejected disabled mode. low also satisfies Opus 5 / Haiku 5.5's
    # effort ceiling when thinking is disabled.
    effective = "low" if effort in ("none", "minimal") else effort
    kwargs.setdefault("output_config", {})["effort"] = effective


class AnthropicProvider(_DeferredToolsMixin, LLMProvider):
    name = "anthropic"

    def __init__(self, config: EngineConfig):
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. Install with: pip install clawagents[anthropic]"
            )
        client_kwargs: dict[str, Any] = {"api_key": config.anthropic_api_key}
        base = (getattr(config, "anthropic_base_url", None) or "").strip()
        if base:
            # SDK appends /v1/messages — Mantle expects …/anthropic/v1/messages.
            client_kwargs["base_url"] = base.rstrip("/")
        self.client = _anthropic_mod.AsyncAnthropic(**client_kwargs)
        self.model = config.anthropic_model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._reasoning_effort = normalize_reasoning_effort(config.reasoning_effort)

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        try:
            return await self._chat_once(
                messages,
                on_chunk,
                cancel_event,
                tools,
                session_id=session_id,
                on_first_token=on_first_token,
            )
        except Exception as exc:
            # Unverifiable wire shape must never cost a turn: drop deferral for
            # this provider and replay the request with the full tool list.
            if not self._disable_deferred_tools(exc):
                raise
            return await self._chat_once(
                messages,
                on_chunk,
                cancel_event,
                tools,
                session_id=session_id,
                on_first_token=on_first_token,
            )

    async def _chat_once(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
        *,
        session_id: str | None = None,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        _ = session_id
        system_parts = []
        api_messages: list[dict[str, Any]] = []

        # Cache-preserving tool activation: tools introduced mid-run are sent
        # as deferred schemas + a transcript-level reference, leaving the
        # cached tool-list prefix byte-identical. Disabled by default and
        # auto-disabled for this provider after a rejection (see chat()).
        deferred_schemas: dict[str, Any] = {}
        if tools and self._deferred_tools_enabled():
            from clawagents.providers.deferred_tools import split_deferred_tools

            immediate, deferred_schemas = split_deferred_tools(messages, tools)
            if deferred_schemas:
                tools = immediate + list(deferred_schemas.values())
        deferred_names = set(deferred_schemas)
        referenced: set[str] = set()

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content if isinstance(m.content, str) else str(m.content))
            elif m.role == "tool" and m.tool_call_id:
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                extra_blocks: list[dict[str, Any]] = []
                if deferred_names:
                    from clawagents.providers.deferred_tools import tool_reference_blocks

                    refs = tool_reference_blocks(m, deferred_names, referenced)
                    if refs:
                        # Anthropic rejects tool_reference mixed with ordinary
                        # tool-result content, so the real output moves to a
                        # sibling text block in the same user message.
                        block["content"] = refs
                        if m.content:
                            extra_blocks.append(
                                {
                                    "type": "text",
                                    "text": m.content
                                    if isinstance(m.content, str)
                                    else json.dumps(m.content),
                                }
                            )
                # Coalesce consecutive tool results into ONE user message —
                # Anthropic requires every tool_result block answering a single
                # assistant turn (parallel tool calls) to be in the same user
                # message, otherwise the API rejects the transcript.
                prev = api_messages[-1] if api_messages else None
                if (
                    prev is not None
                    and prev.get("role") == "user"
                    and isinstance(prev.get("content"), list)
                    and prev["content"]
                    and isinstance(prev["content"][0], dict)
                    and prev["content"][0].get("type") == "tool_result"
                ):
                    prev["content"].append(block)
                    prev["content"].extend(extra_blocks)
                else:
                    api_messages.append({"role": "user", "content": [block, *extra_blocks]})
            elif m.role == "assistant" and getattr(m, "anthropic_blocks", None):
                # Signed thinking is an opaque transcript block, not display
                # text. Never rebuild/filter it or duplicate its tool_use list.
                api_messages.append({"role": "assistant", "content": deepcopy(m.anthropic_blocks)})
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
                api_messages.append(
                    {"role": role, "content": _anthropic_message_content(m.content)}
                )

        try:
            from clawagents.config.features import is_enabled as _feat_cache

            if _feat_cache("cache_boundary"):
                _apply_conversation_cache_breakpoints(api_messages)
        except Exception:
            pass

        from clawagents.graph.model_profiles import resolve_model_profile

        profile = resolve_model_profile(self.model) or {}
        output_limit = min(
            self._max_tokens, int(profile.get("max_output_tokens", self._max_tokens))
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": output_limit,
            "messages": api_messages,
        }
        if system_parts:
            joined = "\n".join(system_parts)
            # Feature: Cache boundary optimization
            # Split on __CACHE_BOUNDARY__ marker to create static (cached) + dynamic blocks
            if "__CACHE_BOUNDARY__" in joined:
                static_part, dynamic_part = joined.split("__CACHE_BOUNDARY__", 1)
                system_blocks = [
                    {"type": "text", "text": static_part.strip(), "cache_control": {"type": "ephemeral"}},
                ]
                if dynamic_part.strip():
                    system_blocks.append({"type": "text", "text": dynamic_part.strip()})
                kwargs["system"] = system_blocks
            else:
                kwargs["system"] = joined
        # Send temperature whenever it's set (including 0). Gating on ``> 0``
        # dropped ``temperature=0`` — the config default — so Anthropic silently
        # sampled at the API default of 1.0 while OpenAI/Gemini honoured 0,
        # making "temperature: 0" runs non-deterministic only on Claude.
        # Opus 4.7+ reject the field entirely (400) — omit it for those models.
        if (
            not anthropic_model_rejects_sampling_params(self.model)
            and self._temperature is not None
            and self._temperature >= 0
        ):
            kwargs["temperature"] = self._temperature
        schema = getattr(self, "_structured_json_schema", None)
        if isinstance(schema, dict) and schema and not tools:
            # Anthropic structured output suppresses tools — only apply when
            # this turn is tool-free (matches Grok shell StructuredOutput tool path).
            try:
                from clawagents.structured_output import anthropic_output_format

                kwargs["output_config"] = {"format": anthropic_output_format(schema)}
            except Exception:
                pass
        _apply_current_anthropic_effort(
            kwargs, self.model, getattr(self, "_reasoning_effort", None)
        )
        if _current_anthropic_model(self.model) == "fable-5-1":
            # Documented recovery for edited/compacted prefixes. Send original
            # blocks; the server alone decides which bindings are invalid.
            kwargs["thinking"] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
            kwargs["extra_headers"] = {"anthropic-beta": "thinking-binding-controls-2026-08-01"}
        if tools:
            from clawagents.providers.tool_schema import emit_openai_schema_node

            kwargs["tools"] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            k: emit_openai_schema_node(v)
                            for k, v in s.parameters.items()
                            if isinstance(v, dict)
                        },
                        "required": [
                            k for k, v in s.parameters.items() if v.get("required")
                        ],
                    },
                    # Deferred tools stay out of the cached prefix; the model
                    # learns about them from the transcript tool_reference.
                    **({"defer_loading": True} if s.name in deferred_names else {}),
                }
                for s in tools
            ]

        if not on_chunk:
            from clawagents.circuit_breaker import breaker_key as _bk

            return await _with_retry(
                "anthropic",
                lambda: self._request_once(kwargs),
                policy=getattr(self, "retry_policy", None),
                breaker_tag=_bk("anthropic", model=self.model),
            )
        return await self._stream_with_retry(
            kwargs,
            on_chunk,
            cancel_event,
            on_first_token=on_first_token,
        )

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

        usage = resp.usage
        uncached = getattr(usage, "input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        prompt_total = uncached + cache_creation + cache_read
        return LLMResponse(
            content="".join(text_parts),
            model=self.model,
            tokens_used=(prompt_total + usage.output_tokens) if usage else 0,
            tool_calls=tool_calls if tool_calls else None,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            prompt_tokens=prompt_total,
            anthropic_blocks=_serialize_anthropic_blocks(resp.content),
            finish_reason=getattr(resp, "stop_reason", None),
        )

    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
        on_chunk: OnChunkCallback,
        cancel_event: asyncio.Event | None,
        *,
        on_first_token: Any | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None
        breaker = _get_stream_breaker(
            "anthropic",
            model=getattr(self, "model", None) or kwargs.get("model"),
        )

        # De-duplicate text across retry attempts (see _StreamReplayGuard).
        replay_guard = _StreamReplayGuard(on_chunk)
        if on_chunk is not None:
            on_chunk = replay_guard.feed

        attempt = 0
        while attempt <= _MAX_RETRIES:
            if attempt > 0:
                delay = _jittered_delay(attempt - 1)
                logger.warning("  [anthropic] Retry %d/%d after %.1fs", attempt, _MAX_RETRIES, delay)
                await asyncio.sleep(delay)

            try:
                await _admit_stream_breaker(breaker)
            except Exception as exc:
                last_error = exc
                try:
                    from clawagents.circuit_breaker import BreakerOpen as _BO

                    if isinstance(exc, _BO):
                        await asyncio.sleep(max(0.05, float(exc.retry_after) or 0.05))
                        continue
                except Exception:
                    pass
                raise

            replay_guard.begin_attempt()
            chunks: list[str] = []
            tool_calls: list[NativeToolCall] = []
            current_tool: dict[str, Any] | None = None
            output_tokens = 0
            cache_creation = 0
            cache_read = 0
            prompt_tokens = 0
            first_token_fired = False
            preserved: dict[int, dict[str, Any]] = {}
            complete_blocks: set[int] = set()
            finish_reason: str | None = None

            def completed_content() -> list[dict[str, Any]] | None:
                return [deepcopy(preserved[i]) for i in sorted(complete_blocks) if i in preserved] or None

            try:
                async with self.client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if cancel_event and cancel_event.is_set():
                            return LLMResponse(
                                content="".join(chunks), model=self.model,
                                tokens_used=prompt_tokens + cache_creation + cache_read + output_tokens,
                                prompt_tokens=prompt_tokens + cache_creation + cache_read,
                                partial=True,
                                tool_calls=tool_calls if tool_calls else None,
                                cache_creation_tokens=cache_creation,
                                cache_read_tokens=cache_read,
                                anthropic_blocks=completed_content(),
                                finish_reason=finish_reason,
                            )

                        index = getattr(event, "index", 0)
                        if event.type == "content_block_start":
                            blocks = _serialize_anthropic_blocks([event.content_block])
                            if blocks:
                                preserved[index] = blocks[0]
                        elif event.type == "content_block_delta" and index in preserved:
                            block = preserved[index]
                            delta = event.delta
                            for key in ("text", "thinking", "signature"):
                                value = getattr(delta, key, None)
                                if isinstance(value, str):
                                    block[key] = str(block.get(key, "")) + value
                        elif event.type == "content_block_stop":
                            if index in preserved and preserved[index].get("type") == "tool_use" and current_tool and current_tool["input_json"]:
                                preserved[index]["input"] = _repair_json(current_tool["input_json"] or "{}")
                            complete_blocks.add(index)

                        if event.type == "message_start" and hasattr(event, "message"):
                            u = getattr(event.message, "usage", None)
                            if u:
                                prompt_tokens = getattr(u, "input_tokens", 0) or 0
                                cache_creation = getattr(u, "cache_creation_input_tokens", 0) or 0
                                cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
                        elif event.type == "content_block_start":
                            if hasattr(event.content_block, "type"):
                                if event.content_block.type == "tool_use":
                                    if not first_token_fired:
                                        first_token_fired = True
                                        _fire_first_token(on_first_token)
                                    current_tool = {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "input_json": "",
                                        "input": getattr(event.content_block, "input", {}) or {},
                                    }
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                chunks.append(event.delta.text)
                                await _invoke_callback(on_chunk, event.delta.text)
                            elif hasattr(event.delta, "partial_json") and current_tool:
                                if not first_token_fired:
                                    first_token_fired = True
                                    _fire_first_token(on_first_token)
                                current_tool["input_json"] += event.delta.partial_json
                        elif event.type == "content_block_stop":
                            if current_tool:
                                tool_calls.append(NativeToolCall(
                                    tool_name=current_tool["name"],
                                    args=_repair_json(current_tool["input_json"]) if current_tool["input_json"] else current_tool["input"],
                                    tool_call_id=current_tool["id"],
                                ))
                                current_tool = None
                        elif event.type == "message_delta":
                            stop = getattr(getattr(event, "delta", None), "stop_reason", None)
                            if isinstance(stop, str):
                                finish_reason = stop
                            if hasattr(event.usage, "output_tokens"):
                                output_tokens = event.usage.output_tokens

                _record_stream_breaker(breaker, success=True)
                return LLMResponse(
                    content="".join(chunks),
                    model=self.model,
                    # input+output, matching the non-streaming path (it used to
                    # report output-only, understating usage by the prompt size).
                    tokens_used=prompt_tokens + cache_creation + cache_read + output_tokens,
                    tool_calls=tool_calls if tool_calls else None,
                    cache_creation_tokens=cache_creation,
                    cache_read_tokens=cache_read,
                    prompt_tokens=prompt_tokens + cache_creation + cache_read,
                    anthropic_blocks=completed_content(),
                    finish_reason=finish_reason,
                )

            except Exception as exc:
                last_error = exc
                # Retry retryable mid-stream failures before surfacing a
                # truncated partial; include accumulated tool calls when we do.
                if _is_retryable(exc):
                    _record_stream_breaker(breaker, success=False)
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "  [anthropic] Stream interrupted after %d chars — retrying",
                        len("".join(chunks)),
                    )
                    attempt += 1
                    continue
                if chunks or tool_calls:
                    return LLMResponse(
                        content="".join(chunks), model=self.model,
                        tokens_used=prompt_tokens + cache_creation + cache_read + output_tokens,
                        prompt_tokens=prompt_tokens + cache_creation + cache_read,
                        partial=True,
                        tool_calls=tool_calls if tool_calls else None,
                        cache_creation_tokens=cache_creation,
                        cache_read_tokens=cache_read,
                        anthropic_blocks=completed_content(),
                        finish_reason=finish_reason,
                    )
                break

        raise last_error  # type: ignore[misc]
