"""LLM providers — public entry point.

The implementation lives in sibling modules, one per provider family:

* ``base`` — Provider-neutral message/response types, the thinking-stream filter, and the LLMProvider interface.
* ``stream_retry`` — Retry, circuit-breaker and stall-guard plumbing shared by every streaming provider.
* ``tool_convert`` — Truncated-JSON repair and conversion of native tool schemas to each provider's wire format.
* ``openai_provider`` — OpenAI and OpenAI-compatible provider (Chat Completions and Responses), with the GPT request-contract helpers.
* ``gemini_provider`` — Google Gemini provider.
* ``anthropic_provider`` — Anthropic provider (direct API).
* ``bedrock_provider`` — Amazon Bedrock providers: Claude on Bedrock, Claude on Mantle, and the Converse API.
* ``factory`` — Provider selection: endpoint/model classification helpers and ``create_provider``.

Everything is re-exported here, so ``from clawagents.providers.llm import X``
keeps working for every public and private name. Assigning an attribute on
this module (``monkeypatch.setattr(llm, "genai", fake)``) also updates it in
the implementation modules — see ``_PropagatingModule`` at the bottom.
"""
# ruff: noqa: F401  (this module exists to re-export)
from __future__ import annotations

import sys as _sys
import types as _pytypes

import asyncio
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import logging
import os
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
from clawagents.errors.taxonomy import MantleCredentialsError
logger = logging.getLogger("clawagents.providers.llm")
T = TypeVar("T")
# ─── Feature H: Thinking Token Preservation ────────────────────────────────

import re as _re
# ─── Anthropic Provider ───────────────────────────────────────────────────

try:
    import anthropic as _anthropic_mod
    _HAS_ANTHROPIC = True
except ImportError:
    _anthropic_mod = None  # type: ignore
    _HAS_ANTHROPIC = False

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

from clawagents.providers import base as _base
from clawagents.providers import stream_retry as _stream_retry
from clawagents.providers import tool_convert as _tool_convert
from clawagents.providers import openai_provider as _openai_provider
from clawagents.providers import gemini_provider as _gemini_provider
from clawagents.providers import anthropic_provider as _anthropic_provider
from clawagents.providers import bedrock_provider as _bedrock_provider
from clawagents.providers import factory as _factory
from clawagents.providers.base import (  # noqa: F401
    LLMMessage,
    NativeToolSchema,
    NativeToolCall,
    LLMResponse,
    OnChunkCallback,
    _THINK_BLOCK_RE,
    _THINK_OPEN,
    _THINK_CLOSE,
    strip_thinking_tokens,
    _ThinkStreamFilter,
    rebuild_thinking_content,
    LLMProvider,
)
from clawagents.providers.stream_retry import (  # noqa: F401
    _MAX_RETRIES,
    _INITIAL_DELAY_S,
    _MAX_DELAY_S,
    _CHUNK_STALL_S,
    _RETRYABLE_STATUS_CODES,
    _STATUS_IN_TEXT_RE,
    _is_retryable,
    _jittered_delay,
    _stall_guarded_stream,
    _invoke_callback,
    _import_http_client_module,
    _StreamReplayGuard,
    _with_retry,
    _get_stream_breaker,
    _admit_stream_breaker,
    _record_stream_breaker,
)
from clawagents.providers.tool_convert import (  # noqa: F401
    _repair_json,
    _to_openai_tools,
    _to_gemini_tools,
    _hashed_session_key,
    _openai_affinity,
    _fire_first_token,
    _usage_detail_int,
    _openai_reasoning_tokens,
    _openai_reasoning_text,
    _openai_cache_tokens,
    _openai_cached_tokens,
    _tool_args_text,
    _args_json_clean,
    _coerce_args_dict,
    _synth_tool_call_id,
    _parse_openai_tool_calls,
)
from clawagents.providers.openai_provider import (  # noqa: F401
    _FIXED_TEMPERATURE_MODELS,
    _NON_REASONING_MODELS,
    _resolve_temperature,
    _bare_openai_model_id,
    openai_model_rejects_temperature,
    _with_temperature,
    anthropic_model_rejects_sampling_params,
    _GPT6_MODELS,
    _GPT6_RESPONSES_ONLY_TOOLS,
    _HOSTED_MUSE_PREFIXES,
    is_hosted_muse_model,
    _requires_responses_for_tools,
    _responses_only_model_label,
    _chat_completions_needs_reasoning_none,
    _REASONING_EFFORT_VALUES,
    normalize_reasoning_effort,
    _GROK_EFFORT_LEVELS,
    clamp_reasoning_effort_for_model,
    model_supports_reasoning_effort,
    _normalize_wire_api,
    supports_openai_fast_mode,
    _responses_endpoint_likely,
    prefers_responses_api,
    _is_responses_unsupported,
    _apply_tool_reasoning_compat,
    _apply_responses_reasoning,
    _chat_tools_to_responses_tools,
    _content_to_responses_parts,
    _messages_to_responses_input,
    _parse_responses_result,
    _parse_responses_extras,
    _sanitize_openai_tool_pairs,
    _openai_chat_messages,
    _ResponsesDeferredMixin,
    OpenAIProvider,
)
from clawagents.providers.gemini_provider import (  # noqa: F401
    _serialize_gemini_parts,
    _restore_gemini_parts_for_api,
    _stamp_function_call_ids,
    _part_has_function_call,
    _part_has_function_response,
    _parts_have_function_call,
    _parts_have_function_response,
    _model_parts_from_tool_meta,
    _GEMINI_FR_KEYS,
    _GEMINI_FC_KEYS,
    _GEMINI_PART_KEYS,
    _is_gemini_sdk_shape_error,
    _scrub_gemini_function_response,
    _scrub_gemini_function_call,
    _scrub_gemini_part,
    _scrub_gemini_contents,
    _sanitize_gemini_contents,
    _is_gemini_history_400,
    GEMINI_ANSWER_NUDGE,
    GEMINI_SUMMARIZE_MARKER,
    GEMINI_EVIDENCE_MARKER,
    _GEMINI_DUMP_LINE_RE,
    looks_like_gemini_command_dump,
    _gemini_function_response_body,
    _stamp_function_response_ids,
    _flatten_gemini_fc_text,
    _flatten_gemini_fr_text,
    _upsert_gemini_stream_function_call,
    _absorb_gemini_stream_parts,
    _flatten_gemini_tool_history,
    _gemini_part_from_block,
    _gemini_38_model,
    _GEMINI_BLOCK_FINISH_REASONS,
    _gemini_finish_reason_name,
    _gemini_prompt_block_reason,
    _gemini_4_fallback_model,
    _gemini_function_retry_config,
    _gemini_thought_tokens,
    GeminiProvider,
)
from clawagents.providers.anthropic_provider import (  # noqa: F401
    _serialize_anthropic_blocks,
    _anthropic_message_content,
    _apply_conversation_cache_breakpoints,
    _DeferredToolsMixin,
    _ALWAYS_ADAPTIVE_ANTHROPIC_MODELS,
    _current_anthropic_model,
    _apply_current_anthropic_effort,
    AnthropicProvider,
)
from clawagents.providers.bedrock_provider import (  # noqa: F401
    _GATEWAY_KEY_PLACEHOLDERS,
    _mantle_gateway_key,
    _aws_region,
    _bedrock_client_kwargs,
    _is_bedrock_claude_model,
    BedrockProvider,
    MantleAnthropicProvider,
    _converse_content_blocks,
    _converse_doc_name,
    BedrockConverseProvider,
    _mantle_origin,
    mantle_anthropic_base_url,
    _mantle_region_from_url,
)
from clawagents.providers.factory import (  # noqa: F401
    _OLLAMA_PREFIXES,
    _OLLAMA_DEFAULT_BASE_URL,
    _looks_like_ollama,
    _is_mantle_url,
    mantle_openai_base_url,
    is_mantle_anthropic_model,
    is_mantle_openai_responses_model,
    is_mantle_xai_model,
    needs_mantle_openai_base,
    _mantle_openai_model_id,
    create_provider,
    has_active_anthropic_tool_turn,
)


_IMPLEMENTATION_MODULES = (
    _base,
    _stream_retry,
    _tool_convert,
    _openai_provider,
    _gemini_provider,
    _anthropic_provider,
    _bedrock_provider,
    _factory,
)


class _PropagatingModule(_pytypes.ModuleType):
    """Module whose attribute assignments reach the implementation modules.

    The code that used to live in this file now reads its globals from the
    module it moved to. Tests (and the VS Code sidecar's legacy Gemini shim)
    replace names on *this* module, e.g. ``llm.genai`` or
    ``llm._get_stream_breaker``. Without propagation those replacements would
    change only the re-export here and the real code would keep using the
    original object. Setting a name here therefore also sets it in every
    implementation module that has that name.
    """

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _IMPLEMENTATION_MODULES:
            if name in module.__dict__:
                setattr(module, name, value)


_sys.modules[__name__].__class__ = _PropagatingModule
