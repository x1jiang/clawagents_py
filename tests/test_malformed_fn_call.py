"""
Tests for the Gemini MALFORMED_FUNCTION_CALL retry logic.

When Gemini returns finish_reason=MALFORMED_FUNCTION_CALL with 0 parts,
the provider should retry with tool_config mode=ANY instead of returning
an empty response that the agent loop interprets as "done".

Run: python -m pytest tests/test_malformed_fn_call.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from clawagents.providers.llm import GeminiProvider, LLMResponse, NativeToolSchema


# ─── Helpers: mock Gemini SDK objects ─────────────────────────────────────

def _make_candidate(finish_reason="STOP", parts=None):
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    if parts is not None:
        candidate.content = MagicMock()
        candidate.content.parts = parts
    else:
        candidate.content = None
    return candidate


def _make_response(candidates=None, tokens=10):
    resp = MagicMock()
    resp.candidates = candidates
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.candidates_token_count = tokens
    return resp


def _make_part_with_function_call(name="search_db", args=None):
    p = MagicMock()
    p.text = None
    p.thought = False
    p.thought_signature = None
    p.function_call = MagicMock()
    p.function_call.name = name
    p.function_call.args = args or {"query": "test"}
    return p


def _make_text_part(text="some text"):
    p = MagicMock()
    p.text = text
    p.thought = False
    p.thought_signature = None
    p.function_call = None
    return p


def _make_config():
    cfg = MagicMock()
    cfg.geminiApiKey = "test-key"
    cfg.geminiModel = "gemini-2.5-flash"
    cfg.maxTokens = 8192
    cfg.temperature = 0
    return cfg


TOOL_SCHEMAS = [
    NativeToolSchema(
        name="search_db",
        description="Search the database",
        parameters={"query": {"type": "string", "description": "search query", "required": True}},
    ),
]


# ─── Test: _request_once retries on MALFORMED_FUNCTION_CALL ──────────────

class TestMalformedFunctionCallNonStream:

    @pytest.mark.asyncio
    async def test_retry_on_malformed_returns_valid_tool_calls(self):
        """When first call returns MALFORMED_FUNCTION_CALL, retry with mode=ANY should succeed."""
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock(
            max_output_tokens=8192, temperature=0, system_instruction=None, tools=None,
        ))

        malformed_resp = _make_response(
            candidates=[_make_candidate(finish_reason="MALFORMED_FUNCTION_CALL", parts=None)]
        )
        valid_part = _make_part_with_function_call("search_db", {"query": "patients"})
        valid_resp = _make_response(
            candidates=[_make_candidate(finish_reason="STOP", parts=[valid_part])]
        )

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[malformed_resp, valid_resp]
        )

        with patch("clawagents.providers.llm.types", mock_types), \
             patch("clawagents.providers.llm.genai", MagicMock()):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider.client = mock_client
            provider.model = "gemini-2.5-flash"
            provider._max_tokens = 8192
            provider._temperature = 0

            config = MagicMock()
            config.max_output_tokens = 8192
            config.temperature = 0
            config.system_instruction = None
            config.tools = [{"function_declarations": [{"name": "search_db"}]}]

            result = await provider._request_once([], config)

        assert mock_client.aio.models.generate_content.call_count == 2
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search_db"

    @pytest.mark.asyncio
    async def test_no_retry_when_finish_reason_is_stop(self):
        """Normal STOP finish_reason should NOT trigger retry."""
        mock_types = MagicMock()

        text_part = _make_text_part("Here is your answer")
        normal_resp = _make_response(
            candidates=[_make_candidate(finish_reason="STOP", parts=[text_part])]
        )

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=normal_resp)

        with patch("clawagents.providers.llm.types", mock_types), \
             patch("clawagents.providers.llm.genai", MagicMock()):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider.client = mock_client
            provider.model = "gemini-2.5-flash"
            provider._max_tokens = 8192
            provider._temperature = 0

            config = MagicMock()
            config.max_output_tokens = 8192

            result = await provider._request_once([], config)

        assert mock_client.aio.models.generate_content.call_count == 1
        assert result.content == "Here is your answer"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_no_infinite_recursion_on_double_malformed(self):
        """If retry with mode=ANY also returns MALFORMED, should NOT recurse again."""
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock(
            max_output_tokens=8192, temperature=0, system_instruction=None, tools=None,
        ))

        malformed_resp = _make_response(
            candidates=[_make_candidate(finish_reason="MALFORMED_FUNCTION_CALL", parts=None)]
        )

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=malformed_resp)

        with patch("clawagents.providers.llm.types", mock_types), \
             patch("clawagents.providers.llm.genai", MagicMock()):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider.client = mock_client
            provider.model = "gemini-2.5-flash"
            provider._max_tokens = 8192
            provider._temperature = 0

            config = MagicMock()
            config.max_output_tokens = 8192
            config.temperature = 0
            config.system_instruction = None
            config.tools = None

            result = await provider._request_once([], config)

        # First call (malformed) + second call (retry with mode=ANY) = 2 calls total, no infinite loop
        assert mock_client.aio.models.generate_content.call_count == 2
        assert result.content == ""
        assert result.tool_calls is None


# ─── Test: _stream_with_retry retries on MALFORMED_FUNCTION_CALL ─────────

class TestMalformedFunctionCallStream:

    @pytest.mark.asyncio
    async def test_stream_retry_on_malformed(self):
        """Streaming: MALFORMED_FUNCTION_CALL should fall back to non-stream retry with mode=ANY."""
        mock_types = MagicMock()
        mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock(
            max_output_tokens=8192, temperature=0, system_instruction=None, tools=None,
        ))

        # Streaming returns a malformed chunk
        malformed_candidate = _make_candidate(finish_reason="MALFORMED_FUNCTION_CALL", parts=None)
        malformed_chunk = MagicMock()
        malformed_chunk.text = None
        malformed_chunk.candidates = [malformed_candidate]
        malformed_chunk.usage_metadata = MagicMock()
        malformed_chunk.usage_metadata.candidates_token_count = 5

        async def mock_stream():
            yield malformed_chunk

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        # The fallback non-stream call returns valid function call
        valid_part = _make_part_with_function_call("search_db", {"query": "patients"})
        valid_resp = _make_response(
            candidates=[_make_candidate(finish_reason="STOP", parts=[valid_part])]
        )
        mock_client.aio.models.generate_content = AsyncMock(return_value=valid_resp)

        with patch("clawagents.providers.llm.types", mock_types), \
             patch("clawagents.providers.llm.genai", MagicMock()), \
             patch("clawagents.providers.llm._stall_guarded_stream", side_effect=lambda s, t: s):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider.client = mock_client
            provider.model = "gemini-2.5-flash"
            provider._max_tokens = 8192
            provider._temperature = 0

            config = MagicMock()
            config.max_output_tokens = 8192
            config.temperature = 0
            config.system_instruction = None
            config.tools = [{"function_declarations": [{"name": "search_db"}]}]

            on_chunk = MagicMock()
            result = await provider._stream_with_retry([], config, on_chunk, None)

        assert mock_client.aio.models.generate_content_stream.call_count == 1
        assert mock_client.aio.models.generate_content.call_count == 1
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search_db"

    @pytest.mark.asyncio
    async def test_stream_normal_stop_no_retry(self):
        """Streaming with STOP finish_reason should return normally without retry."""
        mock_types = MagicMock()

        text_part = _make_text_part("Stream answer")
        normal_candidate = _make_candidate(finish_reason="STOP", parts=[text_part])
        normal_chunk = MagicMock()
        normal_chunk.text = "Stream answer"
        normal_chunk.candidates = [normal_candidate]
        normal_chunk.usage_metadata = MagicMock()
        normal_chunk.usage_metadata.candidates_token_count = 12

        async def mock_stream():
            yield normal_chunk

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())
        mock_client.aio.models.generate_content = AsyncMock()

        with patch("clawagents.providers.llm.types", mock_types), \
             patch("clawagents.providers.llm.genai", MagicMock()), \
             patch("clawagents.providers.llm._stall_guarded_stream", side_effect=lambda s, t: s):
            provider = GeminiProvider.__new__(GeminiProvider)
            provider.client = mock_client
            provider.model = "gemini-2.5-flash"
            provider._max_tokens = 8192
            provider._temperature = 0

            config = MagicMock()
            on_chunk = MagicMock()
            result = await provider._stream_with_retry([], config, on_chunk, None)

        assert mock_client.aio.models.generate_content.call_count == 0
        assert result.content == "Stream answer"
