"""Current-model requests retain supported thinking options through fallback."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from clawagents.config.config import EngineConfig
from clawagents.providers import llm


def gemini_response(*, malformed=False, thoughts=7):
    part = SimpleNamespace(text="answer", thought=False, function_call=None, thought_signature=None)
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="MALFORMED_FUNCTION_CALL" if malformed else "STOP",
                                    content=None if malformed else SimpleNamespace(parts=[part]))],
        usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=3,
                                       cached_content_token_count=2, thoughts_token_count=thoughts),
    )


@pytest.fixture
def gemini_client(monkeypatch):
    pytest.importorskip("google.genai")
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=gemini_response())
    monkeypatch.setattr(llm.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(llm, "_get_stream_breaker", lambda *args, **kwargs: None)
    return client


def gemini_provider(model="gemini-3.8-flash", effort="", max_tokens=100000):
    return llm.GeminiProvider(EngineConfig(gemini_api_key="test", gemini_model=model,
                                         reasoning_effort=effort, temperature=0.2, max_tokens=max_tokens))


@pytest.mark.parametrize("effort,expected", [("", None), ("none", "low"), ("minimal", "low"),
                                             ("low", "low"), ("medium", "medium"),
                                             ("high", "high"), ("xhigh", "high"),
                                             ("max", "high"), ("EXTRA HIGH", "high")])
async def test_gemini38_request_sampling_output_limit_and_effort(gemini_client, effort, expected):
    provider = gemini_provider(effort=effort)
    result = await provider.chat([llm.LLMMessage("user", "hello")])
    config = gemini_client.aio.models.generate_content.call_args.kwargs["config"]
    wire = config.model_dump(exclude_none=True)
    assert not {"temperature", "top_p", "top_k", "candidate_count"} & wire.keys()
    assert wire["max_output_tokens"] == 65536
    if expected is None:
        assert "thinking_config" not in wire
    else:
        assert str(wire["thinking_config"]["thinking_level"]).lower().split(".")[-1] == expected
        assert "thinking_budget" not in wire["thinking_config"]
    assert result.tokens_used == 21 and result.reasoning_tokens == 7


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3.7-flash", "gemini-3.80-flash"])
async def test_older_gemini_requests_keep_existing_configuration(gemini_client, model):
    await gemini_provider(model=model, effort="high").chat([llm.LLMMessage("user", "hello")])
    wire = gemini_client.aio.models.generate_content.call_args.kwargs["config"].model_dump(exclude_none=True)
    assert wire["temperature"] == 0.2
    assert wire["max_output_tokens"] == 100000
    assert "thinking_config" not in wire


@pytest.mark.parametrize("streaming", [False, True])
async def test_gemini38_malformed_fallback_preserves_thinking_and_request_fields(gemini_client, streaming):
    provider = gemini_provider(effort="xhigh", max_tokens=2048)
    tools = [llm.NativeToolSchema("check", "check", {"path": {"type": "string", "required": True}})]
    if streaming:
        async def stream():
            yield gemini_response(malformed=True)
        gemini_client.aio.models.generate_content_stream = AsyncMock(return_value=stream())
        gemini_client.aio.models.generate_content = AsyncMock(return_value=gemini_response())
    else:
        gemini_client.aio.models.generate_content = AsyncMock(side_effect=[gemini_response(malformed=True), gemini_response()])
    result = await provider.chat([llm.LLMMessage("system", "system rule"), llm.LLMMessage("user", "hello")],
                                 tools=tools, on_chunk=MagicMock() if streaming else None)
    calls = gemini_client.aio.models.generate_content.call_args_list
    initial = (gemini_client.aio.models.generate_content_stream.call_args.kwargs if streaming else calls[0].kwargs)["config"]
    retry = calls[-1].kwargs["config"]
    assert initial.tool_config is None  # fallback must not mutate original
    assert retry.thinking_config == initial.thinking_config
    assert retry.thinking_config is not None
    assert retry.temperature is None
    assert retry.max_output_tokens == 2048 and retry.system_instruction == "system rule"
    assert retry.tools == initial.tools
    assert str(retry.tool_config.function_calling_config.mode).endswith("ANY")
    assert result.content == "answer" and result.reasoning_tokens == 7


async def test_gemini38_stream_preserves_level_and_counts_thought_tokens(gemini_client):
    async def stream():
        yield gemini_response()
    gemini_client.aio.models.generate_content_stream = AsyncMock(return_value=stream())
    callback = MagicMock()
    result = await gemini_provider(effort="medium").chat([llm.LLMMessage("user", "hello")], on_chunk=callback)
    config = gemini_client.aio.models.generate_content_stream.call_args.kwargs["config"]
    assert config.temperature is None
    assert config.thinking_config.thinking_level.value.lower() == "medium"
    assert result.tokens_used == 21 and result.reasoning_tokens == 7
    callback.assert_called_once_with("answer")


async def test_gemini_structured_output_survives_malformed_fallback(gemini_client):
    provider = gemini_provider(effort="low")
    provider._structured_json_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    gemini_client.aio.models.generate_content = AsyncMock(side_effect=[gemini_response(malformed=True), gemini_response()])
    await provider.chat([llm.LLMMessage("user", "hello")])
    initial, retried = [call.kwargs["config"] for call in gemini_client.aio.models.generate_content.call_args_list]
    assert retried.response_mime_type == initial.response_mime_type == "application/json"
    assert retried.response_schema == initial.response_schema


def anthropic_response():
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="answer")],
                           usage=SimpleNamespace(input_tokens=11, output_tokens=3,
                                                 cache_creation_input_tokens=0, cache_read_input_tokens=0))


@pytest.fixture
def anthropic_client(monkeypatch):
    pytest.importorskip("anthropic")
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=anthropic_response())
    monkeypatch.setattr(llm._anthropic_mod, "AsyncAnthropic", lambda **kwargs: client)
    monkeypatch.setattr(llm._anthropic_mod, "AsyncAnthropicBedrock", lambda **kwargs: client)
    monkeypatch.setattr(llm._anthropic_mod, "AsyncAnthropicBedrockMantle", lambda **kwargs: client, raising=False)
    return client


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5", "claude-sonnet-5"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_claude_current_models_send_effort_without_sampling(anthropic_client, model, effort):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model=model,
                                                 reasoning_effort=effort, temperature=0))
    await provider.chat([llm.LLMMessage("user", "hello")])
    wire = anthropic_client.messages.create.call_args.kwargs
    assert wire["output_config"]["effort"] == effort
    assert not {"temperature", "top_p", "top_k"} & wire.keys()
    assert wire.get("thinking", {}).get("type", "adaptive") == "adaptive"


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-sonnet-4-5"])
async def test_unset_claude_effort_preserves_model_default(anthropic_client, model):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model=model))
    await provider.chat([llm.LLMMessage("user", "hello")])
    wire = anthropic_client.messages.create.call_args.kwargs
    assert "output_config" not in wire
    assert ("thinking" in wire) == (model == "claude-fable-5-1")


@pytest.mark.parametrize("model,thinking", [("claude-fable-5-1", "adaptive"), ("claude-opus-5", "disabled"), ("claude-sonnet-5", "disabled")])
async def test_none_effort_respects_claude_model_thinking_constraints(anthropic_client, model, thinking):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model=model, reasoning_effort="none"))
    await provider.chat([llm.LLMMessage("user", "hello")])
    wire = anthropic_client.messages.create.call_args.kwargs
    assert wire["output_config"]["effort"] == "low"
    assert wire.get("thinking", {}).get("type", "adaptive") == thinking


async def test_claude_effort_merges_with_structured_output(anthropic_client):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-opus-5", reasoning_effort="max"))
    provider._structured_json_schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
    await provider.chat([llm.LLMMessage("user", "hello")])
    config = anthropic_client.messages.create.call_args.kwargs["output_config"]
    assert config["effort"] == "max" and config["format"]["type"] == "json_schema"


@pytest.mark.parametrize("kind", ["mantle", "bedrock"])
async def test_claude_gateway_constructors_preserve_effort(anthropic_client, monkeypatch, kind):
    monkeypatch.setattr(llm, "_mantle_gateway_key", lambda config: "test")
    model = "anthropic.claude-sonnet-5"
    config = EngineConfig(anthropic_model=model, bedrock_model=model, reasoning_effort="xhigh",
                          openai_base_url="https://bedrock-mantle.us-east-1.api.aws")
    provider = (llm.MantleAnthropicProvider if kind == "mantle" else llm.BedrockProvider)(config)
    await provider.chat([llm.LLMMessage("user", "hello")])
    assert anthropic_client.messages.create.call_args.kwargs["output_config"]["effort"] == "xhigh"


async def test_older_claude_effort_request_is_unchanged(anthropic_client):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-sonnet-4-5", reasoning_effort="max", temperature=0.2))
    await provider.chat([llm.LLMMessage("user", "hello")])
    wire = anthropic_client.messages.create.call_args.kwargs
    assert wire["temperature"] == 0.2 and "output_config" not in wire and "thinking" not in wire


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5", "claude-sonnet-5"])
async def test_current_claude_caps_output_to_model_profile(anthropic_client, model):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model=model, max_tokens=200000))
    await provider.chat([llm.LLMMessage("user", "hello")])
    assert anthropic_client.messages.create.call_args.kwargs["max_tokens"] == 128000


async def test_current_claude_stream_sends_effort_and_output_cap(anthropic_client, monkeypatch):
    from contextlib import asynccontextmanager
    seen = {}
    @asynccontextmanager
    async def stream(**kwargs):
        seen.update(kwargs)
        async def events():
            yield SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="answer"))
            yield SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=3))
        yield events()
    monkeypatch.setattr(llm, "_get_stream_breaker", lambda *args, **kwargs: None)
    anthropic_client.messages.stream = stream
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-opus-5", max_tokens=200000, reasoning_effort="xhigh"))
    result = await provider.chat([llm.LLMMessage("user", "hello")], on_chunk=MagicMock())
    assert seen["output_config"]["effort"] == "xhigh"
    assert seen["max_tokens"] == 128000 and "temperature" not in seen
    assert result.content == "answer"


async def test_current_claude_deferred_fallback_keeps_effort(anthropic_client, monkeypatch):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-fable-5-1", reasoning_effort="max"))
    anthropic_client.messages.create = AsyncMock(side_effect=[ValueError("deferred shape"), anthropic_response()])
    monkeypatch.setattr(provider, "_disable_deferred_tools", lambda exc: True)
    result = await provider.chat([llm.LLMMessage("user", "hello")])
    requests = [call.kwargs for call in anthropic_client.messages.create.call_args_list]
    assert len(requests) == 2 and all(row["output_config"]["effort"] == "max" for row in requests)
    assert all(row["thinking"]["type"] == "adaptive" and "temperature" not in row for row in requests)
    assert result.content == "answer"


@pytest.mark.parametrize("model,expected", [("us.amazon.nova-2-lite-v1:0", 65000), ("us.meta.llama4-maverick-17b-instruct-v1:0", 8000)])
async def test_native_bedrock_request_respects_model_output_cap(monkeypatch, model, expected):
    boto3 = pytest.importorskip("boto3")
    client = MagicMock()
    client.converse.return_value = {"output": {"message": {"content": [{"text": "answer"}]}}, "usage": {"inputTokens": 11, "outputTokens": 3}}
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: SimpleNamespace(client=lambda *args, **kwargs: client))
    provider = llm.BedrockConverseProvider(EngineConfig(bedrock_model=model, max_tokens=200000))
    result = await provider.chat([llm.LLMMessage("user", "hello")])
    assert client.converse.call_args.kwargs["inferenceConfig"]["maxTokens"] == expected
    assert result.content == "answer"


@pytest.mark.parametrize("model", ["anthropic.claude-opus-5", "anthropic.claude-sonnet-5"])
@pytest.mark.parametrize("region", ["us-west-2", "us-east-2"])
def test_mantle_current_claude_rejects_unavailable_regions_without_rerouting(anthropic_client, monkeypatch, model, region):
    monkeypatch.setattr(llm, "_mantle_gateway_key", lambda config: "test")
    endpoint = f"https://bedrock-mantle.{region}.api.aws"
    config = EngineConfig(anthropic_model=model, openai_base_url=endpoint)
    with pytest.raises(ValueError, match="supported Mantle region") as error:
        llm.MantleAnthropicProvider(config)
    assert region in str(error.value) and "us-east-1" in str(error.value)
    assert config.openai_base_url == endpoint
    anthropic_client.messages.create.assert_not_called()


@pytest.mark.parametrize("region", ["us-east-1", "eu-north-1", "eu-west-1", "ap-southeast-4", "us-gov-west-1"])
def test_mantle_current_claude_accepts_documented_regions(anthropic_client, monkeypatch, region):
    monkeypatch.setattr(llm, "_mantle_gateway_key", lambda config: "test")
    config = EngineConfig(anthropic_model="anthropic.claude-opus-5", openai_base_url=f"https://bedrock-mantle.{region}.api.aws")
    assert llm.MantleAnthropicProvider(config).client is anthropic_client


def test_native_claude_region_is_not_mantle_restricted(anthropic_client):
    config = EngineConfig(bedrock_model="us.anthropic.claude-sonnet-5-v1:0", aws_region="us-west-2", reasoning_effort="high")
    assert llm.BedrockProvider(config).client is anthropic_client


@pytest.mark.parametrize("model", ["xai.grok-4.6", "grok-4.6"])
@pytest.mark.parametrize("region", ["us-east-1", "us-east-2", "eu-west-1"])
def test_mantle_grok46_rejects_other_regions_without_rerouting(monkeypatch, model, region):
    monkeypatch.setenv("BEDROCK_API_KEY", "test")
    endpoint = f"https://bedrock-mantle.{region}.api.aws/v1"
    config = EngineConfig(openai_model=model, openai_base_url=endpoint)
    with pytest.raises(ValueError, match="Grok 4.6.*us-west-2"):
        llm.create_provider(model, config)
    assert config.openai_base_url == endpoint


@pytest.mark.parametrize("effort,expected", [("none", "low"), ("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"), ("max", "xhigh")])
async def test_mantle_grok46_routes_responses_and_keeps_supported_effort(monkeypatch, effort, expected):
    monkeypatch.setattr(llm, "_mantle_gateway_key", lambda config: "test")
    monkeypatch.setenv("BEDROCK_API_KEY", "test")
    config = EngineConfig(openai_model="xai.grok-4.6", reasoning_effort=effort,
                          openai_base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
                          openai_wire_api="chat_completions")
    provider = llm.create_provider("xai.grok-4.6", config)
    assert provider._base_url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
    assert llm.model_supports_reasoning_effort(provider.model)
    seen = {}
    async def responses(messages, on_chunk, cancel_event, oai_tools, **kwargs):
        seen.update(provider._responses_kwargs(messages, oai_tools))
        return llm.LLMResponse(content="answer", model=provider.model, tokens_used=0)
    monkeypatch.setattr(provider, "_stream_with_retry_responses", responses)
    await provider.chat([llm.LLMMessage("user", "hello")], tools=[llm.NativeToolSchema("read_file", "Read", {"path": {"type": "string"}})])
    assert seen["model"] == "xai.grok-4.6" and seen["reasoning"] == {"effort": expected}
    assert seen["tools"][0]["name"] == "read_file" and "temperature" not in seen


def test_bare_grok46_uses_mantle_catalog_id_only_at_mantle(monkeypatch):
    monkeypatch.setattr(llm, "_mantle_gateway_key", lambda config: "test")
    monkeypatch.setenv("BEDROCK_API_KEY", "test")
    monkeypatch.setenv("XAI_API_KEY", "test")
    direct = llm.create_provider("grok-4.6", EngineConfig())
    assert direct._base_url == "https://api.x.ai/v1" and direct.model == "grok-4.6"
    assert direct._wire_api == "chat_completions"
    mantle = llm.create_provider("grok-4.6", EngineConfig(openai_base_url="https://bedrock-mantle.us-west-2.api.aws/v1"))
    assert mantle.model == "xai.grok-4.6" and mantle._wire_api == "responses"


@pytest.mark.parametrize("model", ["minimax.minimax-m2.5", "mistral.devstral-2-123b", "qwen.qwen3-coder-next", "nvidia.nemotron-super-3-120b", "mistral.mistral-large-3-675b-instruct"])
@pytest.mark.parametrize("wire", ["responses", "chat_completions", "auto"])
@pytest.mark.parametrize("path", ["/openai/v1", "/v1", "/anthropic"])
def test_mantle_verified_chat_models_clear_stale_frontier_transport(monkeypatch, model, wire, path):
    monkeypatch.setattr(llm, "_mantle_gateway_key", lambda config: "test")
    endpoint = "https://bedrock-mantle.us-east-1.api.aws" + path
    config = EngineConfig(openai_model=model, openai_base_url=endpoint, openai_wire_api=wire)
    provider = llm.create_provider(model, config)
    assert provider._base_url == "https://bedrock-mantle.us-east-1.api.aws/v1"
    assert provider._wire_api == "chat_completions" and not provider._should_use_responses(True)
    assert provider.model == model
    assert config.openai_base_url == endpoint and config.openai_wire_api == wire


def test_mantle_unknown_model_preserves_explicit_transport(monkeypatch):
    monkeypatch.setattr(llm, "_mantle_gateway_key", lambda config: "test")
    endpoint = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    provider = llm.create_provider("example.custom-model", EngineConfig(openai_base_url=endpoint, openai_wire_api="responses"))
    assert provider._base_url == endpoint and provider._wire_api == "responses"


def signed_blocks(count=2):
    return [
        {"type": "thinking", "thinking": "", "signature": "signed-empty"},
        {"type": "redacted_thinking", "data": "opaque-redacted"},
        *[{"type": "tool_use", "id": f"call-{i}", "name": "lookup", "input": {"value": i}} for i in range(count)],
    ]


def sdk_response(blocks, stop="tool_use"):
    return SimpleNamespace(content=[SimpleNamespace(**block) for block in blocks], stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=11, output_tokens=3,
                                                 cache_creation_input_tokens=0, cache_read_input_tokens=0))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("count,blocked,failed", [(1, False, False), (2, False, False), (2, True, False), (1, True, False), (1, False, True)])
async def test_real_claude_tool_turn_replays_signed_blocks_and_all_results(anthropic_client, tmp_path, count, blocked, failed, streaming, monkeypatch):
    import json
    from clawagents.agent import ClawAgent
    from clawagents.config.features import _FEATURE_DEFAULTS
    from clawagents.run_context import RunContext
    from clawagents.run_result import RunResult
    from clawagents.session.persistence import SessionReader
    from clawagents.tools.registry import ToolRegistry, ToolResult

    monkeypatch.setattr("clawagents.session.persistence._sessions_path", lambda: tmp_path / "sessions")
    executed = []
    class Lookup:
        name = "lookup"
        description = "Look up a value"
        parameters = {"value": {"type": "integer", "required": True}}
        async def execute(self, args):
            executed.append(args["value"])
            return ToolResult(not failed, "exact observation " * 250, error="fixture failure" if failed else None)
    registry = ToolRegistry()
    registry.register(Lookup())
    original = signed_blocks(count)
    replies = [sdk_response(original), sdk_response([{"type": "text", "text": "Finished."}], "end_turn")]
    requests = []
    async def create(**kwargs):
        requests.append(kwargs)
        return replies.pop(0)
    anthropic_client.messages.create = AsyncMock(side_effect=create)
    if streaming:
        from contextlib import asynccontextmanager
        monkeypatch.setattr(llm, "_get_stream_breaker", lambda *args, **kwargs: None)
        @asynccontextmanager
        async def stream(**kwargs):
            requests.append(kwargs)
            reply = replies.pop(0)
            async def events():
                for index, block in enumerate(reply.content):
                    values = dict(vars(block))
                    if block.type == "text":
                        values["text"] = ""
                    elif block.type == "tool_use":
                        values["input"] = {}
                    yield SimpleNamespace(type="content_block_start", index=index, content_block=SimpleNamespace(**values))
                    if block.type == "text":
                        yield SimpleNamespace(type="content_block_delta", index=index, delta=SimpleNamespace(text=block.text))
                    elif block.type == "tool_use":
                        yield SimpleNamespace(type="content_block_delta", index=index, delta=SimpleNamespace(partial_json=json.dumps(block.input)))
                    yield SimpleNamespace(type="content_block_stop", index=index)
                yield SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason=reply.stop_reason), usage=SimpleNamespace(output_tokens=3))
            yield events()
        anthropic_client.messages.stream = stream
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-fable-5-1"))
    features = {name: False for name in _FEATURE_DEFAULTS}
    features["session_persistence"] = True
    agent = ClawAgent(provider, registry, streaming=streaming, workspace=tmp_path, features=features,
                      before_tool=(lambda name, args: args["value"] != count - 1) if blocked else None)
    state = await agent.invoke("Look up the values and finish.", run_context=RunContext(skip_memory=True), max_iterations=4)
    assert state.status == "done", state.result
    assert len(requests) == 2
    request = requests[1]["messages"]
    assistant = next(m for m in request if m["role"] == "assistant" and any(b.get("type") == "tool_use" for b in m["content"]))
    assert assistant["content"] == original
    results = [b for m in request if m["role"] == "user" for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert [b["tool_use_id"] for b in results] == [f"call-{i}" for i in range(count)]
    assert sorted(executed) == list(range(count - int(blocked)))
    if blocked:
        assert "skip" in str(results[-1]["content"]).lower()
    saved = RunResult.from_state(json.loads(json.dumps(RunResult.from_agent_state(state).to_state())))
    assert next(m.anthropic_blocks for m in saved.messages if m.tool_calls_meta) == original
    assert state.session_file
    reader = SessionReader(state.session_file)
    resumed = reader.reconstruct_messages()
    assert not llm.has_active_anthropic_tool_turn(resumed, provider)
    persisted_results = [event for event in reader.events if event["type"] == "tool_result"]
    assert [event["success"] for event in persisted_results] == [not failed and not (blocked and i == count - 1) for i in range(count)]
    if failed:
        assert persisted_results[0]["error"] == "fixture failure"
    # Persist the canonical observation, including outputs beyond the old 2K cap
    # and results for calls rejected by policy.
    expected = [m for m in state.messages if m.role == "tool"]
    actual = [m for m in resumed if m.role == "tool"]
    assert [(m.tool_call_id, m.content) for m in actual] == [(m.tool_call_id, m.content) for m in expected]
    anthropic_client.messages.create = AsyncMock(return_value=sdk_response([{"type": "text", "text": "Resumed."}], "end_turn"))
    await provider.chat(resumed)
    replay = anthropic_client.messages.create.call_args.kwargs["messages"]
    assert next(m["content"] for m in replay if m["role"] == "assistant" and any(b.get("type") == "tool_use" for b in m["content"])) == original


async def test_streamed_signatures_redaction_and_stop_reason_are_retained(anthropic_client, monkeypatch):
    from contextlib import asynccontextmanager
    monkeypatch.setattr(llm, "_get_stream_breaker", lambda *args, **kwargs: None)
    @asynccontextmanager
    async def stream(**kwargs):
        async def events():
            yield SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="thinking", thinking="", signature=""))
            for part in ["signed-", "empty"]:
                yield SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(signature=part))
            yield SimpleNamespace(type="content_block_stop", index=0)
            yield SimpleNamespace(type="content_block_start", index=1, content_block=SimpleNamespace(type="redacted_thinking", data="opaque-redacted"))
            yield SimpleNamespace(type="content_block_stop", index=1)
            yield SimpleNamespace(type="content_block_start", index=2, content_block=SimpleNamespace(type="tool_use", id="call-0", name="lookup", input={}))
            yield SimpleNamespace(type="content_block_delta", index=2, delta=SimpleNamespace(partial_json='{"value":'))
            yield SimpleNamespace(type="content_block_delta", index=2, delta=SimpleNamespace(partial_json='0}'))
            yield SimpleNamespace(type="content_block_stop", index=2)
            yield SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="max_tokens"), usage=SimpleNamespace(output_tokens=9))
        yield events()
    anthropic_client.messages.stream = stream
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-opus-5"))
    response = await provider.chat([llm.LLMMessage("user", "look up")], on_chunk=MagicMock())
    assert response.anthropic_blocks == signed_blocks(1)
    assert response.finish_reason == "max_tokens"
    assert response.tool_calls[0].args == {"value": 0}
    messages = [llm.LLMMessage("user", "look up"), llm.LLMMessage("assistant", response.content,
                anthropic_blocks=response.anthropic_blocks, tool_calls_meta=[{"id": "call-0", "name": "lookup", "args": {"value": 0}}]),
                llm.LLMMessage("tool", "result", tool_call_id="call-0")]
    await provider.chat(messages)
    assert anthropic_client.messages.create.call_args.kwargs["messages"][1]["content"] == signed_blocks(1)


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
async def test_session_backends_roundtrip_raw_claude_blocks(tmp_path, backend):
    from clawagents.session.backends import JsonlFileSession, SQLiteSession
    session = JsonlFileSession("signed", file_path=tmp_path / "session.jsonl") if backend == "jsonl" else SQLiteSession("signed", db_path=tmp_path / "session.db")
    original = llm.LLMMessage("assistant", "", anthropic_blocks=signed_blocks(), tool_calls_meta=[{"id": "call-0", "name": "lookup", "args": {}}])
    await session.add_items([original])
    restored = await session.get_items()
    assert restored[0].anthropic_blocks == original.anthropic_blocks


def test_hidden_claude_reasoning_is_counted_and_not_display_trimmed():
    from clawagents.tokenizer import count_messages_tokens
    from clawagents.memory.output_trim import trim_verbose_messages
    message = llm.LLMMessage("assistant", "display " * 3000, anthropic_blocks=[{"type": "thinking", "thinking": "hidden " * 20000, "signature": "opaque"}])
    trimmed, count = trim_verbose_messages([message])
    assert trimmed[0] is message and count == 0
    assert count_messages_tokens([message]) > count_messages_tokens([llm.LLMMessage("assistant", message.content)]) * 2


def signed_driver(provider, budget=1000):
    from clawagents.graph.turn_driver import TurnDriver
    driver = TurnDriver.__new__(TurnDriver)
    driver._llm = provider
    driver._run_context = None
    driver._token_ledger = None
    driver._context_window = 1000000
    driver._resolved_model_name = "claude-fable-5-1"
    driver._events = SimpleNamespace(emit=MagicMock())
    driver._token_multiplier = 1.0
    driver._cached_system_tokens = 0
    driver._external_hooks = None
    driver._before_llm = MagicMock(side_effect=lambda rows: rows)
    driver._input_budget = lambda: budget
    driver._budget_tokens = lambda rows, *args: 10 if len(rows) == 1 else 100
    driver._micro_compact = MagicMock(side_effect=lambda rows, tokens: (rows, tokens))
    driver._compact = AsyncMock(side_effect=lambda rows: rows)
    return driver


def signed_history():
    return [llm.LLMMessage("user", "task"), llm.LLMMessage("assistant", "", anthropic_blocks=signed_blocks(1),
              tool_calls_meta=[{"id": "call-0", "name": "lookup", "args": {"value": 0}}]),
              llm.LLMMessage("tool", "result", tool_call_id="call-0")]


async def test_active_signed_round_defers_compaction_and_resumes_after_answer(anthropic_client):
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-fable-5-1"))
    driver = signed_driver(provider)
    messages = signed_history()
    assert await driver._prepare_messages(messages) == messages
    driver._micro_compact.assert_not_called()
    driver._compact.assert_not_called()
    driver._before_llm.assert_called_once()
    driver._before_llm.reset_mock()
    messages.append(llm.LLMMessage("assistant", "finished", anthropic_blocks=[{"type": "text", "text": "finished"}]))
    await driver._prepare_messages(messages)
    driver._micro_compact.assert_called_once()
    driver._before_llm.assert_called_once()
    assert not llm.has_active_anthropic_tool_turn(signed_history(), SimpleNamespace(model="other"))


async def test_active_signed_round_stops_on_local_budget_or_provider_overflow(anthropic_client):
    from clawagents.graph.context_management import _InputBudgetExceeded
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-fable-5-1"))
    driver = signed_driver(provider, budget=50)
    with pytest.raises(_InputBudgetExceeded, match="active Claude"):
        await driver._prepare_messages(signed_history())
    state = SimpleNamespace(status="running", result="")
    outcome = await driver._recover_from_error(signed_history(), state, 1, ValueError("context_length_exceeded"))
    assert outcome.action == "stop" and state.status == "error"
    assert "deferred" in state.result
    driver._compact.assert_not_called()


def test_signed_round_guard_handles_advisor_messages_and_fallback_wrappers(anthropic_client):
    from clawagents.providers.fallback import FallbackProvider
    provider = llm.AnthropicProvider(EngineConfig(anthropic_api_key="test", anthropic_model="claude-opus-5"))
    messages = signed_history() + [llm.LLMMessage("assistant", "external advisor message")]
    assert llm.has_active_anthropic_tool_turn(messages, provider)
    assert llm.has_active_anthropic_tool_turn(messages, FallbackProvider(SimpleNamespace(model="other"), [provider]))
    assert not llm.has_active_anthropic_tool_turn(messages, FallbackProvider(SimpleNamespace(model="other"), []))


async def test_fable_binding_controls_reach_real_sdk_request_without_network():
    import json
    import anthropic
    from anthropic import _base_client

    # Anthropic 1.x moved to httpx2; construct the transport from the SDK
    # dependency so this wire test exercises both supported SDK generations.
    httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx
    captured = []
    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"id": "msg_fixture", "type": "message", "role": "assistant",
            "model": "claude-fable-5-1", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 1}})
    provider = llm.AnthropicProvider.__new__(llm.AnthropicProvider)
    provider.model = "claude-fable-5-1"
    provider._max_tokens = 1000
    provider._temperature = 0.2
    provider._reasoning_effort = "high"
    provider.client = anthropic.AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        result = await provider.chat(signed_history())
    finally:
        await provider.client.close()
    assert result.content == "ok"
    request = captured[0]
    wire = json.loads(request.content)
    assert request.headers["anthropic-beta"] == "thinking-binding-controls-2026-08-01"
    assert wire["thinking"] == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert wire["messages"][1]["content"] == signed_blocks(1)


def test_session_finalizer_does_not_attach_raw_response_to_rewritten_output(tmp_path):
    from clawagents.graph.run_finalizer import RunFinalizer
    from clawagents.session.persistence import SessionReader, SessionWriter
    finalizer = RunFinalizer.__new__(RunFinalizer)
    finalizer._session_writer = SessionWriter(session_dir=tmp_path)
    state = SimpleNamespace(result="Guardrail refused this answer.", iterations=1, tool_calls=0, status="error",
                            messages=[llm.LLMMessage("assistant", "Original answer", anthropic_blocks=[
                                {"type": "thinking", "thinking": "", "signature": "signed-empty"},
                                {"type": "text", "text": "Original answer"}])])
    finalizer._write_session_completion(state)
    restored = SessionReader(state.session_file).reconstruct_messages()
    assert restored[-1].content == state.result
    assert restored[-1].anthropic_blocks is None
