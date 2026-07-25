"""Cache-preserving tool activation: which tools defer, and referencing them."""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from clawagents.providers.deferred_tools import (
    is_deferred_tool_rejection,
    model_supports_tool_references,
    split_deferred_tools,
    tool_reference_blocks,
)
from clawagents.providers.llm import AnthropicProvider, LLMMessage, NativeToolSchema


def _schema(name: str) -> NativeToolSchema:
    return NativeToolSchema(name, f"{name} description", {})


def _activation(call_id: str, added: list[str]) -> LLMMessage:
    return LLMMessage(
        role="tool",
        content=json.dumps({"activated": "web", "tools": added}),
        tool_call_id=call_id,
        added_tool_names=added,
    )


def test_tools_added_by_a_tool_result_are_deferred():
    messages = [
        LLMMessage(role="user", content="search the web"),
        _activation("c1", ["web_search", "web_fetch"]),
    ]
    schemas = [_schema("read_file"), _schema("web_search"), _schema("web_fetch")]

    immediate, deferred = split_deferred_tools(messages, schemas)

    assert [s.name for s in immediate] == ["read_file"]
    assert sorted(deferred) == ["web_fetch", "web_search"]


def test_deferred_stays_deferred_once_the_model_calls_it():
    """Stability is the point: the deferred set must not churn as turns arrive.

    Promoting a tool into the prefix after it gets called would rewrite the
    cached prefix — the exact cost this mechanism avoids — so an activation
    that deferred a tool keeps deferring it for the rest of the run.
    """
    messages = [
        _activation("c1", ["web_search"]),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c2", "name": "web_search", "args": {}}],
        ),
    ]
    immediate, deferred = split_deferred_tools(messages, [_schema("web_search")])

    assert immediate == []
    assert list(deferred) == ["web_search"]


def test_a_tool_called_before_its_activation_is_not_deferred():
    """Already established in the prefix — referencing it would be redundant."""
    messages = [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c1", "name": "web_search", "args": {}}],
        ),
        _activation("c2", ["web_search"]),
    ]
    immediate, deferred = split_deferred_tools(messages, [_schema("web_search")])

    assert [s.name for s in immediate] == ["web_search"]
    assert deferred == {}


def test_disabled_keeps_every_tool_immediate():
    messages = [_activation("c1", ["web_search"])]
    schemas = [_schema("read_file"), _schema("web_search")]

    immediate, deferred = split_deferred_tools(messages, schemas, enabled=False)

    assert sorted(s.name for s in immediate) == ["read_file", "web_search"]
    assert deferred == {}


def test_duplicate_schemas_collapse_by_name():
    immediate, _ = split_deferred_tools([], [_schema("read_file"), _schema("read_file")])
    assert len(immediate) == 1


def test_reference_blocks_introduce_each_tool_exactly_once():
    msg = _activation("c1", ["web_search", "web_fetch"])
    seen: set[str] = set()

    first = tool_reference_blocks(msg, {"web_search", "web_fetch"}, seen)
    assert first == [
        {"type": "tool_reference", "tool_name": "web_search"},
        {"type": "tool_reference", "tool_name": "web_fetch"},
    ]

    # A repeat reference to an already-introduced tool is rejected upstream.
    assert tool_reference_blocks(msg, {"web_search", "web_fetch"}, seen) == []


def test_reference_blocks_skip_tools_that_are_not_deferred():
    msg = _activation("c1", ["web_search"])
    assert tool_reference_blocks(msg, set(), set()) == []


def test_activation_tool_records_what_it_added():
    """The activating ToolResult must carry the names for the transcript."""
    from clawagents.tools.registry import ToolResult

    result = ToolResult(True, "{}", added_tool_names=["web_search"])
    assert result.added_tool_names == ["web_search"]
    # Default stays None so ordinary results are unaffected.
    assert ToolResult(True, "ok").added_tool_names is None


# ── Anthropic wire emission + self-healing fallback ────────────────────────


def _transcript_with_activation():
    return [
        LLMMessage(role="user", content="search the web"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "c1", "name": "activate_tool_group", "args": {}}],
        ),
        _activation("c1", ["web_search"]),
    ]


def _fake_anthropic(monkeypatch, *, fail_first=False, model="claude-opus-4-8"):
    """AnthropicProvider wired to a stub client that records the request."""
    captured: dict = {}
    state = {"calls": 0}

    class _Messages:
        async def create(self, **kw):
            state["calls"] += 1
            captured.clear()
            captured.update(kw)
            if fail_first and state["calls"] == 1:
                raise RuntimeError(
                    "400 invalid_request_error: unexpected field `defer_loading`"
                )
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="ok")],
                stop_reason="end_turn",
                usage=types.SimpleNamespace(
                    input_tokens=10,
                    output_tokens=2,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            )

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = model
    provider._max_tokens = 100
    provider._temperature = 0.0
    provider._streaming = False
    provider._structured_json_schema = None
    provider.retry_policy = None
    provider.client = types.SimpleNamespace(messages=_Messages())
    return provider, captured, state


@pytest.fixture()
def deferral_on(monkeypatch):
    monkeypatch.setenv("CLAW_FEATURE_DEFERRED_TOOL_LOADING", "1")
    from clawagents.config import features

    features.reset()
    yield
    features.reset()


def test_anthropic_emits_defer_loading_and_tool_reference(deferral_on, monkeypatch):
    provider, captured, _ = _fake_anthropic(monkeypatch)
    tools = [_schema("read_file"), _schema("web_search")]

    asyncio.run(provider.chat(_transcript_with_activation(), tools=tools))

    flags = {t["name"]: t.get("defer_loading") for t in captured["tools"]}
    assert flags == {"read_file": None, "web_search": True}

    results = [
        b
        for m in captured["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    assert results[0]["content"] == [
        {"type": "tool_reference", "tool_name": "web_search"}
    ]
    # Anthropic rejects references mixed with content, so the real tool output
    # must survive as a sibling text block rather than being dropped.
    texts = [
        b
        for m in captured["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "text"
    ]
    assert any("web_search" in b["text"] for b in texts)


def test_models_without_tool_reference_support_are_skipped(deferral_on, monkeypatch):
    provider, captured, _ = _fake_anthropic(monkeypatch, model="claude-haiku-4-5")
    asyncio.run(
        provider.chat(_transcript_with_activation(), tools=[_schema("web_search")])
    )
    assert not any(t.get("defer_loading") for t in captured["tools"])


def test_flag_off_sends_the_ordinary_tool_list(monkeypatch):
    from clawagents.config import features

    monkeypatch.delenv("CLAW_FEATURE_DEFERRED_TOOL_LOADING", raising=False)
    features.reset()
    provider, captured, _ = _fake_anthropic(monkeypatch)
    asyncio.run(
        provider.chat(_transcript_with_activation(), tools=[_schema("web_search")])
    )
    assert not any(t.get("defer_loading") for t in captured["tools"])


def test_rejection_disables_deferral_and_retries_without_losing_the_turn(
    deferral_on, monkeypatch
):
    """The wire shape is unverifiable offline, so it must fail soft."""
    provider, captured, state = _fake_anthropic(monkeypatch, fail_first=True)

    response = asyncio.run(
        provider.chat(_transcript_with_activation(), tools=[_schema("web_search")])
    )

    assert response.content == "ok"
    assert state["calls"] == 2
    assert provider._deferred_disabled
    assert not any(t.get("defer_loading") for t in captured["tools"])


def test_unrelated_errors_are_not_swallowed(deferral_on, monkeypatch):
    provider, _captured, state = _fake_anthropic(monkeypatch)

    async def _boom(**_kw):
        state["calls"] += 1
        raise RuntimeError("529 overloaded_error")

    provider.client.messages.create = _boom
    with pytest.raises(RuntimeError, match="overloaded"):
        asyncio.run(provider.chat(_transcript_with_activation(), tools=[_schema("x")]))
    assert state["calls"] == 1  # no retry
    assert not provider._deferred_disabled


def test_rejection_detector_is_specific():
    assert is_deferred_tool_rejection(
        RuntimeError("400 invalid_request_error: unknown field `defer_loading`")
    )
    assert not is_deferred_tool_rejection(RuntimeError("529 overloaded_error"))
    assert not is_deferred_tool_rejection(RuntimeError("400 invalid model name"))


def test_compat_matrix():
    assert model_supports_tool_references("claude-opus-4-8")
    assert not model_supports_tool_references("claude-haiku-4-5")
    assert not model_supports_tool_references("")


# ── OpenAI Responses wire (tool_search_call / tool_search_output) ──────────
from clawagents.providers.llm import (  # noqa: E402
    OpenAIProvider,
    _openai_chat_messages,
    _sanitize_openai_tool_pairs,
    _to_openai_tools,
)


def _responses_provider():
    p = OpenAIProvider.__new__(OpenAIProvider)
    p.model = "gpt-5.6-luna"
    p._max_tokens = 100
    p._temperature = None
    p._base_url = ""
    p._reasoning_effort = "medium"
    p._structured_json_schema = None
    p._wire_api = "responses"
    p._deferred_disabled = False
    return p


def _responses_kwargs_for(messages, tools):
    provider = _responses_provider()
    oai = _to_openai_tools(tools)
    provider._prepare_deferred_tools(messages, tools, oai)
    formatted = _sanitize_openai_tool_pairs(_openai_chat_messages(messages))
    return provider, formatted, provider._responses_kwargs(formatted, oai)


def test_responses_moves_deferred_tool_out_of_the_cached_prefix(deferral_on):
    messages = _transcript_with_activation()
    tools = [_schema("read_file"), _schema("web_search")]

    _p, _fmt, kwargs = _responses_kwargs_for(messages, tools)

    # Unlike Anthropic, the deferred schema must NOT remain in `tools` —
    # leaving it there would rewrite the very prefix this protects.
    assert [t["name"] for t in kwargs["tools"]] == ["read_file"]

    pair = [i for i in kwargs["input"] if str(i.get("type", "")).startswith("tool_search")]
    assert [i["type"] for i in pair] == ["tool_search_call", "tool_search_output"]
    assert pair[0]["call_id"] == pair[1]["call_id"]
    assert pair[0]["arguments"] == {"query": "web_search", "limit": 1}
    assert pair[1]["tools"][0]["defer_loading"] is True


def test_tool_search_pair_follows_the_activating_result(deferral_on):
    _p, _fmt, kwargs = _responses_kwargs_for(
        _transcript_with_activation(), [_schema("web_search")]
    )
    types_ = [str(i.get("type") or i.get("role")) for i in kwargs["input"]]
    assert types_.index("function_call_output") < types_.index("tool_search_call")


def test_deferred_metadata_never_reaches_the_chat_completions_wire(deferral_on):
    """`added_tool_names` is engine-side only; an extra key could 400 chat."""
    _p, formatted, _kwargs = _responses_kwargs_for(
        _transcript_with_activation(), [_schema("web_search")]
    )
    assert all("added_tool_names" not in m for m in formatted)


def test_responses_flag_off_keeps_every_tool_in_the_prefix(monkeypatch):
    from clawagents.config import features

    monkeypatch.delenv("CLAW_FEATURE_DEFERRED_TOOL_LOADING", raising=False)
    features.reset()
    _p, _fmt, kwargs = _responses_kwargs_for(
        _transcript_with_activation(), [_schema("read_file"), _schema("web_search")]
    )
    assert sorted(t["name"] for t in kwargs["tools"]) == ["read_file", "web_search"]
    assert not any(str(i.get("type", "")).startswith("tool_search") for i in kwargs["input"])


def test_responses_rejection_clears_deferral_state(deferral_on):
    provider, _fmt, _kwargs = _responses_kwargs_for(
        _transcript_with_activation(), [_schema("web_search")]
    )
    assert provider._deferred_names  # armed

    assert provider._disable_deferred_tools(
        RuntimeError("400 invalid_request_error: unknown field `tool_search_call`")
    )
    assert provider._deferred_disabled
    assert provider._deferred_names is None
    # An unrelated failure must not disarm it a second time.
    assert not provider._disable_deferred_tools(RuntimeError("529 overloaded"))
