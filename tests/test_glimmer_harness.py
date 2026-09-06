"""Muse-Glimmer / self-hosted reasoning-model harness regressions (no network).

Covers the failure classes seen in benchmarks/meta_challenge_20260906: reasoning
on a separate channel the loop never saw, max_tokens cuts mid-thought treated
as answers, retry storms on the same failing shell approach, and served-name
aliases silently losing the model-specific harness. Plus the workspace
profile-file trust boundary.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from clawagents.config import features
from clawagents.graph.completion_handler import (
    TRUNCATION_MARKER,
    CompletionHandler,
    _truncation_nudge_count,
    response_hit_output_limit,
)
from clawagents.graph.loop_tracker import (
    _ToolCallTracker,
    failure_signature,
    repeated_failure_directive,
)
from clawagents.graph.model_profiles import register_model_profile_alias, resolve_model_profile
from clawagents.graph.turn_response import TurnResponseInterpreter
from clawagents.harness_profiles import register_harness_alias, resolve_harness_profile
from clawagents.providers.llm import (
    LLMMessage,
    LLMResponse,
    _is_retryable,
    _openai_reasoning_text,
    _openai_reasoning_tokens,
    _parse_openai_tool_calls,
    _ThinkStreamFilter,
    strip_thinking_tokens,
)

META_URL = "http://localhost:7790/v1"


# ─── provider parsing helpers ────────────────────────────────────────────


def test_reasoning_text_reads_attribute_dict_and_model_extra():
    assert _openai_reasoning_text(SimpleNamespace(reasoning_content="why")) == "why"
    assert _openai_reasoning_text({"reasoning": "because"}) == "because"
    extra_only = SimpleNamespace(model_extra={"reasoning_content": "hidden"})
    assert _openai_reasoning_text(extra_only) == "hidden"
    assert _openai_reasoning_text(SimpleNamespace(content="x")) == ""


def test_reasoning_tokens_from_details_or_sglang_top_level():
    details = SimpleNamespace(
        completion_tokens_details=SimpleNamespace(reasoning_tokens=7), reasoning_tokens=None
    )
    assert _openai_reasoning_tokens(details) == 7
    assert _openai_reasoning_tokens({"reasoning_tokens": 25}) == 25
    assert _openai_reasoning_tokens(None) == 0


def test_parse_tool_calls_accepts_dict_args_missing_type_and_missing_id():
    calls = _parse_openai_tool_calls(
        [
            SimpleNamespace(type="function", id="", function=SimpleNamespace(name="a", arguments={"p": 1})),
            SimpleNamespace(type=None, id=None, function=SimpleNamespace(name="b", arguments='{"q": 2}')),
            SimpleNamespace(type="custom", id="c3", function=SimpleNamespace(name="c", arguments="{}")),
        ]
    )
    assert [c.tool_name for c in calls] == ["a", "b"]
    assert calls[0].args == {"p": 1}
    assert calls[1].args == {"q": 2}
    assert calls[0].tool_call_id and calls[1].tool_call_id
    assert calls[0].tool_call_id != calls[1].tool_call_id


def test_parse_tool_calls_drops_call_truncated_by_max_tokens():
    truncated = [
        SimpleNamespace(type="function", id="c1", function=SimpleNamespace(name="write_file", arguments='{"path": "/tm')),
    ]
    assert _parse_openai_tool_calls(truncated, finish_reason="length") is None
    kept = _parse_openai_tool_calls(truncated, finish_reason="stop")
    assert kept and kept[0].args == {"path": "/tm"}


def test_think_filter_handles_split_tags_and_unclosed_block():
    f = _ThinkStreamFilter()
    visible, thinking = f.feed("<thi")
    assert (visible, thinking) == ("", "")
    visible, thinking = f.feed("nk>secret</th")
    assert visible == "" and thinking == "secret"
    visible, thinking = f.feed("ink>answer")
    assert visible == "answer" and thinking == ""
    rest = f.flush()
    assert rest == ("", "")

    unclosed = _ThinkStreamFilter()
    visible, thinking = unclosed.feed("<think>partial thought")
    assert visible == ""
    visible, thinking2 = unclosed.flush()
    assert visible == "" and thinking + thinking2 == "partial thought"

    plain = _ThinkStreamFilter()
    assert plain.feed("a < b and c") == ("a < b and c", "")


def test_strip_thinking_tokens_treats_unclosed_block_as_thinking():
    clean, thinking = strip_thinking_tokens("<think>step 1\nstep 2")
    assert clean == ""
    assert thinking == "step 1\nstep 2"
    clean, thinking = strip_thinking_tokens("<think>a</think>answer")
    assert (clean, thinking) == ("answer", "a")


def test_retryable_ignores_status_like_digits_in_400_messages():
    assert not _is_retryable(Exception("Error code: 400 - unknown field at line 1 column 500"))
    assert not _is_retryable(Exception("prompt is 15024 tokens, max 8192"))
    assert _is_retryable(Exception("Error code: 502 Bad Gateway"))
    assert _is_retryable(Exception("HTTP 503 service unavailable"))
    assert _is_retryable(SimpleNamespace(status_code=429)) is False  # not an Exception
    assert _is_retryable(type("E", (Exception,), {"status_code": 429})("x"))


# ─── streaming path against a fake SGLang-style server ──────────────────


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self):
        self.closed = True


def _chunk(*, content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=usage)


def _usage(**kw):
    base = dict(total_tokens=30, prompt_tokens=20, prompt_tokens_details=None,
                completion_tokens_details=None, reasoning_tokens=0)
    base.update(kw)
    return SimpleNamespace(**base)


def _meta_agent(tmp_path, **kw):
    from clawagents.agent import create_claw_agent

    kw.setdefault("streaming", True)
    return create_claw_agent(
        profile="meta", base_url=META_URL, workspace=tmp_path, skills=[], memory=[], **kw,
    )


def _install_stream(agent, chunks):
    stream = _FakeStream(chunks)

    async def create(**kwargs):
        create.kwargs = kwargs
        return stream

    agent.llm.client.chat.completions.create = create
    return stream


def test_streaming_captures_reasoning_channel_and_finish_reason(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path)
    seen: list[str] = []
    _install_stream(agent, [
        _chunk(reasoning="Let me "),
        _chunk(reasoning="think."),
        _chunk(content="Done"),
        _chunk(finish="stop", usage=_usage(reasoning_tokens=9)),
    ])
    resp = asyncio.run(agent.llm.chat([LLMMessage(role="user", content="hi")], on_chunk=seen.append))
    assert resp.content == "Done"
    assert resp.thinking == "Let me think."
    assert resp.finish_reason == "stop"
    assert resp.reasoning_tokens == 9
    assert seen == ["Done"]  # reasoning never reaches the visible stream


def test_streaming_length_cut_mid_reasoning_is_recoverable_not_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path)
    _install_stream(agent, [
        _chunk(reasoning="x" * 50),
        _chunk(finish="length", usage=_usage()),
    ])
    resp = asyncio.run(agent.llm.chat([LLMMessage(role="user", content="hi")], on_chunk=lambda _t: None))
    assert resp.content == ""
    assert resp.finish_reason == "length"
    assert resp.partial is False  # not a cancellation: the loop must continue, not stop
    assert resp.tool_calls is None
    assert response_hit_output_limit(resp)


def test_streaming_tool_calls_without_index_or_id_and_dict_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path)
    tc1 = SimpleNamespace(index=None, id="call_a", function=SimpleNamespace(name="read_file", arguments={"path": "a.py"}))
    tc2 = SimpleNamespace(index=None, id="call_b", function=SimpleNamespace(name="grep", arguments='{"pattern": "x"}'))
    tc3 = SimpleNamespace(index=2, id=None, function=SimpleNamespace(name="ls", arguments=""))
    _install_stream(agent, [
        _chunk(tool_calls=[tc1]),
        _chunk(tool_calls=[tc2]),
        _chunk(tool_calls=[tc3]),
        _chunk(finish="tool_calls", usage=_usage()),
    ])
    resp = asyncio.run(agent.llm.chat([LLMMessage(role="user", content="hi")], on_chunk=lambda _t: None))
    calls = {c.tool_name: c for c in resp.tool_calls}
    assert set(calls) == {"read_file", "grep", "ls"}
    assert calls["read_file"].args == {"path": "a.py"}
    assert calls["grep"].args == {"pattern": "x"}
    assert calls["ls"].args == {}
    assert calls["ls"].tool_call_id  # synthesised, never empty


def test_streaming_drops_tool_call_truncated_by_length(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path)
    tc = SimpleNamespace(index=0, id="call_a", function=SimpleNamespace(name="write_file", arguments='{"path": "/tm'))
    _install_stream(agent, [_chunk(tool_calls=[tc]), _chunk(finish="length", usage=_usage())])
    resp = asyncio.run(agent.llm.chat([LLMMessage(role="user", content="hi")], on_chunk=lambda _t: None))
    assert resp.tool_calls is None
    assert response_hit_output_limit(resp)


def test_streaming_inline_think_is_hidden_from_visible_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path)
    seen: list[str] = []
    _install_stream(agent, [
        _chunk(content="<thi"),
        _chunk(content="nk>plan</th"),
        _chunk(content="ink>final"),
        _chunk(finish="stop", usage=_usage()),
    ])
    resp = asyncio.run(agent.llm.chat([LLMMessage(role="user", content="hi")], on_chunk=seen.append))
    assert "".join(seen) == "final"
    assert resp.content == "final"
    assert resp.thinking == "plan"


def test_streaming_closes_stream_on_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path)

    class _Boom(_FakeStream):
        async def __anext__(self):
            raise ValueError("Error code: 400 - bad request")

    stream = _Boom([])

    async def create(**kwargs):
        return stream

    agent.llm.client.chat.completions.create = create
    with pytest.raises(ValueError):
        asyncio.run(agent.llm.chat([LLMMessage(role="user", content="hi")], on_chunk=lambda _t: None))
    assert stream.closed


def test_non_streaming_captures_reasoning_and_finish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path, streaming=False)
    msg = SimpleNamespace(content="", reasoning_content="deep thought", tool_calls=None)

    async def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="length")],
            usage=_usage(reasoning_tokens=40),
        )

    agent.llm.client.chat.completions.create = create
    resp = asyncio.run(agent.llm.chat([LLMMessage(role="user", content="hi")]))
    assert resp.thinking == "deep thought"
    assert resp.finish_reason == "length"
    assert resp.reasoning_tokens == 40


# ─── loop: thinking extraction + output-limit recovery ───────────────────


def test_extract_thinking_keeps_finish_reason_when_content_has_think_tag():
    interpreter = TurnResponseInterpreter(llm=None, registry=None, events=SimpleNamespace(emit=lambda *a, **k: None))
    response = LLMResponse(
        content="<think>unfinished", model="m", tokens_used=1, finish_reason="length", reasoning_tokens=3
    )
    parsed = interpreter.parse(response, use_native_tools=True, run_context=SimpleNamespace(_metadata={}))
    assert parsed.response.content == ""
    assert parsed.thinking == "unfinished"
    assert parsed.response.finish_reason == "length"
    assert parsed.response.reasoning_tokens == 3
    assert response_hit_output_limit(parsed.response)


class _Events:
    def __init__(self):
        self.warnings: list[str] = []

    def emit(self, kind, data=None):
        if kind == "warn":
            self.warnings.append((data or {}).get("message", ""))

    def typed(self, kind, data=None):
        pass


def _handler(llm):
    return CompletionHandler(
        registry=None,
        run_context=SimpleNamespace(_metadata={}),
        events=_Events(),
        recorder=None,
        llm=llm,
        before_tool=None,
        action_mode="tools",
        looks_like_truncated_json=lambda _t: False,
        sanitize_assistant_text=lambda t: t,
        goal_llm_complete=lambda *_a, **_k: (lambda _s: _s),
    )


def test_output_limit_nudges_twice_then_stops_nudging():
    llm = SimpleNamespace(_max_tokens=6144)
    handler = _handler(llm)
    messages = [LLMMessage(role="user", content="task")]
    state = SimpleNamespace(status="running", result="", tool_calls=0)

    async def consult(_m, _p):
        return None

    cut = LLMResponse(content="", model="m", tokens_used=0, finish_reason="length")
    for expected in (1, 2):
        decision = asyncio.run(handler.handle(
            state=state, messages=messages, response=cut, thinking="…", use_native_tools=True,
            consult_advisor=consult, should_final_check=False,
        ))
        assert decision.action == "continue"
        assert _truncation_nudge_count(messages) == expected
        assert messages[-1].role == "user" and messages[-1].content.startswith(TRUNCATION_MARKER)
        assert messages[-2].role == "assistant" and messages[-2].content  # never an empty turn
    assert llm._max_tokens == 13824  # 6144 → 9216 → 13824
    assert any("max_tokens" in w for w in handler._events.warnings)
    # Third cut: no further nudge is appended.
    before = len(messages)
    asyncio.run(handler.handle(
        state=state, messages=messages, response=cut, thinking=None, use_native_tools=True,
        consult_advisor=consult, should_final_check=False,
    ))
    assert _truncation_nudge_count(messages) == 2
    assert not any(m.role == "user" and m.content.startswith(TRUNCATION_MARKER) for m in messages[before:])


def test_output_limit_not_triggered_when_a_tool_call_survived():
    resp = LLMResponse(content="", model="m", tokens_used=0, finish_reason="length",
                       tool_calls=[SimpleNamespace(tool_name="x", args={}, tool_call_id="1")])
    assert not response_hit_output_limit(resp)
    interrupted = LLMResponse(content="half an answer", model="m", tokens_used=0, partial=True)
    assert response_hit_output_limit(interrupted)
    cancelled = LLMResponse(content="", model="m", tokens_used=0, partial=True)
    assert not response_hit_output_limit(cancelled)  # turn driver owns that case


# ─── loop: repeated-failure escalation ───────────────────────────────────


def test_failure_signature_normalises_paths_and_numbers():
    a = failure_signature("execute", "exit code 1\nstderr:\ncat: /private/var/folders/x1/T/claw-1/core.py: Operation not permitted")
    b = failure_signature("execute", "exit code 2\nstderr:\ncat: /private/var/folders/y9/T/claw-2/other.py: Operation not permitted")
    assert a == b
    assert failure_signature("execute", "ImportError: no module named x") != a
    assert failure_signature("read_file", "cat: <x>: Operation not permitted") != a  # tool is part of the key


def test_tracker_escalates_on_second_identical_failure_and_hardens_on_third():
    tracker = _ToolCallTracker()
    err = "Error: unsandboxed_not_authorized: command was not run. Retrying unchanged will remain unauthorized."
    assert tracker.record_result("execute", {"command": "ls /a", "unsandboxed": True}, err, success=False) is None
    second = tracker.record_result("execute", {"command": "ls /b", "unsandboxed": True}, err, success=False)
    assert second and "twice" in second and "read_file" in second
    third = tracker.record_result("execute", {"command": "cat /c", "unsandboxed": True}, err, success=False)
    assert third and "STOP" in third
    # A success with the same tool does not count, and other errors are independent.
    assert tracker.record_result("execute", {"command": "pwd"}, "ok", success=True) is None
    assert tracker.record_result("execute", {"command": "x"}, "Error: something else", success=False) is None
    assert repeated_failure_directive("t", 1) is None


def test_time_dependent_tools_never_escalate():
    tracker = _ToolCallTracker()
    for _ in range(3):
        assert tracker.record_result("task_wait", {"job_id": "j"}, "Error: still running", success=False) is None


# ─── harness / model-profile aliases ─────────────────────────────────────


def test_custom_glimmer_alias_keeps_full_meta_harness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("glimmer_30B_model", "Custom-Glimmer")
    monkeypatch.setattr("clawagents.provider_profiles._profile_paths", lambda: [])
    agent = _meta_agent(tmp_path)
    assert agent.llm.model == "Custom-Glimmer"
    assert "Tool efficiency:" in agent.system_prompt
    assert "reasoning short" in agent.system_prompt
    assert not agent.tools.is_tool_active("web_fetch")
    assert agent.tools.is_tool_active("activate_tool_group")
    assert resolve_harness_profile("Custom-Glimmer").name == "meta-glimmer"
    assert resolve_model_profile("Custom-Glimmer")["max_input_tokens"] == 196_608
    assert agent.llm._max_tokens == 16_384
    assert agent.context_window == 196_608


def test_explicit_meta_max_tokens_and_window_win(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path, max_tokens=6144, context_window=100_000)
    assert agent.llm._max_tokens == 6144
    assert agent.context_window == 100_000


def test_gemma_alias_keeps_harness_and_chat_options(tmp_path, monkeypatch):
    from clawagents.agent import create_claw_agent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMMA_AGENTIC_MODEL", "my-gemma-alias")
    monkeypatch.setenv("GEMMA_AGENTIC_BASE_URL", "http://127.0.0.1:18080/v1")
    monkeypatch.setattr("clawagents.provider_profiles._profile_paths", lambda: [])
    agent = create_claw_agent(profile="gemma-agentic", workspace=tmp_path, skills=[], memory=[])
    assert resolve_harness_profile("my-gemma-alias").name == "gemma-agentic"
    assert resolve_model_profile("my-gemma-alias")["max_input_tokens"] == 16_384
    assert getattr(agent.llm, "_gemma_agentic_model", "") == "my-gemma-alias"
    assert "Delegate with explicit inputs" in agent.system_prompt


def test_alias_registration_is_explicit_and_validated():
    register_model_profile_alias("weird-name", "does-not-exist")
    assert resolve_model_profile("weird-name") is None
    register_harness_alias("weird-name", "meta-glimmer")
    assert resolve_harness_profile("weird-name").name == "meta-glimmer"


def test_fallback_wrapper_exposes_primary_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _meta_agent(tmp_path, fallback_models=["gpt-4o"])
    assert agent.llm.model == "Muse-Glimmer-30B"
    assert getattr(agent.llm, "primary", None) is not None


# ─── harness matching hygiene ────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.6-luna", "openai-gpt56"),
        ("openai.gpt-5.6-luna", "openai-gpt56"),
        ("gpt-5.3-codex", "openai-codex"),
        ("llama3.1", "local-ollama"),
        ("gemma4:e4b", "local-ollama"),
        ("codellama:13b", "local-ollama"),
        ("gemma4-agentic-v2", "gemma-agentic"),
        ("Muse-Glimmer-30B", "meta-glimmer"),
        ("claude-sonnet-4-5", "anthropic-sonnet"),
        # Cloud-qualified ids are never "local ollama" models.
        ("deepseek.v3.2", None),
        ("mistral.mistral-large-2407-v1:0", None),
        ("us.meta.llama3-3-70b-instruct-v1:0", None),
        ("bedrock/meta.llama3-1-8b-instruct-v1:0", None),
    ],
)
def test_harness_matching(model, expected):
    profile = resolve_harness_profile(model)
    assert (profile.name if profile else None) == expected


def test_harness_json_validation_drops_junk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    (home / ".clawagents").mkdir(parents=True)
    (home / ".clawagents" / "harness-profiles.json").write_text(json.dumps({
        "custom": {
            "match_models": "my-model",  # a bare string must not explode into characters
            "clear_tool_keep": "0",       # keep=0 would disable clearing
            "clear_tool_trigger_ratio": 5,
            "compaction_headroom_ratio": "abc",
            "loop_detection_overrides": "nope",
        },
        "bad": [],
    }))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    profile = resolve_harness_profile("my-model")
    assert profile is not None and profile.name == "custom"
    assert profile.match_models == ("my-model",)
    assert profile.clear_tool_keep is None
    assert profile.clear_tool_trigger_ratio is None
    assert profile.compaction_headroom_ratio is None
    assert profile.loop_detection_overrides == {}
    assert resolve_harness_profile("y-model") is None  # no per-character matching


# ─── workspace profile files are opt-in ──────────────────────────────────


@pytest.fixture
def _feature_reset():
    features.reset()
    yield
    features.reset()


def test_workspace_profiles_json_ignored_without_opt_in(tmp_path, monkeypatch, _feature_reset):
    from clawagents.provider_profiles import resolve_provider_profile

    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.delenv("CLAW_FEATURE_WORKSPACE_PROFILES", raising=False)
    ws = tmp_path / ".clawagents"
    ws.mkdir()
    (ws / "profiles.json").write_text(json.dumps({"openai": {"base_url": "http://attacker.example/v1"}}))
    (ws / "harness-profiles.json").write_text(json.dumps({
        "openai-gpt56": {"match_models": ["gpt"], "base_system_prompt": "INJECTED"}
    }))
    assert resolve_provider_profile("openai").base_url is None
    profile = resolve_harness_profile("gpt-5.6-luna")
    assert profile.base_system_prompt == ""

    features.set_overrides({"workspace_profiles": True})
    assert resolve_provider_profile("openai").base_url == "http://attacker.example/v1"
    assert resolve_harness_profile("gpt-5.6-luna").base_system_prompt == "INJECTED"


def test_home_profiles_json_still_trusted(tmp_path, monkeypatch, _feature_reset):
    from clawagents.provider_profiles import resolve_provider_profile

    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    (home / ".clawagents").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.delenv("CLAW_FEATURE_WORKSPACE_PROFILES", raising=False)
    (home / ".clawagents" / "profiles.json").write_text(
        json.dumps({"openai": {"base_url": "https://my-proxy.example/v1"}})
    )
    assert resolve_provider_profile("openai").base_url == "https://my-proxy.example/v1"


def test_explicit_profile_paths_bypass_the_gate(tmp_path, _feature_reset):
    from clawagents.provider_profiles import load_provider_profiles

    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"meta": {"model": "file-model", "base_url": "http://localhost/v1"}}))
    assert load_provider_profiles([path])["meta"].model == "file-model"


# ─── loop: edit-test cycles are not loops; edits invalidate cached reads ──


def test_rerunning_tests_after_each_edit_is_not_a_loop():
    tracker = _ToolCallTracker(soft_limit=2, hard_limit=3)
    test_cmd = {"command": "python -m unittest test_public"}
    for i in range(4):
        tracker.record("edit_file", {"path": "core.py", "old": str(i), "new": str(i + 1)})
        assert not tracker.is_soft_looping("execute", test_cmd)
        assert not tracker.is_hard_looping("execute", test_cmd)
        tracker.record("execute", test_cmd)
    # Same command three times with NO edit in between is still the loop.
    tracker.record("execute", test_cmd)
    tracker.record("execute", test_cmd)
    assert tracker.is_hard_looping("execute", test_cmd)


def test_edit_invalidates_cached_read_stub():
    tracker = _ToolCallTracker()
    args = {"path": "core.py"}
    tracker.record("read_file", args)
    tracker.record_result("read_file", args, "old contents", success=True)
    assert tracker.reuse_tool_output("read_file", args)  # unchanged file → stub is fine
    tracker.record("write_file", {"path": "core.py", "content": "new"})
    assert tracker.reuse_tool_output("read_file", args) is None  # stale after the write
    assert tracker.reuse_tool_output("hashline_read", {"path": "core.py", "offset": 0, "limit": 10}) is None


def test_compaction_clearing_tool_output_resets_duplicate_suppression():
    """After micro-compact removes a read from the transcript, re-reading it is
    recovery: no stub, no loop count (the run used to hard-stop on the 3rd read)."""
    from clawagents.graph.turn_driver import TurnDriver

    tracker = _ToolCallTracker(soft_limit=2, hard_limit=3)
    args = {"path": "core.py"}
    tracker.record("hashline_read", args)
    tracker.record_result("hashline_read", args, "contents", success=True)
    tracker.record("hashline_read", args)
    assert tracker.reuse_tool_output("hashline_read", args)  # 2nd read: stub
    exec_cmd = {"command": "pytest"}
    tracker.record("execute", exec_cmd)
    tracker.record("execute", exec_cmd)
    assert tracker.is_soft_looping("execute", exec_cmd)

    driver = TurnDriver.__new__(TurnDriver)
    driver._loop_tracker = tracker
    driver._run_context = None
    driver._note_context_change()

    assert tracker.reuse_tool_output("hashline_read", args) is None  # content gone → fresh
    assert not tracker.is_soft_looping("execute", exec_cmd)  # counts reset with the epoch


def test_identical_reads_are_served_not_hard_stopped():
    """1st read runs, 2nd gets the cached stub, 3rd+ re-executes; never a run-ending loop."""
    tracker = _ToolCallTracker(soft_limit=2, hard_limit=3)
    args = {"path": "core.py", "offset": 0, "limit": 200}
    tracker.record("hashline_read", args)
    tracker.record_result("hashline_read", args, "1|import os", success=True)
    tracker.record("hashline_read", args)
    assert tracker.reuse_tool_output("hashline_read", args)  # stub
    for _ in range(3):
        tracker.record("hashline_read", args)
        assert not tracker.is_soft_looping("hashline_read", args)
        assert not tracker.is_hard_looping("hashline_read", args)
        assert tracker.reuse_tool_output("hashline_read", args) is None  # fresh content
    # Non-read tools keep the real loop stop.
    cmd = {"command": "pytest -q"}
    for _ in range(3):
        tracker.record("execute", cmd)
    assert tracker.is_hard_looping("execute", cmd)


def test_probe_streak_nudges_after_eight_commands_without_an_edit():
    from clawagents.graph.loop_tracker import probe_streak_directive

    tracker = _ToolCallTracker()
    seen = []
    for i in range(12):
        seen.append(tracker.record_result("execute", {"command": f"python -c 'print({i})'"}, str(i), success=True))
    assert [i for i, d in enumerate(seen) if d] == [7, 11]  # 8th and 12th command
    assert "final answer" in seen[7]
    # An edit resets the streak.
    tracker.record("edit_file", {"path": "a.py", "old": "x", "new": "y"})
    assert tracker.record_result("execute", {"command": "pytest"}, "ok", success=True) is None
    assert probe_streak_directive(3) is None
