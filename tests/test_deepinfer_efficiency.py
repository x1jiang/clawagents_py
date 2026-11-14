"""Tests for DeepInfer-inspired inference efficiency mechanisms in clawagents_py:
1. Step-quantized prefix-cache-aligned compaction watermarks
2. DeferredStepFlush post-step bookkeeping overlap
3. Steady-state prefill (TTFT) vs. decode throughput (tok/s) telemetry
4. Grammar-constrained structured control-plane calls
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clawagents.graph.context_management import (
    _MICRO_COMPACT_STEP_MESSAGES,
    should_advance_compaction_watermark,
)
from clawagents.graph.deferred_flush import DeferredStepFlush
from clawagents.graph.run_runtime import HookDispatcher, RunEvents
from clawagents.graph.turn_driver import TurnDriver
from clawagents.graph.turn_llm import TurnLLMCaller
from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse
from clawagents.run_context import RunContext
from clawagents.session.persistence import SessionReader, SessionWriter
from clawagents.structured_output import (
    choice_json_schema,
    invoke_structured_control,
    local_engine_structured_args,
    verdict_json_schema,
)
from clawagents.tools.registry import ToolRegistry
from clawagents.usage import Usage


def test_step_quantized_watermark_preserves_prefix_across_step_window(monkeypatch):
    monkeypatch.setattr("clawagents.config.features.is_enabled", lambda _: True)
    assert should_advance_compaction_watermark(
        10, 0, step_messages=6, current_tokens=5_000, urgent_ceiling=8_000
    )
    assert not should_advance_compaction_watermark(
        12, 10, step_messages=6, current_tokens=5_500, urgent_ceiling=8_000
    )
    assert not should_advance_compaction_watermark(
        14, 10, step_messages=6, current_tokens=6_000, urgent_ceiling=8_000
    )
    assert should_advance_compaction_watermark(
        16, 10, step_messages=6, current_tokens=6_200, urgent_ceiling=8_000
    )
    # Urgent ceiling always overrides step holdback immediately!
    assert should_advance_compaction_watermark(
        12, 10, step_messages=6, current_tokens=8_100, urgent_ceiling=8_000
    )

    rc = RunContext()
    context_changes = 0

    def _note_change():
        nonlocal context_changes
        context_changes += 1

    driver = SimpleNamespace(
        _resolved_model_name="unknown",
        _context_window=10_000,
        _run_context=rc,
        _rebase_ledger=lambda _m: 4_500,
        _note_context_change=_note_change,
    )

    def _make_turn(idx: int) -> list[LLMMessage]:
        return [
            LLMMessage(
                role="assistant",
                content="",
                tool_calls_meta=[{"id": f"tc-{idx}", "name": "read_file"}],
            ),
            LLMMessage(
                role="tool",
                content=f"file content {idx} " * 120,
                tool_call_id=f"tc-{idx}",
            ),
        ]

    msgs: list[LLMMessage] = [LLMMessage(role="system", content="sys")]
    for i in range(5):
        msgs.extend(_make_turn(i))

    # Turn 1 above 40% ratio (4,500 > 4,000, urgent ceiling = 7,500): initial pass runs!
    msgs, _ = TurnDriver._micro_compact(driver, msgs, 4_500)
    assert context_changes == 1

    # Next 2 tool rounds (+2 messages, then +4 messages < _MICRO_COMPACT_STEP_MESSAGES=6):
    # Prefix must stay 100% untouched so provider KV cache hits!
    msgs_after_r1 = [*msgs, *_make_turn(5)]
    out_r1, _ = TurnDriver._micro_compact(driver, msgs_after_r1, 4_800)
    assert out_r1 is msgs_after_r1
    assert context_changes == 1

    msgs_after_r2 = [*out_r1, *_make_turn(6)]
    out_r2, _ = TurnDriver._micro_compact(driver, msgs_after_r2, 5_100)
    assert out_r2 is msgs_after_r2
    assert context_changes == 1

    # Round +3 (+6 messages == _MICRO_COMPACT_STEP_MESSAGES): batch-advances watermark!
    msgs_after_r3 = [*out_r2, *_make_turn(7)]
    assert len(msgs_after_r3) - len(msgs) == _MICRO_COMPACT_STEP_MESSAGES
    out_r3, _ = TurnDriver._micro_compact(driver, msgs_after_r3, 5_400)
    assert out_r3 is not msgs_after_r3
    assert context_changes == 2

    # Crossing urgent ceiling (e.g., 7,600 >= 7,500) fires immediately even after 1 round!
    msgs_urgent = [*out_r3, *_make_turn(8)]
    out_urgent, _ = TurnDriver._micro_compact(driver, msgs_urgent, 7_600)
    assert out_urgent is not msgs_urgent
    assert context_changes == 3


@pytest.mark.asyncio
async def test_deferred_step_flush_overlaps_and_drains_cleanly(tmp_path: Path):
    flush = DeferredStepFlush()
    order: list[str] = []

    def sync_work():
        order.append("sync_ok")

    def broken_work():
        raise RuntimeError("broken sink must be isolated")

    async def async_work():
        await asyncio.sleep(0.01)
        order.append("async_ok")

    flush.defer(sync_work)
    flush.defer(broken_work)
    flush.defer(async_work)
    assert flush.pending_count == 3

    flush.kick()
    await flush.drain()
    assert order == ["sync_ok", "async_ok"]
    assert flush.flushed_count == 2
    assert flush.error_count == 1
    assert flush.pending_count == 0

    # Verify TurnLLMCaller queues usage writes onto DeferredStepFlush and drain persists them
    writer = SessionWriter(session_id="deferred-test", session_dir=tmp_path)
    events = RunEvents(lambda *_a, **_k: None, None)
    rc = RunContext()
    hooks = HookDispatcher([], rc, events)

    class FastLLM:
        model = "gpt-5.4-mini"

        async def chat(self, messages, **kwargs):
            on_first = kwargs.get("on_first_token")
            if callable(on_first):
                on_first()
            await asyncio.sleep(0.005)
            return LLMResponse(
                content="done",
                model=self.model,
                tokens_used=150,
                prompt_tokens=100,
                cache_read_tokens=80,
            )

    turn_flush = DeferredStepFlush()
    caller = TurnLLMCaller(
        llm=FastLLM(),
        events=events,
        hooks=hooks,
        registry=ToolRegistry(),
        session_writer=writer,
        external_hooks=None,
        accumulate_usage=lambda resp, **kw: rc.usage.add_response(
            model=resp.model or "",
            prompt_tokens=resp.prompt_tokens,
            input_tokens=20,
            output_tokens=50,
            cached_input_tokens=resp.cache_read_tokens,
            time_to_first_token_ms=kw.get("time_to_first_token_ms"),
            elapsed_ms=kw.get("elapsed_ms"),
        ),
        deferred_flush=turn_flush,
    )
    res = await caller.call(
        [LLMMessage(role="user", content="hi")],
        resolved_model_name="gpt-5.4-mini",
        use_native_tools=False,
        tools_supplied=False,
        initial_schemas=None,
        handoffs=[],
        streaming=True,
        cancel_event=None,
        run_context=rc,
        output_type=None,
    )
    assert res.elapsed_ms is not None and res.elapsed_ms > 0
    assert turn_flush.pending_count == 1
    await turn_flush.drain()
    reader = SessionReader(writer.path)
    assert any(ev.get("type") == "usage" and ev.get("tokens_used") == 150 for ev in reader.events)


def test_usage_steady_state_decode_tps_and_cache_hit_telemetry():
    u = Usage()
    # Warmup call #0: slow cold start (20 tokens in 500ms decode -> 40 tok/s)
    r0 = u.add_response(
        model="gpt-5.4",
        prompt_tokens=1_000,
        input_tokens=1_000,
        output_tokens=20,
        cached_input_tokens=0,
        time_to_first_token_ms=300.0,
        elapsed_ms=800.0,
    )
    assert r0.decode_ms == 500.0
    assert r0.decode_tokens_per_sec == 40.0
    # With only 1 call, steady_decode_tps falls back to call #0
    assert u.steady_decode_tps == 40.0

    # Call #1 and #2: warm prefix cache + fast steady decode (100 tok in 500ms -> 200 tok/s)
    u.add_response(
        model="gpt-5.4",
        prompt_tokens=1_200,
        input_tokens=200,
        output_tokens=100,
        cached_input_tokens=1_000,
        time_to_first_token_ms=80.0,
        elapsed_ms=580.0,
    )
    u.add_response(
        model="gpt-5.4",
        prompt_tokens=1_400,
        input_tokens=200,
        output_tokens=100,
        cached_input_tokens=1_200,
        time_to_first_token_ms=70.0,
        elapsed_ms=570.0,
    )
    # Warmup call #0 is excluded once >=2 calls exist -> steady_decode_tps == 200.0 tok/s!
    assert u.steady_decode_tps == 200.0
    assert u.avg_ttft_ms == 150.0
    assert u.max_step_ms == 800.0
    assert u.cache_hit_ratio == round(2200 / 3600, 4)

    serialized = u.to_dict()
    assert serialized["steady_decode_tps"] == 200.0
    assert serialized["avg_ttft_ms"] == 150.0
    assert serialized["max_step_ms"] == 800.0
    assert serialized["cache_hit_ratio"] == u.cache_hit_ratio


@pytest.mark.asyncio
async def test_structured_control_schema_binding_fallback_and_local_engine_args():
    schema = choice_json_schema(["A", "B"])
    di_args = local_engine_structured_args("json_schema", schema, engine="deepinfer")
    assert di_args["so_args"]["type"] == "json_schema"
    assert '"enum": ["A", "B"]' in di_args["so_args"]["schema"]

    vllm_args = local_engine_structured_args("regex", r"[AB]", engine="vllm")
    assert vllm_args == {"guided_regex": "[AB]"}

    sgl_args = local_engine_structured_args("grammar", "root ::= 'A' | 'B'", engine="sglang")
    assert sgl_args == {"ebnf": "root ::= 'A' | 'B'"}

    # Provider that supports _structured_json_schema sees schema during call and has prior value restored
    observed_schemas: list[Any] = []

    class SchemaAwareLLM:
        _structured_json_schema: Any = {"prior": True}

        async def chat(self, messages, **kwargs):
            observed_schemas.append(self._structured_json_schema)
            return LLMResponse(content='{"ok": true, "reason": "clean"}', model="stub", tokens_used=5)

    llm = SchemaAwareLLM()
    parsed, _ = await invoke_structured_control(
        llm,
        [LLMMessage(role="user", content="check")],
        schema=verdict_json_schema(),
        parse_fn=lambda s: s,
    )
    assert '"ok": true' in parsed
    assert observed_schemas == [verdict_json_schema()]
    assert llm._structured_json_schema == {"prior": True}

    # Provider endpoint that rejects `response_format` falls back cleanly on retry
    attempts = 0

    class LegacyGatewayLLM:
        async def chat(self, messages, **kwargs):
            nonlocal attempts
            attempts += 1
            if getattr(self, "_structured_json_schema", None) is not None:
                raise RuntimeError("400 Bad Request: response_format json_schema is not supported")
            return LLMResponse(
                content='{"ok": false, "reason": "fallback worked"}',
                model="stub",
                tokens_used=5,
            )

    legacy = LegacyGatewayLLM()
    parsed_fb, _ = await invoke_structured_control(
        legacy,
        [LLMMessage(role="user", content="check")],
        schema=verdict_json_schema(),
        parse_fn=lambda s: s,
    )
    assert attempts == 2
    assert "fallback worked" in parsed_fb
    assert not hasattr(legacy, "_structured_json_schema")


@pytest.mark.asyncio
async def test_background_memory_uses_structured_control():
    from clawagents.trajectory.background_memory import extract_background_memories

    seen_schemas: list[Any] = []

    class MemLLM:
        _structured_json_schema: Any = None

        async def chat(self, messages, **kwargs):
            seen_schemas.append(self._structured_json_schema)
            return LLMResponse(
                content=json.dumps({
                    "memories": [
                        {"type": "fact", "content": "Project uses pytest", "confidence": 0.9},
                        {"type": "fact", "content": "Low confidence guess", "confidence": 0.4},
                    ]
                }),
                model="stub",
                tokens_used=20,
            )

    msgs = [LLMMessage(role="user", content="Run pytest"), LLMMessage(role="assistant", content="Done")]
    mems = await extract_background_memories(MemLLM(), msgs, 0, 2)
    assert len(seen_schemas) == 1 and seen_schemas[0]["type"] == "object"
    assert len(mems) == 1 and mems[0]["content"] == "Project uses pytest"


@pytest.mark.asyncio
async def test_structured_control_concurrent_calls_do_not_leak_schema():
    seen: list[Any] = []

    class SharedMinimalStubLLM:
        async def chat(self, messages):  # minimal 1-arg signature (no **kwargs)
            seen.append(getattr(self, "_structured_json_schema", None))
            await asyncio.sleep(0.01)
            return LLMResponse(content='{"ok": true, "reason": "clean"}', model="stub", tokens_used=5)

    shared = SharedMinimalStubLLM()
    s1 = {"type": "object", "title": "s1"}
    s2 = {"type": "object", "title": "s2"}
    r1, r2 = await asyncio.gather(
        invoke_structured_control(shared, [LLMMessage(role="user", content="1")], schema=s1, parse_fn=json.loads),
        invoke_structured_control(shared, [LLMMessage(role="user", content="2")], schema=s2, parse_fn=json.loads),
    )
    assert r1[0]["ok"] is True and r2[0]["ok"] is True
    assert {item["title"] for item in seen} == {"s1", "s2"}
    assert not hasattr(shared, "_structured_json_schema")


@pytest.mark.asyncio
async def test_background_job_cleans_subprocess_transport_on_exit():
    import sys
    from clawagents.background import BackgroundJobManager

    mgr = BackgroundJobManager()
    job = await mgr.start([sys.executable, "-c", "print('ok')"])
    done = await mgr.await_complete(job.id, timeout=10.0)
    assert done.exit_code == 0
    assert done.running is False
    assert done._process is None


@pytest.mark.asyncio
async def test_fallback_provider_locks_structured_schema_concurrently():
    from clawagents.providers.fallback import FallbackProvider

    observed: list[Any] = []

    class SharedPrimary(LLMProvider):
        name = "primary"
        model = "stub"

        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None, **_kwargs):
            observed.append(getattr(self, "_structured_json_schema", None))
            await asyncio.sleep(0.015)
            return LLMResponse(content='{"ok": true}', model="stub", tokens_used=5)

    fb = FallbackProvider(SharedPrimary(), [])
    r1, r2 = await asyncio.gather(
        invoke_structured_control(fb, [LLMMessage(role="user", content="a")], schema={"type": "object", "title": "one"}, parse_fn=json.loads),
        invoke_structured_control(fb, [LLMMessage(role="user", content="b")], schema={"type": "object", "title": "two"}, parse_fn=json.loads),
    )
    assert r1[0]["ok"] is True and r2[0]["ok"] is True
    assert {item["title"] for item in observed} == {"one", "two"}
    assert not hasattr(fb.primary, "_structured_json_schema")

    # Also ensure two distinct FallbackProviders sharing one primary serialize via _get_llm_control_lock
    observed.clear()
    shared_primary = SharedPrimary()
    fb_a = FallbackProvider(shared_primary, [])
    fb_b = FallbackProvider(shared_primary, [])
    setattr(fb_a, "_structured_json_schema", {"type": "object", "title": "alpha"})
    setattr(fb_b, "_structured_json_schema", {"type": "object", "title": "beta"})
    await asyncio.gather(
        fb_a.chat([LLMMessage(role="user", content="a")]),
        fb_b.chat([LLMMessage(role="user", content="b")]),
    )
    assert {item["title"] for item in observed} == {"alpha", "beta"}
    assert not hasattr(shared_primary, "_structured_json_schema")


@pytest.mark.asyncio
async def test_maybe_extract_memories_writes_to_workspace_with_full_tail(tmp_path: Path):
    from clawagents.config.features import set_overrides
    from clawagents.trajectory.background_memory import maybe_extract_memories

    captured_prompts: list[str] = []

    class CapturingLLM:
        async def chat(self, messages, **_kwargs):
            captured_prompts.append(str(messages[-1].content))
            return LLMResponse(
                content=json.dumps({
                    "memories": [
                        {"type": "project", "content": "Late tool output fact", "confidence": 0.92}
                    ]
                }),
                model="stub",
                tokens_used=10,
            )

    set_overrides({"background_memory": True})
    try:
        # Build > 12 messages across 10 rounds (simulating multi-tool turns)
        msgs: list[LLMMessage] = []
        for i in range(10):
            msgs.append(LLMMessage(role="user", content=f"user-{i}"))
            msgs.append(LLMMessage(role="assistant", content=f"assistant-{i}"))
            msgs.append(LLMMessage(role="tool", content=f"tool-detail-{i}-sentinel"))

        next_turn = await maybe_extract_memories(
            CapturingLLM(),
            msgs,
            round_idx=10,
            last_extraction_turn=5,
            workspace=tmp_path,
        )
        assert next_turn == 10
        assert captured_prompts
        assert "tool-detail-9-sentinel" in captured_prompts[0]
        extracted_file = tmp_path / ".clawagents" / "memories" / "extracted.md"
        assert extracted_file.is_file()
        assert "Late tool output fact" in extracted_file.read_text(encoding="utf-8")
    finally:
        set_overrides({})
