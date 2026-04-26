"""Unit tests for the openai-agents-python-inspired API surfaces.

Covers recommendations #1–#10 implemented on top of the existing
ClawAgents loop without hitting real LLM providers:

1.  ``RunContext`` — typed per-run context + approval store
2.  ``@function_tool`` — auto-schema from signature + docstring
3.  ``StreamEvent`` — typed dataclass events
4.  ``RetryPolicy`` — composable retry built on ``ErrorClass``
5.  ``Usage`` — per-run token accumulator
6.  ``RunHooks`` / ``AgentHooks`` — class-based lifecycle hooks
7.  ``InputGuardrail`` / ``OutputGuardrail`` — allow/reject/raise
8.  ``output_type`` — structured output coercion
9.  Per-call tool approval via ``RunContext``
10. ``Session`` protocol (``InMemorySession`` / ``JsonlFileSession`` /
    ``SQLiteSession``)

Alignment extras (ported from the TypeScript sibling package):
* ``composite_hooks`` — layer multiple ``RunHooks`` observers
* ``ErrorStreamEvent`` — canonical name matching the TS port
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from clawagents import (
    # 1
    ApprovalRecord,
    RunContext,
    # 2
    function_tool,
    # 3
    AssistantDeltaEvent,
    FinalOutputEvent,
    GuardrailTrippedEvent,
    StreamEvent,
    ToolStartedEvent,
    TurnStartedEvent,
    UsageEvent,
    stream_event_from_kind,
    # 4
    RetryPolicy,
    DEFAULT_RETRY_POLICY,
    # 5
    RequestUsage,
    Usage,
    # 6
    AgentHooks,
    RunHooks,
    # 7
    GuardrailBehavior,
    GuardrailResult,
    GuardrailTripwireTriggered,
    InputGuardrail,
    OutputGuardrail,
    input_guardrail,
    output_guardrail,
    # 10
    InMemorySession,
    JsonlFileSession,
    Session,
    SQLiteSession,
    # composite hooks helper
    composite_hooks,
    # typed stream events (new canonical name)
    ErrorStreamEvent,
)
from clawagents.errors.taxonomy import ErrorClass
from clawagents.function_tool import FunctionTool
from clawagents.providers.llm import LLMMessage
from clawagents.tools.registry import ToolResult


# ── #1  RunContext ────────────────────────────────────────────────────────


class TestRunContext:
    def test_default_construction(self):
        ctx: RunContext = RunContext()
        assert ctx.context is None
        assert isinstance(ctx.usage, Usage)
        assert ctx.usage.requests == 0
        assert ctx._approvals == {}
        assert ctx._always_approvals == {}

    def test_typed_user_context(self):
        @dataclass
        class AppCtx:
            user_id: str
            role: str

        ctx: RunContext[AppCtx] = RunContext(context=AppCtx(user_id="u1", role="admin"))
        assert ctx.context is not None
        assert ctx.context.user_id == "u1"
        assert ctx.context.role == "admin"

    def test_approve_and_reject_lookup(self):
        ctx: RunContext = RunContext()
        ctx.approve_tool("call-1", tool_name="search")
        ctx.reject_tool("call-2", tool_name="danger", reason="blocked")
        assert ctx.is_tool_approved("call-1") is True
        assert ctx.is_tool_approved("call-2") is False
        assert ctx.is_tool_approved("unknown") is None
        rejected = ctx.get_approval("call-2")
        assert rejected is not None and rejected.reason == "blocked"

    def test_always_approval_sticky(self):
        ctx: RunContext = RunContext()
        ctx.approve_tool("call-1", always=True, tool_name="search")
        assert ctx.is_tool_approved("call-new", tool_name="search") is True
        assert ctx.is_tool_approved("call-new", tool_name="other") is None

    def test_approval_record_defaults(self):
        rec = ApprovalRecord(approved=True)
        assert rec.always is False
        assert rec.reason is None


def test_session_preload_uses_bounded_default_limit():
    from clawagents.graph.agent_loop import _session_get_items

    class RecordingSession:
        def __init__(self):
            self.seen_limit = None

        async def get_items(self, limit=None):
            self.seen_limit = limit
            return [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "new"},
            ]

    session = RecordingSession()
    items = asyncio.run(_session_get_items(session, limit=200))

    assert session.seen_limit == 200
    assert [item.content for item in items] == ["old", "new"]


# ── #2  @function_tool decorator ──────────────────────────────────────────


class TestFunctionTool:
    def test_bare_decorator_sync(self):
        @function_tool
        def add(a: int, b: int = 2) -> int:
            """Add two ints.

            :param a: first addend
            :param b: second addend
            """
            return a + b

        assert isinstance(add, FunctionTool)
        assert add.name == "add"
        assert add.description == "Add two ints."
        assert add.parameters["a"]["type"] == "integer"
        assert add.parameters["a"]["required"] is True
        assert add.parameters["a"]["description"] == "first addend"
        assert add.parameters["b"]["required"] is False
        assert add.parameters["b"]["default"] == 2

    def test_decorator_with_kwargs_and_async(self):
        @function_tool(name="search_web", description="Look things up.")
        async def _search(query: str, limit: int = 5) -> str:
            return f"{query}:{limit}"

        assert _search.name == "search_web"
        assert _search.description == "Look things up."
        assert _search.is_async is True

    def test_docstring_google_args_parsing(self):
        @function_tool
        def greet(name: str, loud: bool = False) -> str:
            """Say hi.

            Args:
                name: who to greet
                loud: shout if True
            """
            return f"{'HI' if loud else 'hi'} {name}"

        assert greet.parameters["name"]["description"] == "who to greet"
        assert greet.parameters["loud"]["description"] == "shout if True"
        assert greet.parameters["loud"]["type"] == "boolean"

    def test_run_context_param_hidden_from_schema(self):
        @function_tool
        async def who(run_context, topic: str) -> str:
            uid = "anon"
            if run_context is not None and run_context.context is not None:
                uid = run_context.context.get("uid", "anon")
            return f"{uid}:{topic}"

        assert "run_context" not in who.parameters
        assert "topic" in who.parameters
        assert who.accepts_run_context is True
        assert who.context_param_name == "run_context"

    def test_execute_sync_returns_toolresult(self):
        @function_tool
        def double(n: int) -> int:
            return n * 2

        res = asyncio.run(double.execute({"n": 21}))
        assert isinstance(res, ToolResult)
        assert res.success is True
        assert res.output == "42"

    def test_execute_async_passes_run_context(self):
        @function_tool
        async def echo(run_context, msg: str) -> str:
            return f"{run_context.context['prefix']}:{msg}"

        ctx: RunContext = RunContext(context={"prefix": "ok"})
        res = asyncio.run(echo.execute({"msg": "hi"}, run_context=ctx))
        assert res.success is True
        assert res.output == "ok:hi"

    def test_execute_swallows_exception(self):
        @function_tool
        def boom() -> str:
            raise RuntimeError("fail")

        res = asyncio.run(boom.execute({}))
        assert res.success is False
        assert "fail" in (res.error or "")

    def test_optional_type_marks_not_required(self):
        from typing import Optional

        @function_tool
        def opt(a: str, b: Optional[int] = None) -> str:
            return f"{a}:{b}"

        assert opt.parameters["a"]["required"] is True
        assert opt.parameters["b"]["required"] is False


# ── #3  Typed StreamEvent dataclasses ─────────────────────────────────────


class TestStreamEvents:
    def test_turn_started_mapping(self):
        ev = stream_event_from_kind("turn_started", {"iteration": 3, "task": "t"})
        assert isinstance(ev, TurnStartedEvent)
        assert ev.iteration == 3
        assert ev.data == {"iteration": 3, "task": "t"}

    def test_usage_event_mapping(self):
        ev = stream_event_from_kind(
            "usage",
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "model": "gpt"},
        )
        assert isinstance(ev, UsageEvent)
        assert ev.input_tokens == 10
        assert ev.output_tokens == 5
        assert ev.total_tokens == 15
        assert ev.model == "gpt"

    def test_tool_started_mapping_with_extras_in_data(self):
        ev = stream_event_from_kind(
            "tool_started",
            {"tool_name": "search", "call_id": "c1", "args": {"q": "x"}, "extra": 1},
        )
        assert isinstance(ev, ToolStartedEvent)
        assert ev.tool_name == "search"
        assert ev.call_id == "c1"
        assert ev.args == {"q": "x"}
        assert ev.data["extra"] == 1

    def test_final_output_mapping(self):
        ev = stream_event_from_kind("final_output", {"output": {"k": 1}, "raw": "x"})
        assert isinstance(ev, FinalOutputEvent)
        assert ev.output == {"k": 1}
        assert ev.raw == "x"

    def test_guardrail_tripped_mapping(self):
        ev = stream_event_from_kind(
            "guardrail_tripped",
            {"guardrail_name": "pii", "where": "input", "behavior": "reject_content"},
        )
        assert isinstance(ev, GuardrailTrippedEvent)
        assert ev.guardrail_name == "pii"
        assert ev.where == "input"
        assert ev.behavior == "reject_content"

    def test_unknown_kind_falls_back_to_base(self):
        ev = stream_event_from_kind("totally_new_kind", {"x": 1})
        assert type(ev) is StreamEvent
        assert ev.kind == "totally_new_kind"
        assert ev.data == {"x": 1}

    def test_delta_has_its_own_class(self):
        ev = stream_event_from_kind("assistant_delta", {"delta": "ab"})
        assert isinstance(ev, AssistantDeltaEvent)
        assert ev.delta == "ab"


# ── #4  Composable RetryPolicy ────────────────────────────────────────────


class _RateLimit(Exception):
    """Exception whose shape triggers PROVIDER_RATE_LIMIT classification."""


class TestRetryPolicy:
    def test_defaults_retry_on_rate_limit_and_transient(self):
        p = RetryPolicy()
        assert ErrorClass.PROVIDER_RATE_LIMIT in p.retry_on
        assert ErrorClass.PROVIDER_INTERNAL in p.retry_on
        assert ErrorClass.PROVIDER_TRANSPORT in p.retry_on
        assert p.max_retries == 6

    def test_should_retry_honours_per_class_cap(self):
        p = RetryPolicy(
            max_retries=5,
            per_class_max={ErrorClass.PROVIDER_RATE_LIMIT: 2},
        )
        desc_rate = type("D", (), {"error_class": ErrorClass.PROVIDER_RATE_LIMIT})()
        assert p.should_retry(_RateLimit("x"), attempt=1, descriptor=desc_rate) is True
        assert p.should_retry(_RateLimit("x"), attempt=2, descriptor=desc_rate) is True
        assert p.should_retry(_RateLimit("x"), attempt=3, descriptor=desc_rate) is False

    def test_should_retry_false_for_unregistered_class(self):
        p = RetryPolicy(retry_on=frozenset({ErrorClass.PROVIDER_RATE_LIMIT}))
        desc_auth = type("D", (), {"error_class": ErrorClass.PROVIDER_AUTH})()
        assert p.should_retry(Exception("x"), attempt=1, descriptor=desc_auth) is False

    def test_compute_delay_uses_retry_after(self):
        p = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert p.compute_delay(1, retry_after=3.5) == 3.5
        assert p.compute_delay(1, retry_after=999) == 30.0

    def test_compute_delay_exponential(self):
        p = RetryPolicy(base_delay=1.0, max_delay=100.0, jitter=0.0)
        assert p.compute_delay(1) == 1.0
        assert p.compute_delay(2) == 2.0
        assert p.compute_delay(3) == 4.0
        assert p.compute_delay(10) == 100.0  # capped

    def test_default_policy_is_singleton_like(self):
        assert isinstance(DEFAULT_RETRY_POLICY, RetryPolicy)


# ── #5  Per-run Usage accumulator ────────────────────────────────────────


class TestUsage:
    def test_add_response_totals(self):
        u = Usage()
        u.add_response(model="m1", input_tokens=10, output_tokens=5)
        u.add_response(model="m1", input_tokens=3, output_tokens=2, total_tokens=6)
        assert u.requests == 2
        assert u.input_tokens == 13
        assert u.output_tokens == 7
        # first call auto-totals: 15; second call explicit: 6
        assert u.total_tokens == 21
        assert len(u.per_request) == 2

    def test_merge_two_runs(self):
        a = Usage()
        a.add_response(model="x", input_tokens=1, output_tokens=1)
        b = Usage()
        b.add_response(model="x", input_tokens=4, output_tokens=4)
        a.merge(b)
        assert a.requests == 2
        assert a.total_tokens == 10

    def test_to_dict_serialisable(self):
        u = Usage()
        u.add_response(model="m", input_tokens=1, output_tokens=2)
        d = u.to_dict()
        assert d["requests"] == 1
        assert isinstance(d["per_request"], list)
        assert d["per_request"][0]["model"] == "m"

    def test_request_usage_to_dict(self):
        r = RequestUsage(model="m", input_tokens=1, output_tokens=2, total_tokens=3)
        assert r.to_dict()["model"] == "m"


# ── #6  RunHooks / AgentHooks ─────────────────────────────────────────────


class TestLifecycleHooks:
    def test_runhooks_default_methods_are_noops(self):
        h = RunHooks()
        ctx: RunContext = RunContext()
        asyncio.run(h.on_run_start(ctx, "task"))
        asyncio.run(h.on_run_end(ctx, "final"))
        asyncio.run(h.on_llm_start(ctx, "m", []))
        asyncio.run(h.on_llm_end(ctx, "m", "x", None))
        asyncio.run(h.on_tool_start(ctx, "t", "c", {}))
        asyncio.run(h.on_tool_end(ctx, "t", "c", True, "ok", None))

    def test_custom_runhooks_record_calls(self):
        calls: list[tuple[str, ...]] = []

        class MyHooks(RunHooks):
            async def on_llm_start(self, ctx, model, messages):
                calls.append(("llm_start", model))

            async def on_tool_end(self, ctx, tool, cid, success, output, error):
                calls.append(("tool_end", tool, str(success), output))

        h = MyHooks()
        ctx: RunContext = RunContext()
        asyncio.run(h.on_llm_start(ctx, "gpt", [LLMMessage(role="user", content="hi")]))
        asyncio.run(h.on_tool_end(ctx, "calc", "c-1", True, "4", None))
        assert ("llm_start", "gpt") in calls
        assert ("tool_end", "calc", "True", "4") in calls

    def test_agenthooks_is_runhooks_subclass(self):
        assert issubclass(AgentHooks, RunHooks)


# ── #7  Guardrails ────────────────────────────────────────────────────────


class TestGuardrails:
    def test_result_helpers(self):
        assert GuardrailResult.allow().behavior == GuardrailBehavior.ALLOW
        rej = GuardrailResult.reject("nope", message="blocked")
        assert rej.behavior == GuardrailBehavior.REJECT_CONTENT
        assert rej.replacement_output == "nope"
        raise_ = GuardrailResult.raise_exc("die")
        assert raise_.behavior == GuardrailBehavior.RAISE_EXCEPTION

    def test_input_guardrail_decorator(self):
        @input_guardrail("pii")
        async def check(ctx, task):
            if "ssn" in task:
                return GuardrailResult.raise_exc("pii detected")
            return GuardrailResult.allow()

        assert isinstance(check, InputGuardrail)
        assert check.name == "pii"

        ctx: RunContext = RunContext()
        res = asyncio.run(check.run(ctx, "hello"))
        assert res.behavior == GuardrailBehavior.ALLOW
        res2 = asyncio.run(check.run(ctx, "my ssn"))
        assert res2.behavior == GuardrailBehavior.RAISE_EXCEPTION

    def test_output_guardrail_decorator_default_name(self):
        @output_guardrail()
        async def sanitise(ctx, output):
            return GuardrailResult.allow()

        assert isinstance(sanitise, OutputGuardrail)
        assert sanitise.name == "sanitise"

    def test_tripwire_exception_carries_details(self):
        result = GuardrailResult.raise_exc("pii detected")
        err = GuardrailTripwireTriggered("pii", "input", result)
        assert err.guardrail_name == "pii"
        assert err.where == "input"
        assert err.result is result
        assert "pii" in str(err)


# ── #8  output_type coercion ─────────────────────────────────────────────


class TestOutputTypeCoercion:
    def test_str_passthrough(self):
        from clawagents.graph.agent_loop import _coerce_output_type

        assert _coerce_output_type("hello", str) == "hello"

    def test_dict_json_parse(self):
        from clawagents.graph.agent_loop import _coerce_output_type

        assert _coerce_output_type('{"a": 1}', dict) == {"a": 1}

    def test_list_json_parse(self):
        from clawagents.graph.agent_loop import _coerce_output_type

        assert _coerce_output_type("[1, 2, 3]", list) == [1, 2, 3]

    def test_dataclass_parse(self):
        from clawagents.graph.agent_loop import _coerce_output_type

        @dataclass
        class Point:
            x: int
            y: int

        pt = _coerce_output_type('{"x": 3, "y": 4}', Point)
        assert isinstance(pt, Point)
        assert (pt.x, pt.y) == (3, 4)

    def test_invalid_json_returns_raw(self):
        from clawagents.graph.agent_loop import _coerce_output_type

        assert _coerce_output_type("not json", dict) == "not json"

    def test_pydantic_v2_model_if_installed(self):
        pydantic = pytest.importorskip("pydantic")
        from clawagents.graph.agent_loop import _coerce_output_type

        class P(pydantic.BaseModel):
            a: int
            b: str

        p = _coerce_output_type('{"a": 1, "b": "x"}', P)
        assert isinstance(p, P)
        assert p.a == 1 and p.b == "x"


# ── #9  Per-call tool approval ───────────────────────────────────────────


class TestToolApproval:
    def test_approve_then_run_decision(self):
        ctx: RunContext = RunContext()
        ctx.approve_tool("c1", tool_name="shell")
        assert ctx.is_tool_approved("c1") is True

    def test_reject_with_reason(self):
        ctx: RunContext = RunContext()
        ctx.reject_tool("c1", tool_name="shell", reason="dangerous")
        rec = ctx.get_approval("c1")
        assert rec is not None and rec.approved is False
        assert rec.reason == "dangerous"

    def test_always_approval_persists_by_tool_name(self):
        ctx: RunContext = RunContext()
        ctx.approve_tool("first", always=True, tool_name="read_file")
        # A brand-new call id should inherit the always-decision via tool_name
        assert ctx.is_tool_approved("second", tool_name="read_file") is True
        # But a different tool name must not leak the approval
        assert ctx.is_tool_approved("second", tool_name="write_file") is None

    def test_always_rejection_persists(self):
        ctx: RunContext = RunContext()
        ctx.reject_tool("first", always=True, tool_name="rm", reason="no")
        assert ctx.is_tool_approved("later", tool_name="rm") is False


# ── #10  Session protocol + backends ─────────────────────────────────────


class TestSessionBackends:
    def test_protocol_is_runtime_checkable(self):
        assert isinstance(InMemorySession("x"), Session)

    def test_in_memory_roundtrip(self):
        async def run():
            s = InMemorySession("sess")
            await s.add_items([
                LLMMessage(role="user", content="hi"),
                LLMMessage(role="assistant", content="hello"),
            ])
            items = await s.get_items()
            assert [m.role for m in items] == ["user", "assistant"]
            assert [m.content for m in items] == ["hi", "hello"]

            last = await s.pop_item()
            assert last is not None and last.content == "hello"
            assert len(await s.get_items()) == 1

            await s.clear_session()
            assert await s.get_items() == []
        asyncio.run(run())

    def test_in_memory_limit(self):
        async def run():
            s = InMemorySession("s")
            for i in range(5):
                await s.add_items([LLMMessage(role="user", content=str(i))])
            tail = await s.get_items(limit=2)
            assert [m.content for m in tail] == ["3", "4"]
        asyncio.run(run())

    def test_sqlite_roundtrip_tmpdir(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                db = Path(d) / "s.db"
                s = SQLiteSession("sess", db_path=db)
                await s.add_items([
                    LLMMessage(role="user", content="hi"),
                    LLMMessage(role="assistant", content="there"),
                ])

                # Open a *second* handle on the same DB to prove persistence.
                s2 = SQLiteSession("sess", db_path=db)
                items = await s2.get_items()
                assert [(m.role, m.content) for m in items] == [
                    ("user", "hi"),
                    ("assistant", "there"),
                ]

                popped = await s2.pop_item()
                assert popped is not None and popped.content == "there"
                remaining = await s2.get_items()
                assert len(remaining) == 1

                await s2.clear_session()
                assert await s2.get_items() == []
        asyncio.run(run())

    def test_sqlite_preserves_tool_call_metadata(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                db = Path(d) / "s.db"
                s = SQLiteSession("sess", db_path=db)
                await s.add_items([
                    LLMMessage(
                        role="tool",
                        content="42",
                        tool_call_id="call-7",
                        tool_calls_meta=[{"id": "call-7", "name": "calc"}],
                    )
                ])
                items = await s.get_items()
                assert items[0].tool_call_id == "call-7"
                assert items[0].tool_calls_meta == [{"id": "call-7", "name": "calc"}]
        asyncio.run(run())

    def test_jsonl_file_roundtrip_tmpdir(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                s = JsonlFileSession("sess", dir_path=d)
                assert s.file_path.parent == Path(d).resolve()

                await s.add_items([
                    LLMMessage(role="user", content="hi"),
                    LLMMessage(role="assistant", content="there"),
                ])

                s2 = JsonlFileSession("sess", dir_path=d)
                items = await s2.get_items()
                assert [(m.role, m.content) for m in items] == [
                    ("user", "hi"),
                    ("assistant", "there"),
                ]

                tail = await s2.get_items(limit=1)
                assert len(tail) == 1 and tail[0].content == "there"

                popped = await s2.pop_item()
                assert popped is not None and popped.content == "there"
                remaining = await s2.get_items()
                assert len(remaining) == 1 and remaining[0].content == "hi"

                await s2.clear_session()
                assert await s2.get_items() == []
        asyncio.run(run())

    def test_jsonl_file_skips_malformed_lines(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                s = JsonlFileSession("sess", file_path=Path(d) / "m.jsonl")
                await s.add_items([LLMMessage(role="user", content="hi")])
                # Corrupt the file with a broken line in the middle.
                with s.file_path.open("a", encoding="utf-8") as f:
                    f.write("not-json\n")
                await s.add_items([LLMMessage(role="assistant", content="there")])
                items = await s.get_items()
                assert [m.content for m in items] == ["hi", "there"]
        asyncio.run(run())

    def test_jsonl_file_preserves_tool_call_metadata(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                s = JsonlFileSession("sess", dir_path=d)
                await s.add_items([
                    LLMMessage(
                        role="tool",
                        content="42",
                        tool_call_id="call-7",
                        tool_calls_meta=[{"id": "call-7", "name": "calc"}],
                    )
                ])
                items = await s.get_items()
                assert items[0].tool_call_id == "call-7"
                assert items[0].tool_calls_meta == [{"id": "call-7", "name": "calc"}]
        asyncio.run(run())


# ── composite_hooks helper ────────────────────────────────────────────────


class TestCompositeHooks:
    def test_empty_composite_is_noop(self):
        async def run():
            h = composite_hooks()
            assert isinstance(h, RunHooks)
            # Calling through the no-op shouldn't raise.
            await h.on_run_start(RunContext(), "task")
            await h.on_tool_end(
                RunContext(), tool_name="x", call_id="c", success=True,
                output="ok", error=None,
            )
        asyncio.run(run())

    def test_single_hook_returned_as_is(self):
        a = RunHooks()
        assert composite_hooks(a) is a

    def test_ignores_none_entries(self):
        a = RunHooks()
        # None is treated as "no observer", single real hook is passed through.
        assert composite_hooks(None, a, None) is a

    def test_dispatches_to_all_hooks_in_order(self):
        class Recorder(RunHooks):
            def __init__(self, name):
                self.name = name
                self.calls: list[tuple[str, str]] = []

            async def on_run_start(self, context, task):
                self.calls.append(("start", task))

            async def on_tool_start(self, context, tool_name, call_id, args):
                self.calls.append(("tool_start", tool_name))

        a, b = Recorder("a"), Recorder("b")
        h = composite_hooks(a, b)

        async def run():
            ctx = RunContext()
            await h.on_run_start(ctx, "hello")
            await h.on_tool_start(ctx, tool_name="calc", call_id="c1", args={})

        asyncio.run(run())
        assert a.calls == [("start", "hello"), ("tool_start", "calc")]
        assert b.calls == [("start", "hello"), ("tool_start", "calc")]

    def test_exception_in_one_hook_doesnt_break_others(self):
        class Noisy(RunHooks):
            async def on_run_end(self, context, final_output):
                raise RuntimeError("boom")

        class Quiet(RunHooks):
            def __init__(self):
                self.seen: list[object] = []

            async def on_run_end(self, context, final_output):
                self.seen.append(final_output)

        noisy, quiet = Noisy(), Quiet()
        h = composite_hooks(noisy, quiet)

        async def run():
            await h.on_run_end(RunContext(), "done")

        asyncio.run(run())
        assert quiet.seen == ["done"]


# ── ErrorStreamEvent alias ────────────────────────────────────────────────


class TestErrorStreamEvent:
    def test_canonical_name_matches_ts(self):
        ev = ErrorStreamEvent(error="boom", recoverable=False)
        assert ev.kind == "error"
        assert ev.error == "boom"

    def test_backcompat_alias_still_points_to_same_class(self):
        # Callers from 6.1.x still using ``ErrorEvent`` should not break —
        # the alias must resolve to the same class object.
        from clawagents.stream_events import ErrorEvent as LegacyErrorEvent
        assert LegacyErrorEvent is ErrorStreamEvent


# ── integration: exported symbols present on package ─────────────────────


def test_top_level_package_reexports_all_surfaces():
    import clawagents

    expected = [
        "RunContext", "ApprovalRecord",
        "Usage", "RequestUsage",
        "RunHooks", "AgentHooks", "composite_hooks",
        "InputGuardrail", "OutputGuardrail",
        "GuardrailBehavior", "GuardrailResult", "GuardrailTripwireTriggered",
        "input_guardrail", "output_guardrail",
        "StreamEvent", "TurnStartedEvent", "AssistantTextEvent",
        "AssistantDeltaEvent", "ToolCallPlannedEvent", "ToolStartedEvent",
        "ToolResultEvent", "ApprovalRequiredEvent", "UsageEvent",
        "GuardrailTrippedEvent", "FinalOutputEvent",
        "ErrorStreamEvent", "ErrorEvent",
        "stream_event_from_kind",
        "function_tool",
        "RetryPolicy", "DEFAULT_RETRY_POLICY",
        "Session", "InMemorySession", "JsonlFileSession", "SQLiteSession",
    ]
    for name in expected:
        assert hasattr(clawagents, name), f"missing re-export: {name}"
