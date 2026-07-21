"""Port of clawagents/src/performance-regressions.test.ts."""

from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from clawagents.config.config import EngineConfig
from clawagents.graph.agent_loop import IncrementalTokenLedger, run_agent_graph
from clawagents.providers.llm import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    OpenAIProvider,
    _hashed_session_key,
    _openai_affinity,
)
from clawagents.sandbox.local import LocalBackend
from clawagents.tool_output_artifacts import load_tool_artifact
from clawagents.tools.exec import ExecTool, _ToolProgressEmitter
from clawagents.tools.registry import ToolRegistry
from clawagents.utils.bounded_output import BoundedTextAccumulator


def test_openai_prompt_cache_affinity_stable_opaque_key() -> None:
    calls: list[dict[str, Any]] = []

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None)
                    )
                ],
                usage=SimpleNamespace(
                    total_tokens=15,
                    prompt_tokens=10,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=7),
                ),
            )

    class _Chat:
        completions = _Completions()

    cfg = EngineConfig(
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
        openai_base_url="https://api.openai.com/v1",
        max_tokens=128,
        temperature=1.0,
        openai_wire_api="chat_completions",
    )

    provider = OpenAIProvider(cfg)
    provider.client = SimpleNamespace(chat=_Chat())
    provider._force_chat_completions = True
    provider._wire_api = "chat_completions"

    session_id = "customer/session/with sensitive path"

    async def _run() -> LLMResponse:
        first = await provider.chat(
            [LLMMessage(role="user", content="one")],
            session_id=session_id,
        )
        await provider.chat(
            [LLMMessage(role="user", content="two")],
            session_id=session_id,
        )
        return first

    first = asyncio.run(_run())

    assert len(calls) == 2
    first_key = calls[0]["prompt_cache_key"]
    assert isinstance(first_key, str)
    assert first_key == calls[1]["prompt_cache_key"]
    assert first_key != session_id
    assert first_key == _hashed_session_key(session_id)
    assert calls[0]["extra_headers"]["session_id"] == first_key
    assert calls[0]["extra_headers"]["x-client-request-id"] == first_key
    assert first.cache_read_tokens == 7

    affinity = _openai_affinity("https://api.openai.com/v1", session_id)
    assert affinity["prompt_cache_key"] == first_key


def test_incremental_token_ledger_counts_only_appended_messages() -> None:
    calls: list[int] = []

    def estimate(messages: list[Any]) -> int:
        calls.append(len(messages))
        return sum(len(getattr(m, "content", "") or "") for m in messages)

    initial = [LLMMessage(role="system", content="large-prefix")]
    ledger = IncrementalTokenLedger(estimate)
    ledger.rebase(initial, 100)

    next_msgs = [*initial, LLMMessage(role="user", content="new")]
    assert ledger.estimate(next_msgs) == 103
    assert calls == [1]

    ledger.record_provider_usage(next_msgs, 80)
    assert ledger.estimate(
        [*next_msgs, LLMMessage(role="assistant", content="tail")]
    ) == 84
    assert calls == [1, 1]


def test_performance_telemetry_ttft_cache_and_peak_memory() -> None:
    class TelemetryLLM(LLMProvider):
        name = "telemetry-test"

        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None, **kwargs):
            # Tool-call-only streams have no text chunks; TTFT must not
            # depend exclusively on on_chunk.
            cb = kwargs.get("on_first_token")
            if cb is not None:
                cb()
            return LLMResponse(
                content="hello",
                model="telemetry-test",
                tokens_used=15,
                prompt_tokens=10,
                cache_read_tokens=6,
            )

    state = asyncio.run(
        run_agent_graph(
            "hello",
            TelemetryLLM(),
            tools=ToolRegistry(),
            max_iterations=1,
            streaming=True,
            context_window=10_000,
            on_event=lambda *_a, **_k: None,
            use_native_tools=False,
            session_end_tail=False,
        )
    )

    assert state.usage is not None
    assert state.usage.prompt_tokens == 10
    assert state.usage.input_tokens == 4
    assert state.usage.cached_input_tokens == 6
    assert state.usage.per_request
    assert state.usage.per_request[0].time_to_first_token_ms is not None
    assert (state.usage.peak_memory_bytes or 0) > 0


def test_bounded_text_accumulator_keeps_head_and_tail() -> None:
    output = BoundedTextAccumulator(20)
    output.append("abcdefghij")
    output.append("klmnopqrstuvwxyz")
    result = str(output)

    assert result.startswith("abcdefghij")
    assert result.endswith("qrstuvwxyz")
    assert "truncated 6 chars" in result
    assert output.total_chars == 26


def test_local_backend_bounds_memory_and_spools_complete_output() -> None:
    async def _run() -> tuple[Any, str]:
        with tempfile.TemporaryDirectory(prefix="claw-bounded-exec-") as root:
            backend = LocalBackend(root)
            py = sys.executable
            result = await backend.exec(
                f'{py} -c "print(\'a\' * 200000, end=\'\')"',
                max_output_chars=1_000,
            )
            assert result.stdout_path is not None
            path = Path(result.stdout_path)
            full = path.read_text(encoding="utf-8")
            path.unlink()
            return result, full

    result, full = asyncio.run(_run())
    assert result.exit_code == 0
    assert len(result.stdout) < 1_200
    assert "truncated 199000 chars" in result.stdout
    assert full == "a" * 200_000


def test_execute_archives_complete_spilled_output(monkeypatch) -> None:
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_AUTO_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_BACKGROUND", "1")
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_STREAMING", "1")

    async def _run(root: str) -> Any:
        backend = LocalBackend(root)
        return await ExecTool(backend).execute(
            {
                "command": f'{sys.executable} -c "print(\'z\' * 50000, end=\'\')"',
                "timeout": 10_000,
            }
        )

    with tempfile.TemporaryDirectory(prefix="claw-exec-artifact-") as root:
        result = asyncio.run(_run(root))
        assert result.success
        match = re.search(r"archived id=([^;]+);", str(result.output))
        assert match is not None
        ok, full, meta = load_tool_artifact(match.group(1), workspace=root)
        assert ok
        assert full == "z" * 50_000
        assert meta and meta["complete_command_output"] is True


def test_tool_progress_updates_are_throttled_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("CLAW_FEATURE_EXECUTE_STREAMING", "1")
    events: list[tuple[str, dict[str, Any]]] = []
    context = SimpleNamespace(on_event=lambda kind, data: events.append((kind, data)))

    async def _run() -> None:
        emitter = _ToolProgressEmitter(context, interval_s=10.0)
        for _ in range(100):
            emitter.feed("stdout", "x" * 100)
        emitter.flush()

    asyncio.run(_run())
    assert 1 <= len(events) <= 2
    assert all(kind == "tool_progress" for kind, _ in events)
    assert all(len(data["delta"]) <= 2_000 for _, data in events)
    assert events[-1][1]["total_bytes"] == 10_000


def test_openai_stream_affinity_and_ttft_are_request_local(monkeypatch) -> None:
    import clawagents.providers.llm as llm_module

    arrivals = 0
    release = asyncio.Event()

    async def _barrier(_breaker: Any) -> None:
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            release.set()
        await release.wait()

    monkeypatch.setattr(llm_module, "_admit_stream_breaker", _barrier)
    calls: list[dict[str, Any]] = []

    class _Stream:
        def __init__(self) -> None:
            self._items = iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="x", tool_calls=None))],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[],
                        usage=SimpleNamespace(
                            total_tokens=2,
                            prompt_tokens=1,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                        ),
                    ),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def close(self) -> None:
            return None

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return _Stream()

    cfg = EngineConfig(
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
        openai_base_url="https://api.openai.com/v1",
        max_tokens=128,
        temperature=1.0,
        openai_wire_api="chat_completions",
    )
    provider = OpenAIProvider(cfg)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    provider._force_chat_completions = True
    provider._wire_api = "chat_completions"
    first_tokens: list[str] = []

    async def _run() -> None:
        await asyncio.gather(
            provider.chat(
                [LLMMessage(role="user", content="a")],
                on_chunk=lambda _chunk: None,
                session_id="session-a",
                on_first_token=lambda: first_tokens.append("a"),
            ),
            provider.chat(
                [LLMMessage(role="user", content="b")],
                on_chunk=lambda _chunk: None,
                session_id="session-b",
                on_first_token=lambda: first_tokens.append("b"),
            ),
        )

    asyncio.run(_run())
    keys = {call["prompt_cache_key"] for call in calls}
    assert keys == {_hashed_session_key("session-a"), _hashed_session_key("session-b")}
    assert sorted(first_tokens) == ["a", "b"]
