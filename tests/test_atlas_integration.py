"""Unit tests for ATLAS × ClawAgents integration (stubbed atlas_runtime)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from clawagents.graph.agent_loop import run_agent_graph
from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse


# ── Stub atlas_runtime / finding before importing clawagents.atlas ──────


class _FakeSession:
    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.delivery = types.SimpleNamespace(
            taxonomy_id="MAST",
            taxonomy={
                "codes": [
                    {
                        "id": "MAST-3",
                        "name": "Step repetition",
                        "description": "Retrying without new info",
                    }
                ]
            },
            runtime_protocol="ATLAS runtime protocol (stub).",
            dashboard_url=None,
        )
        self.workspace = MagicMock()
        self._ended = False
        self._pending_traces: list[Any] = []


_SESSIONS: list[_FakeSession] = []
_RECORDED: list[Any] = []
_ENDED: list[Any] = []
_REFLECTIONS: list[Any] = []


class _ScriptedLLM(LLMProvider):
    name = "atlas-test"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
        index = min(self._index, len(self._responses) - 1)
        self._index += 1
        content = self._responses[index]
        return LLMResponse(content=content, model=self.name, tokens_used=1)


def _install_atlas_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    atlas_runtime = types.ModuleType("atlas_runtime")
    finding = types.ModuleType("finding")
    finding_resolver = types.ModuleType("finding.resolver")
    finding_resolver.ABSENT = object()
    finding_resolver.NONE = "none"

    def start_session(inherit=None, **kwargs):
        s = _FakeSession()
        _SESSIONS.append(s)
        return s

    def load_atlas_config(path=None):
        if path is None:
            return {}
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        import json

        return json.loads(p.read_text())

    def render_reflection_prompt(**kwargs):
        return (
            f"ATLAS reflection for {kwargs.get('gate_label')} "
            f"checkpoint={kwargs.get('checkpoint_id')}\n"
            "Observe:\nCorrelate:\nMap:\nDecide:\n"
        )

    def render_format_repair(**kwargs):
        return f"ATLAS format repair for {kwargs.get('checkpoint_id')}"

    class _Harvest:
        def __init__(self, result=None, partial=None, error=None):
            self.result = result
            self.partial = partial
            self.error = error

    def harvest_reflection(text, *, checkpoint_id, known_code_ids):
        if "Observe:" in text and "Decide:" in text:
            return _Harvest(
                result=types.SimpleNamespace(
                    checkpoint_id=checkpoint_id,
                    codes=[],
                    text=text,
                )
            )
        return _Harvest(error="missing Observe/Decide")

    def pre_submission(session, gate_text):
        allow = "READY_TO_SUBMIT" in gate_text or "none apply" in gate_text.lower()
        return types.SimpleNamespace(allow=allow, reason="needs repair" if not allow else "ok")

    def pin_gate_decision(decision, pinned_status, max_retries=3):
        return decision, False

    def record_reflection(trace_output, meta, reflection, *, gate, task_id):
        _REFLECTIONS.append({"gate": gate, "task_id": task_id, "meta": meta})

    def record_trace(session, trace):
        _RECORDED.append(trace)
        session._pending_traces.append(trace)

    def end_session(session):
        session._ended = True
        _ENDED.append(session)
        return types.SimpleNamespace(persisted_traces=1)

    def redact_trace(trace):
        return trace

    class GenerationTrace:
        def __init__(self, problem_id, task, raw_trajectory, metadata=None):
            self.problem_id = problem_id
            self.task = task
            self.raw_trajectory = raw_trajectory
            self.metadata = metadata or {}

    atlas_runtime.start_session = start_session
    atlas_runtime.load_atlas_config = load_atlas_config
    atlas_runtime.render_reflection_prompt = render_reflection_prompt
    atlas_runtime.render_format_repair = render_format_repair
    atlas_runtime.harvest_reflection = harvest_reflection
    atlas_runtime.pre_submission = pre_submission
    atlas_runtime.pin_gate_decision = pin_gate_decision
    atlas_runtime.record_reflection = record_reflection
    atlas_runtime.record_trace = record_trace
    atlas_runtime.end_session = end_session
    atlas_runtime.redact_trace = redact_trace
    atlas_runtime.GenerationTrace = GenerationTrace

    finding.resolver = finding_resolver

    monkeypatch.setitem(sys.modules, "atlas_runtime", atlas_runtime)
    monkeypatch.setitem(sys.modules, "finding", finding)
    monkeypatch.setitem(sys.modules, "finding.resolver", finding_resolver)

    # Drop cached clawagents.atlas modules so they pick up stubs.
    for key in list(sys.modules):
        if key == "clawagents.atlas" or key.startswith("clawagents.atlas."):
            del sys.modules[key]


@pytest.fixture
def atlas_stubs(monkeypatch):
    _SESSIONS.clear()
    _RECORDED.clear()
    _ENDED.clear()
    _REFLECTIONS.clear()
    _install_atlas_stubs(monkeypatch)
    yield


def test_atlas_disabled_does_not_import(monkeypatch):
    from clawagents.atlas.config import resolve_atlas_enabled
    from clawagents.atlas.adapter import AtlasAdapter

    monkeypatch.delenv("CLAW_ATLAS", raising=False)
    assert resolve_atlas_enabled(None) is False
    assert AtlasAdapter.maybe_create(atlas=False) is None
    assert AtlasAdapter.maybe_create(atlas=None) is None


def test_missing_package_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "atlas_runtime", None)  # type: ignore[assignment]
    # Force import failure
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "atlas_runtime" or name.startswith("atlas_runtime."):
            raise ImportError("No module named atlas_runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for key in list(sys.modules):
        if key == "clawagents.atlas" or key.startswith("clawagents.atlas."):
            del sys.modules[key]

    from clawagents.atlas._runtime import require_atlas_runtime

    with pytest.raises(ImportError, match="atlas-skill"):
        require_atlas_runtime()


def test_start_injects_protocol_not_taxonomy(atlas_stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.atlas.adapter import AtlasAdapter

    adapter = AtlasAdapter(config={
        "trace_output": tmp_path / "atlas-program",
        "repair_rounds": 3,
        "format_retries": 2,
        "failure_throttle_calls": 5,
        "failure_recency_seconds": 30,
        "recent_activity_messages": 8,
        "recent_activity_chars": 12000,
        "redact_traces": True,
        "gate_exhaustion_policy": "release",
        "dashboard": False,
    })
    run = adapter.start("do the thing")
    protocol = adapter.protocol_for_system()
    assert "ATLAS runtime protocol" in protocol
    assert "MAST-3" not in protocol  # taxonomy codes must not dump into ordinary context
    assert run.taxonomy_id == "MAST"
    assert len(_SESSIONS) == 1


def test_tool_failure_triggers_advisory_once(atlas_stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.atlas.adapter import AtlasAdapter

    adapter = AtlasAdapter(config={
        "trace_output": tmp_path / "atlas-program",
        "repair_rounds": 3,
        "format_retries": 2,
        "failure_throttle_calls": 5,
        "failure_recency_seconds": 30,
        "recent_activity_messages": 8,
        "recent_activity_chars": 12000,
        "redact_traces": True,
        "gate_exhaustion_policy": "release",
        "dashboard": False,
    })
    adapter.start("task")
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="task"),
    ]
    action = adapter.on_tool_failure(messages, tool_name="execute", error="boom")
    assert action.inject is not None
    assert "ATLAS reflection" in action.inject
    assert action.skip_rethink is True

    # Throttled: immediate second failure should not inject again
    action2 = adapter.on_tool_failure(messages, tool_name="execute", error="boom2")
    assert action2.inject is None


def test_final_gate_blocks_then_allows(atlas_stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.atlas.adapter import AtlasAdapter

    adapter = AtlasAdapter(config={
        "trace_output": tmp_path / "atlas-program",
        "repair_rounds": 3,
        "format_retries": 2,
        "failure_throttle_calls": 5,
        "failure_recency_seconds": 30,
        "recent_activity_messages": 8,
        "recent_activity_chars": 12000,
        "redact_traces": True,
        "gate_exhaustion_policy": "release",
        "dashboard": False,
    })
    adapter.start("task")
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="task"),
        LLMMessage(role="assistant", content="proposed answer"),
    ]
    gate = adapter.begin_final_gate(messages)
    assert gate.continue_loop is True
    assert gate.inject is not None

    # Bad reflection → format retry or give_up depending on retries
    bad = adapter.process_assistant_text(messages, "not a reflection")
    assert bad.inject is not None  # format repair

    # Reflection without READY_TO_SUBMIT → repair (stub gate)
    repair_text = (
        "Observe: incomplete\nCorrelate: skipped checks\n"
        "Map: MAST-12\nDecide: verify before submit\n"
        "Final ATLAS status: REPAIR_REQUIRED\n"
    )
    adapter.run.pending = None  # type: ignore[union-attr]
    adapter.run.final_gate_allowed = None  # type: ignore[union-attr]
    adapter.begin_final_gate(messages)
    blocked = adapter.process_assistant_text(messages, repair_text)
    assert blocked.continue_loop is True
    assert blocked.inject is not None
    assert "blocked completion" in blocked.inject.lower() or "ATLAS" in blocked.inject

    adapter.run.pending = None  # type: ignore[union-attr]
    adapter.run.final_gate_allowed = None  # type: ignore[union-attr]
    adapter.begin_final_gate(messages)
    ok_text = (
        "Observe: done\nCorrelate: n/a\nMap: none apply\nDecide: submit\n"
        "Final ATLAS status: READY_TO_SUBMIT\n"
    )
    allowed = adapter.process_assistant_text(messages, ok_text)
    assert allowed.allow_done is True


def test_finalize_records_trace_and_ends(atlas_stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.atlas.adapter import AtlasAdapter

    adapter = AtlasAdapter(config={
        "trace_output": tmp_path / "atlas-program",
        "repair_rounds": 3,
        "format_retries": 2,
        "failure_throttle_calls": 5,
        "failure_recency_seconds": 30,
        "recent_activity_messages": 8,
        "recent_activity_chars": 12000,
        "redact_traces": True,
        "gate_exhaustion_policy": "release",
        "dashboard": False,
    })
    adapter.start("task")
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="task"),
        LLMMessage(role="assistant", content="done"),
    ]
    adapter.finalize(messages, metadata={"source": "test"})
    assert len(_RECORDED) == 1
    assert _RECORDED[0].metadata["harness"] == "clawagents"
    assert _RECORDED[0].metadata["source"] == "test"
    assert len(_ENDED) == 1


def test_resolve_config_defaults(atlas_stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLAW_ATLAS_CONFIG", raising=False)
    from clawagents.atlas.config import resolve_atlas_config

    cfg = resolve_atlas_config(None)
    assert "atlas-program" in str(cfg["trace_output"])
    assert cfg["gate_exhaustion_policy"] == "release"
    assert cfg["dashboard"] is False


@pytest.mark.asyncio
async def test_final_gate_exception_fails_closed(atlas_stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.atlas.adapter import AtlasAdapter

    def fail_gate(self, messages):
        raise RuntimeError("gate unavailable")

    monkeypatch.setattr(AtlasAdapter, "begin_final_gate", fail_gate)
    state = await run_agent_graph(
        "task",
        _ScriptedLLM(["proposed answer"]),
        atlas=True,
        streaming=False,
        max_iterations=2,
    )

    assert state.status == "error"
    assert "ATLAS final gate failed" in str(state.result)
    assert "gate unavailable" in str(state.result)


@pytest.mark.asyncio
async def test_harvest_exception_fails_closed(atlas_stubs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from clawagents.atlas.adapter import AtlasAdapter

    def fail_harvest(self, messages, text):
        raise RuntimeError("harvest unavailable")

    monkeypatch.setattr(AtlasAdapter, "process_assistant_text", fail_harvest)
    state = await run_agent_graph(
        "task",
        _ScriptedLLM(["proposed answer", "ATLAS reflection"]),
        atlas=True,
        streaming=False,
        max_iterations=3,
    )

    assert state.status == "error"
    assert "ATLAS harvest failed" in str(state.result)
    assert "harvest unavailable" in str(state.result)
