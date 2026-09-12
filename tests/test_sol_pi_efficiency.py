"""Run-scoped efficiency, todo carryover, and receipt retention regressions."""

import pytest

from clawagents.run_context import RunContext
from clawagents.run_result import RunResult
from clawagents.tools.todolist import WriteTodosTool, UpdateTodoTool, pending_todos
from clawagents.efficiency import get_efficiency, efficiency_snapshot, record_compaction


@pytest.mark.asyncio
async def test_todos_are_isolated_and_only_pending_are_carried():
    first, second = RunContext(), RunContext()
    await WriteTodosTool().execute({"todos": ["Read", "Fix"]}, first)
    await WriteTodosTool().execute({"todos": ["Other task"]}, second)
    await UpdateTodoTool().execute({"index": 0}, first)
    assert pending_todos(first) == ["Fix"]
    assert pending_todos(second) == ["Other task"]
    assert pending_todos(RunContext()) == []
    assert pending_todos(None) == []


def test_efficiency_isolated_snapshot_and_old_result_roundtrip():
    first, second = RunContext(), RunContext()
    get_efficiency(first)["round_trips_avoided"] += 1
    record_compaction(first, "micro")
    snap = efficiency_snapshot(first)
    get_efficiency(first)["compactions"]["micro"] += 1
    assert snap["compactions"] == {"micro": 1}
    assert get_efficiency(second)["round_trips_avoided"] == 0
    result = RunResult.from_state({"efficiency": snap})
    assert result.to_state()["efficiency"] == snap
    assert RunResult.from_state({}).efficiency["cache_debt_tokens"] == 0


@pytest.mark.parametrize("prefix", ["", "Applied changes to file.py\n\n[then_run:failed]\n"])
def test_receipts_survive_native_and_text_micro_compaction(monkeypatch, prefix):
    from clawagents.graph.context_management import _micro_compact_tool_results
    from clawagents.providers.llm import LLMMessage

    monkeypatch.setattr("clawagents.config.features.is_enabled", lambda _: True)
    receipt = prefix + "clawagents_evidence_receipt_v1\n" + "failure evidence " * 100
    msgs = [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "a", "name": "execute"}],
        ),
        LLMMessage(role="tool", content=receipt, tool_call_id="a"),
        LLMMessage(role="assistant", content='[{"tool":"edit_file"},{"tool":"execute"}]'),
        LLMMessage(role="user", content="[Tool Result] " + receipt),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "b", "name": "execute"}],
        ),
        LLMMessage(role="tool", content="recent", tool_call_id="b"),
    ]
    after = _micro_compact_tool_results(msgs, keep_recent=0)
    assert after[1] is msgs[1]
    assert after[3] is msgs[3]


@pytest.mark.asyncio
async def test_compaction_receives_live_pending_todos(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from clawagents.graph.context_management import _compact_if_needed
    from clawagents.providers.llm import LLMMessage
    from clawagents.config.features import set_overrides, reset

    seen = []

    async def compact(messages, llm, **kwargs):
        seen.append(kwargs.get("system_reminder"))
        return [LLMMessage(role="user", content="continued with plan")]

    monkeypatch.setattr(
        "clawagents.memory.full_replace_compaction.apply_full_replace_compaction",
        compact,
    )
    rc = RunContext()
    rc._metadata["workspace"] = str(tmp_path)
    await WriteTodosTool().execute({"todos": ["Completed step", "Keep this step"]}, rc)
    await UpdateTodoTool().execute({"index": 0}, rc)
    msgs = [
        LLMMessage(role="user", content=f"task {i} " + "data " * 1000)
        for i in range(24)
    ]
    set_overrides({"full_replace_compaction": True, "compaction_segments": False})
    try:
        await _compact_if_needed(
            msgs, 1000, SimpleNamespace(), lambda *args: None, run_context=rc
        )
    finally:
        reset()
    assert seen
    assert "Keep this step" in seen[0]
    assert "Completed step" not in seen[0]


def test_run_result_snapshot_and_stream_preserve_efficiency():
    from types import SimpleNamespace
    from clawagents.stream_events import stream_event_from_kind

    rc = RunContext()
    rc.efficiency["round_trips_avoided"] = 2
    state = SimpleNamespace(
        messages=[],
        current_task="task",
        status="done",
        final_output="ok",
        result="ok",
        iterations=1,
        max_iterations=2,
        tool_calls=1,
        run_context=rc,
    )
    result = RunResult.from_agent_state(state)
    rc.efficiency["round_trips_avoided"] = 3
    assert result.efficiency["round_trips_avoided"] == 2
    event = stream_event_from_kind(
        "usage",
        {
            "prompt_tokens": 100,
            "input_tokens": 10,
            "cached_input_tokens": 80,
            "cache_creation_tokens": 10,
            "efficiency": result.efficiency,
        },
    )
    assert event.prompt_tokens == 100
    assert event.efficiency["round_trips_avoided"] == 2


def test_micro_compaction_does_not_recount_stubs(monkeypatch):
    from clawagents.graph.context_management import _micro_compact_tool_results
    from clawagents.providers.llm import LLMMessage

    monkeypatch.setattr("clawagents.config.features.is_enabled", lambda _: True)
    msgs = [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": "a", "name": "execute"}],
        ),
        LLMMessage(
            role="tool",
            content="[Old tool result cleared to save context]",
            tool_call_id="a",
        ),
    ]
    assert _micro_compact_tool_results(msgs, keep_recent=0) is msgs


@pytest.mark.asyncio
async def test_agent_as_tool_isolates_todos_and_efficiency_but_inherits_permissions():
    from types import SimpleNamespace
    from clawagents.agent import _AgentAsTool
    from clawagents.permissions.mode import PermissionMode

    parent = RunContext(context={"user": "shared"}, permission_mode=PermissionMode.PLAN)
    parent.efficiency["round_trips_avoided"] = 4
    parent.todos = [{"text": "Parent plan", "done": False}]
    parent.approve_tool("approved")

    class Wrapped:
        async def invoke(self, task, run_context):
            assert run_context is not parent
            assert run_context.context is parent.context
            assert run_context.permission_mode == parent.permission_mode
            assert run_context.is_tool_approved("approved") is True
            assert run_context.todos == []
            run_context.efficiency["round_trips_avoided"] = 7
            run_context.todos.append({"text": "Child plan", "done": False})
            return SimpleNamespace(result="ok")

    tool = _AgentAsTool(Wrapped(), tool_name="child", tool_description="child")
    assert (await tool.execute({"task": "child task"}, parent)).success
    assert parent.efficiency["round_trips_avoided"] == 4
    assert pending_todos(parent) == ["Parent plan"]


@pytest.mark.parametrize("prefix", ["", "Applied changes to file.py\n\n[then_run:failed]\n"])
def test_soft_trim_preserves_verified_receipt(prefix):
    from clawagents.graph.context_management import _soft_trim_messages
    from clawagents.providers.llm import LLMMessage
    receipt = LLMMessage(role="tool", content=prefix + "clawagents_evidence_receipt_v1\n" + "key evidence\n" * 200, tool_call_id="receipt")
    msgs = [receipt] + [LLMMessage(role="user", content="recent") for _ in range(12)]
    assert _soft_trim_messages(msgs, 1000, 1.0, lambda *args: None, current_tokens=5000)[0] is receipt


def test_micro_handle_swap_counts_only_actual_context_reduction(monkeypatch):
    from types import SimpleNamespace
    from clawagents.graph.turn_driver import TurnDriver
    from clawagents.providers.llm import LLMMessage
    monkeypatch.setattr("clawagents.config.features.is_enabled", lambda _: True)
    rc = RunContext()
    driver = SimpleNamespace(
        _resolved_model_name="unknown", _context_window=1000, _run_context=rc,
        _rebase_ledger=lambda messages: 1000, _note_context_change=lambda: None,
    )
    msgs = []
    for i in range(4):
        msgs.extend([
            LLMMessage(role="assistant", content="", tool_calls_meta=[{"id": str(i), "name": "execute"}]),
            LLMMessage(role="tool", content="data " * 1500 + " artifact id=stored-1234", tool_call_id=str(i)),
        ])
    compacted, _ = TurnDriver._micro_compact(driver, msgs, 5000)
    assert rc.efficiency["tokens_avoided_by_handles"] > 0
    assert rc.efficiency["compactions"] == {"micro": 1}
    snapshot = efficiency_snapshot(rc)
    TurnDriver._micro_compact(driver, compacted, 5000)
    assert rc.efficiency == snapshot


@pytest.mark.parametrize("prefix", ["", "Applied changes to file.py\n\n[then_run:failed]\n"])
def test_summarizer_preflight_preserves_receipt_while_trimming_raw_tools(prefix):
    from clawagents.memory.compact_tool_results import compact_tool_results
    from clawagents.providers.llm import LLMMessage
    receipt = LLMMessage(role="tool", content=prefix + "clawagents_evidence_receipt_v1\n" + "failure evidence\n" * 200, tool_call_id="receipt")
    raw = LLMMessage(role="tool", content="noisy line\n" * 1000, tool_call_id="raw")
    result, modified = compact_tool_results([receipt, raw], max_input_tokens=1000)
    assert modified
    assert result[0] is receipt
    assert len(result[1].content) < len(raw.content)


@pytest.mark.parametrize("text, expected", [
    ("clawagents_evidence_receipt_v1\nevidence", True),
    ("[Tool Result] clawagents_evidence_receipt_v1\nevidence", True),
    ("Applied edit\n[then_run:failed]\nclawagents_evidence_receipt_v1\nevidence", True),
    ("The log mentions clawagents_evidence_receipt_v1 here", False),
    ("clawagents_evidence_receipt_v1_extra\nevidence", False),
])
def test_receipt_detection_requires_a_complete_marker_line(text, expected):
    from clawagents.efficiency import contains_evidence_receipt
    assert contains_evidence_receipt(text) is expected
