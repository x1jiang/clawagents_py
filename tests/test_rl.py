"""Hermetic tests for the RL fine-tuning adapter.

These tests must run without ``trl``, ``datasets`` or ``atropos`` —
they exercise the trajectory data model, the recorder's event hooks,
the scorers, and the JSONL/ChatML/TRL/Atropos export shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from clawagents.rl import (
    ATROPOS_AVAILABLE,
    AtroposAdapter,
    CompositeScorer,
    ContainsScorer,
    ExactMatchScorer,
    LengthPenaltyScorer,
    MissingRLDependencyError,
    RLRecorder,
    RecorderConfig,
    RegexScorer,
    TRL_AVAILABLE,
    ToolCall,
    Trajectory,
    TrajectoryStep,
    TrlAdapter,
    export_jsonl,
    load_jsonl,
    to_atropos_rollout,
    to_chatml,
    to_trl_dpo,
    to_trl_sft,
)
from clawagents.rl.export import export_atropos_rollouts_jsonl, export_trl_sft_jsonl
from clawagents.rl.scorers import score_all


# ──────────────────────────────────────────────────────────────────────
# Trajectory data model
# ──────────────────────────────────────────────────────────────────────


def test_trajectory_round_trip_preserves_steps_and_rewards() -> None:
    t = Trajectory(task="answer", model="claude")
    t.add_system("you are helpful")
    t.add_user("what is 2+2?")
    t.add_assistant("4")
    t.reward = 1.0
    t.rewards = {"contains": 1.0, "length": 0.5}
    t.metadata["topic"] = "math"

    restored = Trajectory.from_dict(t.to_dict())
    assert restored.task == "answer"
    assert restored.model == "claude"
    assert [s.role for s in restored.steps] == ["system", "user", "assistant"]
    assert restored.steps[-1].content == "4"
    assert restored.reward == 1.0
    assert restored.rewards == {"contains": 1.0, "length": 0.5}
    assert restored.metadata == {"topic": "math"}


def test_trajectory_assistant_text_concatenates_only_assistant_steps() -> None:
    t = Trajectory()
    t.add_user("q")
    t.add_assistant("first")
    t.add_tool("ran")
    t.add_assistant("second")
    assert t.assistant_text == "first\nsecond"
    final = t.final_assistant
    assert final is not None and final.content == "second"


def test_tool_call_dict_round_trip() -> None:
    tc = ToolCall(
        id="tc1",
        name="bash",
        arguments={"cmd": "ls"},
        result="a\nb",
        success=True,
        duration_ms=12.0,
    )
    restored = ToolCall.from_dict(tc.to_dict())
    assert restored == tc


def test_trajectory_step_with_tool_calls_serialises_inline() -> None:
    t = Trajectory()
    t.add_user("run a command")
    t.add_assistant(
        "calling tool",
        tool_calls=[
            ToolCall(id="tc1", name="bash", arguments={"cmd": "ls"}, result="a\nb")
        ],
    )
    payload = t.to_dict()
    assert payload["steps"][1]["tool_calls"][0]["name"] == "bash"
    restored = Trajectory.from_dict(payload)
    assert restored.steps[1].tool_calls[0].arguments == {"cmd": "ls"}


# ──────────────────────────────────────────────────────────────────────
# Recorder event hooks
# ──────────────────────────────────────────────────────────────────────


def test_recorder_assembles_assistant_then_tool_then_assistant() -> None:
    rec = RLRecorder(task="solve")
    rec.add_user("solve x")

    rec.observe("assistant_message", {"content": "let me try"})
    rec.observe("tool_call", {"id": "tc1", "name": "bash", "arguments": {"cmd": "ls"}})
    rec.observe(
        "tool_result",
        {"id": "tc1", "result": "ok", "success": True, "duration_ms": 3.0},
    )
    rec.observe("turn_started", {})
    rec.observe("assistant_message", {"content": "x = 4"})
    rec.observe("agent_done", {})

    traj = rec.finalise()
    roles = [s.role for s in traj.steps]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert traj.steps[1].tool_calls[0].name == "bash"
    assert traj.steps[1].tool_calls[0].result == "ok"
    assert traj.steps[2].tool_call_id == "tc1"
    assert traj.steps[3].content == "x = 4"


def test_recorder_handles_assistant_deltas() -> None:
    rec = RLRecorder()
    rec.add_user("hi")
    rec.observe("assistant_delta", {"delta": "Hel"})
    rec.observe("assistant_delta", {"delta": "lo"})
    rec.observe("agent_done", {})
    t = rec.finalise()
    assert t.steps[-1].role == "assistant"
    assert t.steps[-1].content == "Hello"


def test_recorder_truncates_large_tool_results() -> None:
    big = "x" * 10_000
    rec = RLRecorder(config=RecorderConfig(max_tool_result_chars=100))
    rec.add_user("q")
    rec.observe("tool_call", {"id": "tc1", "name": "bash"})
    rec.observe("tool_result", {"id": "tc1", "result": big})
    rec.observe("agent_done", {})
    t = rec.finalise()
    tool_step = next(s for s in t.steps if s.role == "tool")
    assert len(tool_step.content) <= 101  # 100 chars + ellipsis
    assert tool_step.content.endswith("…")


def test_recorder_redacts_tool_args_when_configured() -> None:
    rec = RLRecorder(config=RecorderConfig(redact_tool_args=True))
    rec.add_user("q")
    rec.observe("tool_call", {"id": "tc1", "name": "bash", "arguments": {"cmd": "rm -rf /"}})
    rec.observe("tool_result", {"id": "tc1", "result": "ok"})
    rec.observe("agent_done", {})
    t = rec.finalise()
    asst = next(s for s in t.steps if s.role == "assistant")
    assert asst.tool_calls[0].arguments == {"_redacted": True}


def test_recorder_finalise_sets_prompt_only_if_missing() -> None:
    rec = RLRecorder()
    rec.observe("assistant_message", {"content": "answer"})
    t = rec.finalise(prompt="implicit prompt")
    assert t.steps[0].role == "user"
    assert t.steps[0].content == "implicit prompt"

    rec2 = RLRecorder()
    rec2.add_user("explicit")
    rec2.observe("assistant_message", {"content": "answer"})
    t2 = rec2.finalise(prompt="implicit prompt")
    assert [s.content for s in t2.steps if s.role == "user"] == ["explicit"]


def test_recorder_finalise_appends_distinct_final_message() -> None:
    rec = RLRecorder()
    rec.add_user("q")
    rec.observe("assistant_message", {"content": "draft"})
    t = rec.finalise(final="final answer")
    assistants = [s.content for s in t.steps if s.role == "assistant"]
    assert assistants == ["draft", "final answer"]


def test_recorder_ignores_unknown_events_and_post_finalise_calls() -> None:
    rec = RLRecorder()
    rec.observe("unknown_event", {"foo": "bar"})  # silently dropped
    rec.add_user("q")
    rec.observe("assistant_message", {"content": "a"})
    t = rec.finalise()
    rec.observe("assistant_message", {"content": "extra"})
    assert all(s.content != "extra" for s in t.steps)


# ──────────────────────────────────────────────────────────────────────
# Scorers
# ──────────────────────────────────────────────────────────────────────


def _traj(answer: str, prompt: str = "q") -> Trajectory:
    t = Trajectory()
    t.add_user(prompt)
    t.add_assistant(answer)
    return t


def test_contains_scorer_all_or_nothing() -> None:
    s = ContainsScorer(["x = 4"])
    assert s(_traj("the answer is x = 4")) == 1.0
    assert s(_traj("nope")) == -1.0


def test_contains_scorer_partial_credit() -> None:
    s = ContainsScorer(["a", "b"], partial_credit=True)
    assert s(_traj("only a here")) == pytest.approx(0.0)  # 1/2 → 0.0
    assert s(_traj("a and b")) == 1.0
    assert s(_traj("nothing")) == -1.0


def test_exact_match_scorer_handles_case_and_whitespace() -> None:
    s = ExactMatchScorer("hello", strip=True, case_sensitive=False)
    assert s(_traj("  HELLO ")) == 1.0
    assert s(_traj("goodbye")) == -1.0
    s2 = ExactMatchScorer("Hello", case_sensitive=True)
    assert s2(_traj("hello")) == -1.0


def test_regex_scorer_compiles_lazily_and_handles_bad_pattern() -> None:
    assert RegexScorer(r"\d+")(_traj("123 ok")) == 1.0
    assert RegexScorer(r"\d+")(_traj("no nums")) == -1.0
    assert RegexScorer(r"(unbalanced")(_traj("anything")) == 0.0


def test_length_penalty_scorer_rewards_target_length() -> None:
    s = LengthPenaltyScorer(target_chars=10, min_chars=0, max_chars=20)
    assert s(_traj("0123456789")) == 1.0  # exact
    assert s(_traj("")) == -1.0
    assert s(_traj("01234")) == pytest.approx(0.0)  # halfway between min and target
    assert s(_traj("0" * 21)) == -1.0  # over max


def test_composite_scorer_blends_with_weights() -> None:
    contains = ContainsScorer(["good"])
    length = LengthPenaltyScorer(target_chars=4, min_chars=0, max_chars=20)
    composite = CompositeScorer(scorers=[contains, length], weights=[2.0, 1.0])
    score = composite(_traj("good"))
    # contains=+1 (weight 2), length=+1 (weight 1) → (2+1)/3 = 1.0
    assert score == pytest.approx(1.0)


def test_composite_scorer_validates_weights_length() -> None:
    cs = CompositeScorer(scorers=[ContainsScorer(["x"])], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        cs(_traj("x"))


def test_score_all_writes_back_to_trajectory() -> None:
    t = _traj("good answer")
    out = score_all(t, {"contains": ContainsScorer(["good"]), "len": LengthPenaltyScorer()})
    assert out["contains"] == 1.0
    assert t.rewards["contains"] == 1.0
    assert t.reward is not None  # mean was assigned


# ──────────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────────


def test_export_and_load_jsonl_round_trip(tmp_path: Path) -> None:
    a = _traj("first")
    b = _traj("second")
    p = tmp_path / "runs.jsonl"
    n = export_jsonl([a, b], p)
    assert n == 2
    loaded = load_jsonl(p)
    assert [t.assistant_text for t in loaded] == ["first", "second"]


def test_to_chatml_renders_tool_calls_in_openai_shape() -> None:
    t = Trajectory()
    t.add_user("run ls")
    t.add_assistant(
        "calling",
        tool_calls=[ToolCall(id="tc1", name="bash", arguments={"cmd": "ls"}, result="ok")],
    )
    t.add_tool("ok", tool_call_id="tc1", name="bash")
    msgs = to_chatml(t)
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "tc1"


def test_to_trl_sft_includes_prompt_and_completion() -> None:
    t = _traj("the answer")
    row = to_trl_sft(t)
    assert row["completion"][0]["content"] == "the answer"
    assert row["prompt"][0]["role"] == "user"
    assert row["metadata"]["run_id"] == t.run_id


def test_to_trl_dpo_uses_chosen_prefix() -> None:
    chosen = _traj("good", prompt="solve")
    rejected = _traj("bad", prompt="solve")
    chosen.reward = 1.0
    rejected.reward = -1.0
    pair = to_trl_dpo(chosen, rejected)
    assert pair["prompt"][0]["content"] == "solve"
    assert pair["chosen"][0]["content"] == "good"
    assert pair["rejected"][0]["content"] == "bad"
    assert pair["metadata"]["chosen_reward"] == 1.0


def test_to_atropos_rollout_passes_through_score_and_metadata() -> None:
    t = _traj("ok")
    t.reward = 0.7
    t.rewards = {"contains": 0.7}
    t.metadata = {"task_id": 42}
    r = to_atropos_rollout(t)
    assert r["score"] == 0.7
    assert r["rewards"] == {"contains": 0.7}
    assert r["metadata"]["task_id"] == 42
    assert r["messages"][0]["role"] == "user"


def test_export_trl_sft_jsonl_writes_rows(tmp_path: Path) -> None:
    p = tmp_path / "trl.jsonl"
    export_trl_sft_jsonl([_traj("a"), _traj("b")], p)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    assert [r["completion"][0]["content"] for r in rows] == ["a", "b"]


def test_export_atropos_rollouts_jsonl_writes_rows(tmp_path: Path) -> None:
    p = tmp_path / "atropos.jsonl"
    export_atropos_rollouts_jsonl([_traj("a"), _traj("b")], p)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    assert all("messages" in r for r in rows)


# ──────────────────────────────────────────────────────────────────────
# Optional adapters
# ──────────────────────────────────────────────────────────────────────


def test_atropos_adapter_to_rollouts_does_not_require_dependency() -> None:
    adapter = AtroposAdapter()
    rollouts = adapter.to_rollouts([_traj("a"), _traj("b")])
    assert [r["messages"][-1]["content"] for r in rollouts] == ["a", "b"]


def test_atropos_adapter_uses_provided_sink_without_dependency() -> None:
    submitted: list[Dict[str, Any]] = []

    class FakeSink:
        def submit(self, rollout: Dict[str, Any]) -> None:
            submitted.append(rollout)

    adapter = AtroposAdapter()
    n = adapter.submit([_traj("a"), _traj("b")], sink=FakeSink())
    assert n == 2
    assert len(submitted) == 2


def test_atropos_adapter_raises_when_dependency_missing() -> None:
    if ATROPOS_AVAILABLE:
        pytest.skip("atropos installed in this environment")
    with pytest.raises(MissingRLDependencyError) as exc_info:
        AtroposAdapter().submit([_traj("a")])
    assert "atropos" in str(exc_info.value).lower()


def test_trl_adapter_raises_when_datasets_missing() -> None:
    try:
        import datasets  # noqa: F401
    except Exception:
        with pytest.raises(MissingRLDependencyError) as exc_info:
            TrlAdapter().build_sft_dataset([_traj("a")])
        assert "trl" in str(exc_info.value).lower()
    else:
        pytest.skip("datasets installed in this environment")
