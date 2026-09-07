"""Public-loop coverage for temporary recovery settings and profile opt-in."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from clawagents.graph.agent_loop import run_agent_graph
from clawagents.graph.run_bootstrapper import RunBootstrapper, _bind_agent_loop_refs
from clawagents.graph.run_config import AgentRunConfig
from clawagents.harness_profiles import register_harness_alias, resolve_harness_profile
from clawagents.providers.llm import LLMProvider, LLMResponse


class _TruncatedThenFinished(LLMProvider):
    name = "fake"
    model = "test-glimmer-integration"

    def __init__(self, *, cancel=False):
        self._max_tokens = 6144
        self.caps = []
        self.cancel = cancel

    async def chat(self, messages, **kwargs):
        self.caps.append(self._max_tokens)
        if len(self.caps) == 1:
            return LLMResponse(content="", model=self.model, tokens_used=6144,
                               finish_reason="length")
        if self.cancel:
            raise asyncio.CancelledError()
        return LLMResponse(content="Finished.", model=self.model, tokens_used=8)


@pytest.mark.parametrize("cancel", [False, True])
def test_public_run_restores_temporary_output_cap(tmp_path, monkeypatch, cancel):
    monkeypatch.chdir(tmp_path)
    llm = _TruncatedThenFinished(cancel=cancel)
    state = asyncio.run(run_agent_graph(
        "Say finished", llm, system_prompt="Be concise.",
        streaming=False, max_iterations=4, on_event=lambda *_: None,
        features={"session_persistence": False}, session_end_tail=False,
    ))
    assert llm.caps == [6144, 9216]
    assert llm._max_tokens == 6144
    assert state.result == ("[cancelled]" if cancel else "Finished.")


@pytest.mark.parametrize("model, expected", [
    ("test-served-glimmer-progress", 8),
    ("gpt-5.6-luna", 0),
    ("unknown-model-without-profile", 0),
])
def test_profile_progress_override_reaches_tracker(monkeypatch, model, expected):
    # Exercise real alias resolution, while isolating the profile content from
    # the separately owned default-profile edit.
    register_harness_alias("test-served-glimmer-progress", "meta-glimmer")

    def profile(name):
        resolved = resolve_harness_profile(name)
        if resolved is not None and resolved.name == "meta-glimmer":
            return replace(resolved, loop_detection_overrides={
                **resolved.loop_detection_overrides, "progress_nudge_after": 8,
            })
        return resolved

    monkeypatch.setattr("clawagents.harness_profiles.resolve_harness_profile", profile)
    _bind_agent_loop_refs()
    bootstrapper = RunBootstrapper(AgentRunConfig(
        task="review", llm=SimpleNamespace(model=model),
    ))
    bootstrapper._resolve_config()
    tracker = bootstrapper._loop_tracker
    assert tracker._progress_nudge_after == expected
    notices = []
    for i in range(8):
        args = {"path": f"{i}.py"}
        tracker.record("read_file", args)
        notices.append(tracker.record_result("read_file", args, "same evidence", success=True))
    assert bool(notices[-1]) == bool(expected)
