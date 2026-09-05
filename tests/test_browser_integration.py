"""Test integration of browser accessibility tools in create_claw_agent."""

from __future__ import annotations

from pathlib import Path

from clawagents.agent import create_claw_agent
from clawagents.config.features import temporary_overrides
from clawagents.providers.llm import LLMProvider


class _StubLLM(LLMProvider):
    name = "stub"

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None, **kwargs):
        raise NotImplementedError()


def test_browser_tools_registered_when_flag_enabled(tmp_path: Path):
    agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path, browser=True)
    expected_tools = [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_hover",
    ]
    for name in expected_tools:
        tool = agent.tools.get(name)
        assert tool is not None, f"Expected {name} to be registered in agent.tools"


def test_browser_tools_registered_via_feature_override(tmp_path: Path):
    with temporary_overrides({"browser_tools": True}):
        agent = create_claw_agent(model=_StubLLM(), workspace=tmp_path)
        assert agent.tools.get("browser_navigate") is not None
        assert agent.tools.get("browser_snapshot") is not None
