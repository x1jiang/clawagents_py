"""Capabilities contract + workspace-scoped tool artifact archival."""

from __future__ import annotations


import pytest


def test_capabilities_advertise_host_contract():
    from clawagents.capabilities import CAPABILITIES_CONTRACT_VERSION, get_capabilities

    caps = get_capabilities()
    assert caps["contract_version"] == CAPABILITIES_CONTRACT_VERSION
    assert caps["gemini_array_items"] is True
    assert caps["workspace_scoped_agent"] is True
    assert caps["raw_tool_output"] is True
    assert caps["artifact_workspace_arg"] is True
    # Defensive copy — callers must not mutate the module map.
    caps["gemini_array_items"] = False
    assert get_capabilities()["gemini_array_items"] is True


def test_prepare_tool_output_uses_workspace_not_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from clawagents.tool_output_artifacts import prepare_tool_output_for_context

    workspace = tmp_path / "agent_ws"
    cwd = tmp_path / "other_cwd"
    workspace.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    big = ("line\n" * 400) + ("x" * 2500)
    prompt, aid = prepare_tool_output_for_context(
        tool_name="execute",
        tool_use_id="ws-1",
        output=big,
        workspace=str(workspace),
    )
    assert aid is not None
    assert "[Crushed tool output" in prompt
    assert aid in prompt

    artifacts = workspace / ".clawagents" / "tool-artifacts"
    assert artifacts.is_dir()
    # Must not archive under the process cwd when workspace= is set.
    stray = cwd / ".clawagents" / "tool-artifacts"
    assert not stray.exists()
    hits = list(artifacts.rglob("*"))
    assert hits, f"expected artifacts under {artifacts}"


def test_run_context_workspace_helper():
    from clawagents.graph.agent_loop import _run_context_workspace

    class RC:
        def __init__(self, meta):
            self._metadata = meta

    assert _run_context_workspace(RC({"workspace": " /tmp/ws "})) == "/tmp/ws"
    assert _run_context_workspace(RC({})) is None
    assert _run_context_workspace(None) is None
