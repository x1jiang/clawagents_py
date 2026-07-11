"""Tests for Headroom-inspired context improvements (cache, tiers, learn, FTS)."""

from __future__ import annotations

from pathlib import Path

import pytest

from clawagents.memory.output_trim import trim_verbose_messages
from clawagents.providers.llm import LLMMessage
from clawagents.tool_output_artifacts import search_tool_artifacts, store_tool_artifact
from clawagents.trajectory.failure_learn import append_failure_lessons_to_agents_md
from clawagents.tools.filesystem import ReadFileTool
from clawagents.tools.retrieve_tool_result import RetrieveToolResultTool


def test_trim_verbose_assistant():
    msgs = [
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="assistant", content="x" * 20_000),
    ]
    out, n = trim_verbose_messages(msgs, assistant_chars=1000)
    assert n == 1
    assert len(out[1].content) < 2000
    assert "trimmed" in out[1].content


def test_failure_learn_appends_agents_md(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lessons = "- Always verify paths exist before writing\n- Prefer L0 read before L2\n"
    written = append_failure_lessons_to_agents_md(lessons, workspace=tmp_path)
    assert len(written) == 2
    text = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "Failure lessons" in text
    assert "verify paths" in text
    # Idempotent
    again = append_failure_lessons_to_agents_md(lessons, workspace=tmp_path)
    assert again == []


@pytest.mark.asyncio
async def test_read_file_tier_l0(tmp_path: Path):
    path = tmp_path / "mod.py"
    path.write_text("x = 1\n\nclass Foo:\n    pass\n\ndef bar():\n    return 2\n", encoding="utf-8")

    class _SB:
        def safe_path(self, p: str) -> str:
            return str(tmp_path / p) if not Path(p).is_absolute() else p

        async def read_file(self, p: str) -> str:
            return Path(p).read_text(encoding="utf-8")

        async def read_file_bytes(self, p: str) -> bytes:
            return Path(p).read_bytes()

    tool = ReadFileTool(_SB())
    r = await tool.execute({"path": str(path), "tier": "L0"})
    assert r.success
    assert "L0 outline" in r.output
    assert "class Foo" in r.output
    assert "def bar" in r.output


def test_search_tool_artifacts(tmp_path: Path):
    aid, _ = store_tool_artifact(
        tool_name="execute",
        tool_use_id="tc-search",
        output="unique_needle_xyz and some padding " + ("z" * 200),
        kind="log",
        workspace=tmp_path,
    )
    hits = search_tool_artifacts("unique_needle_xyz", workspace=tmp_path)
    assert hits
    assert hits[0]["id"] == aid


@pytest.mark.asyncio
async def test_retrieve_tool_query(tmp_path: Path):
    store_tool_artifact(
        tool_name="grep",
        tool_use_id="tc-q",
        output="found secret_token_abc in file",
        kind="search",
        workspace=tmp_path,
    )
    tool = RetrieveToolResultTool(workspace=str(tmp_path))
    r = await tool.execute({"query": "secret_token_abc"})
    assert r.success
    assert "secret_token_abc" in r.output
