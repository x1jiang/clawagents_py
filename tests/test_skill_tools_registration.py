"""Skill catalog budget + tool registration (cost-effective progressive disclosure)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from clawagents.agent import (
    MAX_SKILLS_IN_PROMPT,
    _build_skill_catalog_prompt,
    create_claw_agent,
)


def test_skill_catalog_truncates_to_budget():
    skills = [
        SimpleNamespace(name=f"skill-{i}", description=f"desc {i}")
        for i in range(MAX_SKILLS_IN_PROMPT + 5)
    ]
    text = _build_skill_catalog_prompt(skills)
    assert "list_skills" in text
    assert "use_skill" in text
    assert text.count("- **skill-") == MAX_SKILLS_IN_PROMPT
    assert "5 more skills available" in text


def test_create_claw_agent_registers_list_and_use_skill(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills" / "demo"
    skills_root.mkdir(parents=True)
    (skills_root / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\nDo the demo.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    mock_llm = MagicMock()
    with patch("clawagents.agent._resolve_model", return_value=(mock_llm, "gpt-test", None)):
        agent = create_claw_agent(skills=str(tmp_path / "skills"), memory=[])

    names = {t.name for t in agent.tools.list()}
    assert "use_skill" in names
    assert "list_skills" in names
