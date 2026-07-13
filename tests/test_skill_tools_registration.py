"""Skill catalog budget + tool registration (cost-effective progressive disclosure)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from clawagents.agent import (
    SKILL_LISTING_BUDGET_CEILING_CHARS,
    SKILL_LISTING_BUDGET_FLOOR_CHARS,
    _build_skill_catalog_prompt,
    create_claw_agent,
    skill_listing_budget_chars,
)


def test_skill_listing_budget_scales_with_context(monkeypatch):
    monkeypatch.delenv("CLAW_SKILL_LISTING_CHAR_BUDGET", raising=False)
    monkeypatch.delenv("CLAW_SKILL_LISTING_BUDGET_FRACTION", raising=False)
    small = skill_listing_budget_chars(32_000)
    large = skill_listing_budget_chars(200_000)
    assert small == SKILL_LISTING_BUDGET_FLOOR_CHARS
    assert large >= small
    assert large <= SKILL_LISTING_BUDGET_CEILING_CHARS


def test_skill_listing_budget_absolute_override(monkeypatch):
    monkeypatch.setenv("CLAW_SKILL_LISTING_CHAR_BUDGET", "5000")
    assert skill_listing_budget_chars(200_000) == 5000


def test_skill_catalog_keeps_names_shortens_descriptions(monkeypatch):
    monkeypatch.setenv("CLAW_SKILL_LISTING_CHAR_BUDGET", "2500")
    monkeypatch.setenv("CLAW_SKILL_LISTING_MAX_DESC_CHARS", "80")
    skills = [
        SimpleNamespace(
            name=f"skill-{i:02d}",
            description=("keyword trigger phrase " * 20) + f" id={i}",
        )
        for i in range(40)
    ]
    text = _build_skill_catalog_prompt(skills, context_window=128_000)
    assert "use_skill" in text
    assert "list_skills" in text
    # Names should appear; long raw descriptions should not.
    assert "skill-00" in text
    assert len(text) <= 2500 + 120  # allow tiny overflow footer slack before hard trim
    assert "keyword trigger phrase " * 5 not in text


def test_skill_catalog_overflow_mentions_list_skills(monkeypatch):
    monkeypatch.setenv("CLAW_SKILL_LISTING_CHAR_BUDGET", "1200")
    skills = [
        SimpleNamespace(name=f"skill-{i}", description=f"desc {i} " + ("x" * 60))
        for i in range(50)
    ]
    text = _build_skill_catalog_prompt(skills, context_window=64_000)
    assert "list_skills" in text
    assert "more skills available" in text or "truncated" in text


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
