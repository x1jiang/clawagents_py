"""Skill catalog ranking, fuzzy resolve, and prompt injection upsert."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from clawagents.agent import _build_skill_catalog_prompt, _skill_relevance_score
from clawagents.prompts import (
    INJECTION_BEGIN,
    INJECTION_END,
    PROMPT_CACHE_BOUNDARY,
    append_prompt_injection,
    build_prompt_injection,
)
from clawagents.providers.llm import LLMMessage
from clawagents.tools.skills import (
    SkillStore,
    parse_skill_file,
    resolve_skill,
    suggest_skills,
)


def test_skill_catalog_ranks_matching_skill_first():
    skills = [
        SimpleNamespace(name="unrelated", description="Format a PowerPoint deck", path="a/SKILL.md"),
        SimpleNamespace(
            name="atomic_waterfall_query",
            description="Cohort SQL waterfall validation and extraction for clinical projects",
            path="b/atomic_waterfall_query/SKILL.md",
        ),
        SimpleNamespace(name="caveman", description="Speak briefly", path="c/SKILL.md"),
    ]
    query = (
        "Please update new_project_starting_instruction.md based on raw_request.txt "
        "and run the atomic waterfall cohort workflow for knee osteoarthritis."
    )
    text = _build_skill_catalog_prompt(skills, context_window=128_000, query=query)
    assert "Recommended for this turn" in text
    assert "atomic_waterfall_query" in text
    # Recommended block should mention the cohort skill before unrelated filler.
    rec = text.split("### Recommended for this turn", 1)[1]
    assert rec.find("atomic_waterfall_query") < rec.find("unrelated") or "unrelated" not in rec.split("Call `use_skill`")[0]


def test_skill_relevance_boosts_filename_mentions():
    skill = SimpleNamespace(
        name="project-start",
        description="Start a new data project",
        path="/skills/new_project_starting_instruction/SKILL.md",
    )
    q = "update new_project_starting_instruction.md from raw_request.txt"
    assert _skill_relevance_score(skill, q) >= 50


def test_parse_block_scalar_description():
    content = """---
name: demo
description: |
  First line of description.
  Second line with cohort SQL hints.
---

Body here.
"""
    skill = parse_skill_file(content, "/tmp/demo/SKILL.md")
    assert skill.name == "demo"
    assert "First line" in skill.description
    assert "cohort SQL" in skill.description


def test_resolve_skill_fuzzy_and_suggest(tmp_path):
    store = SkillStore()
    root = tmp_path / "skills" / "atomic_waterfall_query"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: atomic_waterfall_query\ndescription: Cohort waterfall\n---\n\nDo it.\n",
        encoding="utf-8",
    )
    store.add_directory(tmp_path / "skills")
    import asyncio

    asyncio.run(store.load_all())
    assert resolve_skill(store, "Atomic-Waterfall-Query") is not None
    assert resolve_skill(store, "atomic_waterfall_query") is not None
    assert "atomic_waterfall_query" in suggest_skills(store, "atomic_waterfal_query")


def test_append_prompt_injection_upserts_without_duplicating():
    messages = [
        LLMMessage(
            role="system",
            content=f"static tools\n{PROMPT_CACHE_BOUNDARY}\nlessons\n",
        )
    ]
    first = build_prompt_injection(None, "## Available Skills\n- **a**: A")
    second = build_prompt_injection(None, "## Available Skills\n- **b**: B")
    once = append_prompt_injection(messages, first)
    twice = append_prompt_injection(once, second)
    content = twice[0].content
    assert content.count(INJECTION_BEGIN) == 1
    assert content.count(INJECTION_END) == 1
    assert "**b**: B" in content
    assert "**a**: A" not in content
    assert "lessons" in content
