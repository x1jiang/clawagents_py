"""Quality-first regression gates for skill discovery and routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from clawagents.agent import (
    _build_skill_catalog_prompt,
    _latest_user_text,
    _skill_relevance_score,
)
from clawagents.tools.skills import SkillStore, create_skill_tools


def _skill(name: str, description: str, **metadata):
    return SimpleNamespace(
        name=name,
        description=description,
        path=f"/skills/{name}/SKILL.md",
        aliases=metadata.get("aliases", []),
        triggers=metadata.get("triggers", []),
        anti_triggers=metadata.get("anti_triggers", []),
    )


def _catalog():
    core = [
        _skill(
            "citation-management",
            "Verify citations DOI metadata BibTeX references bibliography",
            aliases=["bibtex cleanup", "citation fix"],
            triggers=["verify doi"],
            anti_triggers=["delete bibliography"],
        ),
        _skill("database-migrations", "Database schema migration rollback upgrade"),
        _skill("postgres-performance", "Optimize slow PostgreSQL SQL queries"),
        _skill("slides", "Create presentation slides and visual decks"),
        _skill("imagegen", "Generate and edit images"),
        _skill("academic-writing", "Write academic papers and literature reviews"),
        _skill("pdf", "Read create and inspect PDF documents"),
        _skill("docx", "Create polished Word reports"),
        _skill("security-review", "Audit application security risks"),
        _skill("code-review", "Review code correctness and maintainability"),
        _skill("react-ui", "Review React UI UX accessibility and performance"),
        _skill("spreadsheets", "Create and analyze spreadsheet workbooks"),
    ]
    distractors = [
        _skill(f"generic-{index:02d}", f"General workflow project files task {index}")
        for index in range(52)
    ]
    return core + distractors


@pytest.mark.parametrize(
    ("query", "gold"),
    [
        ("fix this citation", "citation-management"),
        ("fix these citations", "citation-management"),
        ("plan database migrations", "database-migrations"),
        ("migrating the database schema", "database-migrations"),
        ("analyze application security", "security-review"),
        ("creating slides from these images", "slides"),
    ],
)
def test_morphology_keeps_the_specialist_ranked_first(query, gold):
    ranked = sorted(_catalog(), key=lambda item: -_skill_relevance_score(item, query))
    assert ranked[0].name == gold


def test_alias_and_trigger_match_across_stopwords_and_inflection():
    citation = _catalog()[0]
    assert _skill_relevance_score(citation, "clean up my BibTeX references") >= 12
    assert _skill_relevance_score(citation, "verify the DOI metadata") >= 12


def test_anti_trigger_is_token_aware_negatable_and_overridden_by_explicit_name():
    citation = _catalog()[0]
    assert _skill_relevance_score(citation, "delete bibliography") < 0
    assert _skill_relevance_score(citation, "do not delete bibliography; verify DOI") > 0
    assert _skill_relevance_score(
        citation, "use citation-management to explain delete bibliography"
    ) > 0
    short_anti = _skill("notes", "Notebook workflow", anti_triggers=["not"])
    assert _skill_relevance_score(short_anti, "notebook workflow") >= 0


def test_name_matching_respects_word_boundaries():
    slides = _skill("slides", "Presentation deck")
    assert _skill_relevance_score(slides, "backslides are common") < 12


def test_multi_intent_prompt_keeps_every_explicit_domain():
    text = _build_skill_catalog_prompt(
        _catalog(),
        query="Use slides, imagegen, academic-writing, and citation-management together",
    )
    recommendations = text.split("### Relevant skills for this turn", 1)[1]
    for name in ("slides", "imagegen", "academic-writing", "citation-management"):
        assert f"**{name}**" in recommendations


@pytest.mark.parametrize("query", ["hello", "thanks", "how are you", "ok", "continue"])
def test_generic_turns_do_not_flood_the_prompt(query):
    assert _build_skill_catalog_prompt(_catalog(), query=query) == ""


def test_substantive_ambiguous_turn_preserves_discovery_and_name_index():
    text = _build_skill_catalog_prompt(_catalog(), query="help with references")
    assert "Catalog names:" in text
    assert "citation-management" in text


def test_short_followup_carries_prior_substantive_intent():
    messages = [
        {"role": "user", "content": "Create a polished presentation slide deck"},
        {"role": "assistant", "content": "I can do that."},
        {"role": "user", "content": "yes, do that"},
    ]
    query = _latest_user_text(messages)
    assert "Prior substantive request" in query
    assert "presentation slide deck" in query


def test_ranked_list_skills_uses_alias_trigger_and_typo(tmp_path):
    root = tmp_path / "skills"
    skill_dir = root / "citation-management"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        """---
name: citation-management
description: Verify citations and DOI metadata
aliases: [bibtex cleanup]
triggers: [verify doi]
---
Body
""",
        encoding="utf-8",
    )
    other = root / "generic"
    other.mkdir()
    other.joinpath("SKILL.md").write_text(
        "---\nname: generic\ndescription: General workflow\n---\nBody\n",
        encoding="utf-8",
    )
    store = SkillStore()
    store.add_directory(root)
    asyncio.run(store.load_all())
    list_tool = [
        tool
        for tool in create_skill_tools(store, relevance_scorer=_skill_relevance_score)
        if tool.name == "list_skills"
    ][0]

    for query in ("bibtex cleanup", "verify the DOI", "citaitons"):
        result = asyncio.run(list_tool.execute({"query": query}))
        assert "citation-management" in result.output
        assert "Match:" in result.output
