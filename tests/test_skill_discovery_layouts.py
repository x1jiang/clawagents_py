"""Auto-discovery has to cover where skills are actually kept.

The default list was `skills/.skills/skill/.skill/Skills`, which missed both
halves of the real world: the dotted directories agent shells own, and
`.clawagents/skills` — the very directory `marketplace_install` writes to, so
an install reported success and then loaded nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clawagents.agent import _DEFAULT_SKILL_DIRS, _auto_discover_skills


def _seed(root: Path, rel: str) -> None:
    d = root / rel / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: demo\ndescription: d\n---\nbody\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    "layout",
    [".agents/skills", ".agent/skills", ".cursor/skills", ".claude/skills", "skills"],
)
def test_known_layouts_are_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, layout)
    found = [str(p) for p in _auto_discover_skills()]
    assert any(p.endswith(layout) for p in found), f"{layout} not in {found}"


def test_marketplace_install_dir_is_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strongest form: install through the real API, then discover it."""
    monkeypatch.chdir(tmp_path)
    from clawagents.marketplace import skills_install_dir

    target = skills_install_dir(tmp_path)
    (target / "demo").mkdir(parents=True)
    (target / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: d\n---\nbody\n", encoding="utf-8"
    )

    found = [str(p) for p in _auto_discover_skills()]
    assert any(p.endswith(".clawagents/skills") for p in found), (
        "marketplace_install writes here; leaving it undiscovered makes the "
        f"install a no-op. discovered: {found}"
    )


def test_no_skill_dirs_discovers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _auto_discover_skills() == []


def test_legacy_plain_skills_dir_still_first(tmp_path: Path, monkeypatch) -> None:
    """Ordering is precedence: later entries win name collisions."""
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path, "skills")
    _seed(tmp_path, ".clawagents/skills")
    found = [str(p) for p in _auto_discover_skills()]
    plain = next(i for i, p in enumerate(found) if p.endswith("/skills") and ".clawagents" not in p)
    installed = next(i for i, p in enumerate(found) if p.endswith(".clawagents/skills"))
    assert plain < installed, (
        "an explicitly installed skill should override a same-named local one"
    )


def test_discovery_list_has_no_duplicates() -> None:
    assert len(_DEFAULT_SKILL_DIRS) == len(set(_DEFAULT_SKILL_DIRS))
