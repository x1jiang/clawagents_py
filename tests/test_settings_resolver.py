"""Tests for the settings hierarchy resolver.

Exercises every layer-merge case (only-user, user+project, etc.),
policy-always-wins, missing-file ok, malformed JSON ok, and the
dotted-path getter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clawagents.settings import (
    SettingsLayer,
    find_repo_root,
    get_setting,
    resolve_settings,
)
from clawagents.settings.resolver import POLICY_SETTINGS_PATH_ENV


# ─── Helpers ───────────────────────────────────────────────────────────


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Build an isolated filesystem layout: home, repo, and a policy file slot.

    Returns a dict of named paths so tests can write to them directly.
    Patches HOME and the policy-path env var so resolve_settings sees them.
    """
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    policy = tmp_path / "policy.json"
    (home / ".clawagents").mkdir(parents=True)
    (repo / ".clawagents").mkdir(parents=True)
    # Make repo look like a repo root.
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    # Some platforms / Path.home() implementations look at USERPROFILE too.
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv(POLICY_SETTINGS_PATH_ENV, str(policy))
    return {
        "home": home,
        "repo": repo,
        "policy": policy,
        "user_settings": home / ".clawagents" / "settings.json",
        "project_settings": repo / ".clawagents" / "settings.json",
        "local_settings": repo / ".clawagents" / "settings.local.json",
    }


# ─── Layer enum ────────────────────────────────────────────────────────


def test_settings_layer_enum_values() -> None:
    assert SettingsLayer.USER.value == "user"
    assert SettingsLayer.PROJECT.value == "project"
    assert SettingsLayer.LOCAL.value == "local"
    assert SettingsLayer.FLAG.value == "flag"
    assert SettingsLayer.POLICY.value == "policy"


# ─── find_repo_root ────────────────────────────────────────────────────


def test_find_repo_root_walks_up_to_pyproject(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    nested = repo / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == repo.resolve()


def test_find_repo_root_falls_back_to_cwd(tmp_path: Path) -> None:
    # No marker anywhere.
    nested = tmp_path / "no" / "marker"
    nested.mkdir(parents=True)
    # Falls back to the start path itself.
    assert find_repo_root(nested) == nested.resolve()


def test_find_repo_root_finds_git_dir(tmp_path: Path) -> None:
    repo = tmp_path / "g"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "x"
    nested.mkdir()
    assert find_repo_root(nested) == repo.resolve()


# ─── Layer-merge cases ────────────────────────────────────────────────


def test_only_user_layer(fake_env: dict[str, Path]) -> None:
    _write_json(fake_env["user_settings"], {"theme": "dark", "max_turns": 5})
    out = resolve_settings(repo_root=fake_env["repo"])
    assert out == {"theme": "dark", "max_turns": 5}


def test_user_plus_project_deep_merges(fake_env: dict[str, Path]) -> None:
    _write_json(
        fake_env["user_settings"],
        {"theme": "dark", "hooks": {"before_tool": ["log"]}},
    )
    _write_json(
        fake_env["project_settings"],
        {"hooks": {"after_tool": ["audit"]}, "max_turns": 7},
    )
    out = resolve_settings(repo_root=fake_env["repo"])
    assert out == {
        "theme": "dark",
        "max_turns": 7,
        "hooks": {"before_tool": ["log"], "after_tool": ["audit"]},
    }


def test_local_overrides_project_and_user(fake_env: dict[str, Path]) -> None:
    _write_json(fake_env["user_settings"], {"theme": "dark"})
    _write_json(fake_env["project_settings"], {"theme": "light", "lang": "en"})
    _write_json(fake_env["local_settings"], {"theme": "solarized"})
    out = resolve_settings(repo_root=fake_env["repo"])
    assert out["theme"] == "solarized"
    assert out["lang"] == "en"


def test_flag_overrides_lower_layers(fake_env: dict[str, Path]) -> None:
    _write_json(fake_env["user_settings"], {"theme": "dark", "max_turns": 5})
    _write_json(fake_env["project_settings"], {"max_turns": 7})
    _write_json(fake_env["local_settings"], {"max_turns": 10})
    out = resolve_settings(
        repo_root=fake_env["repo"],
        flag_overrides={"max_turns": 99, "extra": True},
    )
    assert out["max_turns"] == 99
    assert out["extra"] is True
    assert out["theme"] == "dark"


def test_policy_always_wins(fake_env: dict[str, Path]) -> None:
    _write_json(fake_env["user_settings"], {"theme": "dark", "max_turns": 5})
    _write_json(fake_env["project_settings"], {"max_turns": 7})
    _write_json(fake_env["local_settings"], {"max_turns": 10})
    _write_json(fake_env["policy"], {"max_turns": 1, "policy_locked": True})
    out = resolve_settings(
        repo_root=fake_env["repo"],
        flag_overrides={"max_turns": 99},
    )
    # policy beats EVERY other layer, including the flag layer.
    assert out["max_turns"] == 1
    assert out["policy_locked"] is True
    assert out["theme"] == "dark"


def test_policy_deep_merges_into_lower_layers(fake_env: dict[str, Path]) -> None:
    _write_json(
        fake_env["user_settings"],
        {"hooks": {"before_tool": ["log"], "after_tool": ["audit"]}},
    )
    _write_json(fake_env["policy"], {"hooks": {"after_tool": ["forced-audit"]}})
    out = resolve_settings(repo_root=fake_env["repo"])
    # Policy replaces nested key but leaves the sibling alone.
    assert out["hooks"]["before_tool"] == ["log"]
    assert out["hooks"]["after_tool"] == ["forced-audit"]


# ─── Graceful failures ────────────────────────────────────────────────


def test_missing_files_yield_empty_layers(fake_env: dict[str, Path]) -> None:
    # Nothing on disk at all — should return {} cleanly, no exception.
    out = resolve_settings(repo_root=fake_env["repo"])
    assert out == {}


def test_malformed_json_is_skipped_with_warning(
    fake_env: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    fake_env["project_settings"].write_text("{not valid json", encoding="utf-8")
    _write_json(fake_env["user_settings"], {"theme": "dark"})
    with caplog.at_level("WARNING", logger="clawagents.settings.resolver"):
        out = resolve_settings(repo_root=fake_env["repo"])
    # User layer still applied; malformed project layer skipped.
    assert out == {"theme": "dark"}
    assert any("malformed JSON" in rec.message for rec in caplog.records)


def test_non_object_json_is_skipped(
    fake_env: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    # A JSON list at the top level isn't a settings object.
    fake_env["project_settings"].write_text("[1, 2, 3]", encoding="utf-8")
    _write_json(fake_env["user_settings"], {"theme": "dark"})
    with caplog.at_level("WARNING", logger="clawagents.settings.resolver"):
        out = resolve_settings(repo_root=fake_env["repo"])
    assert out == {"theme": "dark"}


# ─── get_setting ──────────────────────────────────────────────────────


def test_get_setting_top_level(fake_env: dict[str, Path]) -> None:
    _write_json(fake_env["user_settings"], {"theme": "dark"})
    assert get_setting("theme", repo_root=fake_env["repo"]) == "dark"


def test_get_setting_nested(fake_env: dict[str, Path]) -> None:
    _write_json(
        fake_env["user_settings"],
        {"hooks": {"before_tool": ["log_a", "log_b"]}},
    )
    assert get_setting(
        "hooks.before_tool", repo_root=fake_env["repo"]
    ) == ["log_a", "log_b"]


def test_get_setting_missing_returns_default(fake_env: dict[str, Path]) -> None:
    _write_json(fake_env["user_settings"], {"a": {"b": 1}})
    assert get_setting("a.c", repo_root=fake_env["repo"]) is None
    assert (
        get_setting("a.c", default="fallback", repo_root=fake_env["repo"])
        == "fallback"
    )
    assert get_setting("missing.path", repo_root=fake_env["repo"]) is None


def test_get_setting_segment_into_non_dict_returns_default(
    fake_env: dict[str, Path],
) -> None:
    _write_json(fake_env["user_settings"], {"a": "scalar"})
    # Trying to descend into a string -> default
    assert get_setting("a.b", default=42, repo_root=fake_env["repo"]) == 42


def test_get_setting_with_pre_resolved_settings() -> None:
    # No filesystem access at all.
    settings = {"x": {"y": {"z": 99}}}
    assert get_setting("x.y.z", settings=settings) == 99
    assert get_setting("x.y.missing", default=7, settings=settings) == 7


def test_get_setting_honours_flag_overrides(fake_env: dict[str, Path]) -> None:
    _write_json(fake_env["user_settings"], {"theme": "dark"})
    val = get_setting(
        "theme", repo_root=fake_env["repo"], flag_overrides={"theme": "light"}
    )
    assert val == "light"


# ─── Public surface from top-level package ────────────────────────────


def test_top_level_re_export() -> None:
    # Ensure the v6.4 surface is reachable from `clawagents` directly.
    import clawagents

    assert hasattr(clawagents, "resolve_settings")
    assert hasattr(clawagents, "get_setting")
    assert hasattr(clawagents, "SettingsLayer")
