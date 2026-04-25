"""Tests for clawagents.paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clawagents.paths import (
    DEFAULT_PROFILE,
    HOME_DIRNAME,
    WORKSPACE_DIRNAME,
    display_clawagents_home,
    display_clawagents_workspace_dir,
    get_clawagents_home,
    get_clawagents_workspace_dir,
    get_lessons_dir,
    get_sessions_dir,
    get_trajectories_dir,
    list_profiles,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Clear CLAWAGENTS_* env vars per test and route home into tmp_path."""
    for var in ("CLAWAGENTS_HOME", "CLAWAGENTS_PROFILE", "CLAWAGENTS_WORKSPACE"):
        monkeypatch.delenv(var, raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # Path.home() also reads USERPROFILE on win, but tests run on macOS / linux.
    yield


def test_default_home_uses_user_home_dir():
    home = get_clawagents_home()
    assert home == Path.home() / HOME_DIRNAME / DEFAULT_PROFILE


def test_explicit_profile_overrides_default(monkeypatch):
    home = get_clawagents_home("work")
    assert home.name == "work"
    assert home.parent == Path.home() / HOME_DIRNAME


def test_profile_env_var_picked_up(monkeypatch):
    monkeypatch.setenv("CLAWAGENTS_PROFILE", "personal")
    home = get_clawagents_home()
    assert home.name == "personal"


def test_clawagents_home_env_overrides_root(monkeypatch, tmp_path):
    sandbox = tmp_path / "sandbox"
    monkeypatch.setenv("CLAWAGENTS_HOME", str(sandbox))
    home = get_clawagents_home("work")
    assert home == sandbox / "work"


def test_clawagents_home_env_already_includes_profile(monkeypatch, tmp_path):
    """If CLAWAGENTS_HOME already ends with the profile name, don't double-suffix."""
    sandbox = tmp_path / "sandbox" / "work"
    monkeypatch.setenv("CLAWAGENTS_HOME", str(sandbox))
    home = get_clawagents_home("work")
    assert home == sandbox  # not tmp_path/sandbox/work/work


def test_create_flag_makes_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path / "h"))
    home = get_clawagents_home("p", create=True)
    assert home.exists()
    assert home.is_dir()


def test_workspace_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = get_clawagents_workspace_dir()
    assert ws == tmp_path / WORKSPACE_DIRNAME


def test_workspace_env_override(monkeypatch, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(project))
    ws = get_clawagents_workspace_dir()
    assert ws == project / WORKSPACE_DIRNAME


def test_workspace_create_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(tmp_path))
    ws = get_clawagents_workspace_dir(create=True)
    assert ws.exists()


def test_trajectories_dir_workspace_scope(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = get_trajectories_dir(scope="workspace")
    assert out == tmp_path / WORKSPACE_DIRNAME / "trajectories"


def test_trajectories_dir_home_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path / "h"))
    out = get_trajectories_dir(scope="home", profile="dev")
    assert out == tmp_path / "h" / "dev" / "trajectories"


def test_sessions_dir_home_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path / "h"))
    out = get_sessions_dir(scope="home", profile="dev")
    assert out == tmp_path / "h" / "dev" / "sessions"


def test_lessons_dir_defaults_to_home_scope(monkeypatch, tmp_path):
    """Lessons are user-level by default — they should not land under cwd."""
    (tmp_path / "ignored").mkdir()
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path / "h"))
    monkeypatch.chdir(tmp_path / "ignored")
    out = get_lessons_dir()
    assert (tmp_path / "h") in out.parents


def test_unknown_scope_raises(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        get_trajectories_dir(scope="bogus")
    with pytest.raises(ValueError):
        get_sessions_dir(scope="bogus")
    with pytest.raises(ValueError):
        get_lessons_dir(scope="bogus")


def test_list_profiles_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path / "missing"))
    assert list_profiles() == []


def test_list_profiles_returns_sorted(monkeypatch, tmp_path):
    home = tmp_path / "h"
    monkeypatch.setenv("CLAWAGENTS_HOME", str(home))
    for name in ("work", "default", "personal"):
        (home / name).mkdir(parents=True)
    (home / ".hidden").mkdir()  # should be filtered out
    (home / "junk.txt").write_text("nope")  # should be filtered (not a dir)
    assert list_profiles() == ["default", "personal", "work"]


def test_create_flag_propagates_to_sub_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path / "h"))
    out = get_trajectories_dir(scope="home", profile="dev", create=True)
    assert out.exists()
    out2 = get_sessions_dir(scope="home", profile="dev", create=True)
    assert out2.exists()


# ── display_clawagents_home / display_clawagents_workspace_dir ────────


def test_display_clawagents_home_shortens_under_home():
    """Default profile dir under $HOME should render as ~/.clawagents/default."""
    out = display_clawagents_home()
    assert out == f"~/{HOME_DIRNAME}/{DEFAULT_PROFILE}"


def test_display_clawagents_home_with_custom_profile():
    out = display_clawagents_home("work")
    assert out == f"~/{HOME_DIRNAME}/work"


def test_display_clawagents_home_falls_back_for_external_override(monkeypatch, tmp_path):
    """When CLAWAGENTS_HOME points outside $HOME, return the absolute path."""
    sandbox = tmp_path / "external"
    monkeypatch.setenv("CLAWAGENTS_HOME", str(sandbox))
    out = display_clawagents_home("work")
    assert out == str(sandbox / "work")
    assert "~" not in out


def test_display_clawagents_home_when_override_inside_home(monkeypatch):
    """If CLAWAGENTS_HOME is under $HOME, still render with ~ prefix."""
    inner = Path.home() / "alt-claw"
    monkeypatch.setenv("CLAWAGENTS_HOME", str(inner))
    out = display_clawagents_home("p")
    assert out == "~/alt-claw/p"


def test_display_clawagents_workspace_dir_under_home(monkeypatch):
    """Workspace inside $HOME renders with ~ prefix."""
    project = Path.home() / "proj"
    project.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(project))
    out = display_clawagents_workspace_dir()
    assert out == f"~/proj/{WORKSPACE_DIRNAME}"


def test_display_clawagents_workspace_dir_outside_home(monkeypatch, tmp_path):
    """Workspace outside $HOME falls back to absolute string."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("CLAWAGENTS_WORKSPACE", str(project))
    out = display_clawagents_workspace_dir()
    assert out == str(project / WORKSPACE_DIRNAME)
    assert "~" not in out
