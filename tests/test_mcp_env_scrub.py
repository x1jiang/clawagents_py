"""Tests for ``scrub_env_for_stdio`` — the MCP child-process env sanitiser."""

from __future__ import annotations

import pytest

from clawagents.mcp.server import scrub_env_for_stdio


@pytest.fixture
def parent_env() -> dict[str, str]:
    return {
        # Safe / required for the child
        "PATH": "/usr/bin:/bin",
        "HOME": "/Users/alice",
        "USER": "alice",
        "LOGNAME": "alice",
        "SHELL": "/bin/bash",
        "TERM": "xterm-256color",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "America/Chicago",
        "TMPDIR": "/var/folders/xy",
        "PWD": "/Users/alice/work",
        # Definitely-secret names
        "OPENAI_API_KEY": "sk-proj-xxx",
        "ANTHROPIC_API_KEY": "sk-ant-xxx",
        "GOOGLE_API_KEY": "AIza...",
        "AWS_ACCESS_KEY_ID": "AKIAxxxx",
        "AWS_SECRET_ACCESS_KEY": "wJalrxxx",
        "GITHUB_TOKEN": "ghp_xxx",
        "DB_PASSWORD": "hunter2",
        # Random non-secret operational vars
        "EDITOR": "vim",
        "PAGER": "less",
    }


def test_default_drops_secrets(parent_env: dict[str, str]) -> None:
    out = scrub_env_for_stdio(None, parent_env=parent_env)
    assert "OPENAI_API_KEY" not in out
    assert "ANTHROPIC_API_KEY" not in out
    assert "GITHUB_TOKEN" not in out
    assert "DB_PASSWORD" not in out
    assert "AWS_ACCESS_KEY_ID" not in out


def test_default_keeps_safe_passthrough(parent_env: dict[str, str]) -> None:
    out = scrub_env_for_stdio(None, parent_env=parent_env)
    for k in ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL", "TZ", "TMPDIR", "PWD"):
        assert out[k] == parent_env[k]


def test_unknown_non_secret_vars_dropped(parent_env: dict[str, str]) -> None:
    # EDITOR/PAGER are not in the safe-passthrough list, so they must be
    # dropped — we err on the side of "minimal env".
    out = scrub_env_for_stdio(None, parent_env=parent_env)
    assert "EDITOR" not in out
    assert "PAGER" not in out


def test_user_env_overrides_passthrough(parent_env: dict[str, str]) -> None:
    out = scrub_env_for_stdio(
        {"PATH": "/sandbox/bin", "MY_VAR": "value"},
        parent_env=parent_env,
    )
    assert out["PATH"] == "/sandbox/bin"
    assert out["MY_VAR"] == "value"


def test_allowlist_inherits_specified_keys(parent_env: dict[str, str]) -> None:
    out = scrub_env_for_stdio(
        None,
        allowlist=["GITHUB_TOKEN"],
        parent_env=parent_env,
    )
    assert out["GITHUB_TOKEN"] == "ghp_xxx"
    # Other secrets still dropped
    assert "OPENAI_API_KEY" not in out


def test_allowlist_missing_key_silently_skipped(parent_env: dict[str, str]) -> None:
    out = scrub_env_for_stdio(
        None,
        allowlist=["DOES_NOT_EXIST"],
        parent_env=parent_env,
    )
    assert "DOES_NOT_EXIST" not in out


def test_inherit_safe_false_drops_everything_not_user_supplied(
    parent_env: dict[str, str],
) -> None:
    out = scrub_env_for_stdio(
        {"FOO": "bar"},
        inherit_safe=False,
        parent_env=parent_env,
    )
    assert out == {"FOO": "bar"}


def test_user_env_can_reintroduce_secret(parent_env: dict[str, str]) -> None:
    # Caller is explicitly choosing to forward this — that's allowed.
    out = scrub_env_for_stdio(
        {"OPENAI_API_KEY": "sk-deliberate"},
        parent_env=parent_env,
    )
    assert out["OPENAI_API_KEY"] == "sk-deliberate"


def test_uses_real_os_environ_when_parent_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leaked")
    out = scrub_env_for_stdio(None)
    assert out["PATH"] == "/usr/bin"
    assert "OPENAI_API_KEY" not in out


def test_lc_prefix_keys_pass_through(parent_env: dict[str, str]) -> None:
    parent_env["LC_TIME"] = "en_US.UTF-8"
    parent_env["LC_NUMERIC"] = "en_US.UTF-8"
    out = scrub_env_for_stdio(None, parent_env=parent_env)
    assert out["LC_TIME"] == "en_US.UTF-8"
    assert out["LC_NUMERIC"] == "en_US.UTF-8"
