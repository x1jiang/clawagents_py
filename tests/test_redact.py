"""Tests for clawagents.redact — the display-layer secret scrubber."""

from __future__ import annotations

import os

import pytest

from clawagents.redact import (
    add_pattern,
    is_secret_name,
    redact,
    redact_env,
    redact_obj,
    reset_patterns,
)


@pytest.fixture(autouse=True)
def _enable_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAW_REDACT", raising=False)
    yield
    reset_patterns()


def test_openai_key_redacted():
    raw = "Bearer sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
    out = redact(raw)
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789" not in out
    assert "[REDACTED:OPENAI_KEY]" in out or "[REDACTED:BEARER]" in out


def test_anthropic_key_redacted():
    raw = "key=sk-ant-api03-thisis_a_long_key_value_1234567890"
    out = redact(raw)
    assert "sk-ant-api03-thisis_a_long_key_value_1234567890" not in out


def test_google_key_redacted():
    # Google API keys: AIza + exactly 35 [A-Za-z0-9_-] chars.
    key = "AIzaSyAa1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6"
    assert len(key) == 39  # AIza + 35
    raw = f"config: {key}"
    out = redact(raw)
    assert key not in out
    assert "[REDACTED:GOOGLE_KEY]" in out


def test_github_pat_redacted():
    raw = "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789ABCD"
    out = redact(raw)
    assert "ghp_" not in out


def test_aws_access_key_redacted():
    raw = "Hello AKIAIOSFODNN7EXAMPLE world"
    out = redact(raw)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "Hello " in out and " world" in out


def test_jwt_redacted():
    raw = (
        "Cookie: jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    out = redact(raw)
    assert "eyJhbGciOiJIUzI1NiJ9" not in out


def test_generic_assignment_redacted():
    raw = 'api_key="abcd1234efgh5678"'
    out = redact(raw)
    assert "abcd1234efgh5678" not in out


def test_safe_text_passes_through_unchanged():
    raw = "Just a regular string with no secrets."
    assert redact(raw) == raw


def test_short_alphanumeric_not_falsely_matched():
    # "abcd" is 4 chars, well under the 8-char minimum the generic-secret
    # pattern uses. We don't want to redact short identifiers.
    raw = "user=alice id=42"
    assert redact(raw) == raw


def test_redact_obj_recurses_through_dicts_and_lists():
    obj = {
        "ok": "hello",
        "leaked": "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        "nested": [
            {"inner": "AIzaSyAa1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6"},
            "plain text",
        ],
    }
    out = redact_obj(obj)
    assert out["ok"] == "hello"
    assert "[REDACTED" in out["leaked"]
    assert "[REDACTED" in out["nested"][0]["inner"]
    assert out["nested"][1] == "plain text"


def test_redact_env_masks_secret_named_keys():
    env = {
        "OPENAI_API_KEY": "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxx",
        "USER": "alice",
        "DB_PASSWORD": "super_secret_pw",
        "PATH": "/usr/bin",
    }
    out = redact_env(env)
    assert out["OPENAI_API_KEY"] == "[REDACTED]"
    assert out["DB_PASSWORD"] == "[REDACTED]"
    assert out["USER"] == "alice"
    assert out["PATH"] == "/usr/bin"


def test_add_pattern_picks_up_user_secret():
    add_pattern("INTERNAL", r"INTERNAL-[A-Z0-9]{12}")
    out = redact("token=INTERNAL-AB12CD34EF56 trailing")
    assert "INTERNAL-AB12CD34EF56" not in out
    assert "[REDACTED:INTERNAL]" in out


def test_disabled_via_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_REDACT", "0")
    raw = "key=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
    assert redact(raw) == raw


def test_warn_mode_passes_text_through(monkeypatch: pytest.MonkeyPatch, caplog):
    import logging as _logging

    monkeypatch.setenv("CLAW_REDACT", "warn")
    raw = "key=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
    with caplog.at_level(_logging.WARNING):
        out = redact(raw)
    assert out == raw  # text untouched in warn mode


def test_label_false_uses_fixed_replacement():
    raw = "AIzaSyAa1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6"
    out = redact(raw, label=False)
    assert "[REDACTED]" in out
    assert "[REDACTED:" not in out


def test_is_secret_name():
    assert is_secret_name("OPENAI_API_KEY")
    assert is_secret_name("db_password")
    assert is_secret_name("sessionToken")
    assert not is_secret_name("USER")
    assert not is_secret_name("PATH")
    assert not is_secret_name("")
