"""Tests for config module."""

import os
from unittest.mock import patch

from clawagents.config.config import (
    EngineConfig,
    load_config,
    is_gemini_model,
    is_anthropic_model,
    get_default_model,
)


def test_is_gemini_model():
    assert is_gemini_model("gemini-3-flash")
    assert is_gemini_model("Gemini-Pro")
    assert not is_gemini_model("gpt-5")


def test_is_anthropic_model():
    assert is_anthropic_model("claude-sonnet-4-5")
    assert is_anthropic_model("Claude-3.5")
    assert is_anthropic_model("anthropic/claude")
    assert not is_anthropic_model("gpt-5")


def test_default_model_openai():
    config = EngineConfig(openai_api_key="test-key")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROVIDER", None)
        os.environ.pop("OPENAI_MODEL", None)
        assert config.openai_model == "gpt-5.6-terra"
        assert get_default_model(config) == "gpt-5.6-terra"


def test_default_model_openai_explicit_override():
    config = EngineConfig(openai_api_key="test-key", openai_model="gpt-5-nano")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROVIDER", None)
        assert get_default_model(config) == "gpt-5-nano"


def test_default_model_gemini_fallback():
    config = EngineConfig(
        openai_api_key="", gemini_api_key="test-key", gemini_model="gemini-3-flash"
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROVIDER", None)
        assert get_default_model(config) == "gemini-3-flash"


def test_default_model_anthropic_fallback():
    config = EngineConfig(
        openai_api_key="", gemini_api_key="",
        anthropic_api_key="test-key", anthropic_model="claude-sonnet-4-5",
    )
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROVIDER", None)
        assert get_default_model(config) == "claude-sonnet-4-5"


def test_engine_config_defaults():
    env_override = {
        "OPENAI_API_KEY": "", "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "",
        "MAX_TOKENS": "8192", "TEMPERATURE": "0", "CONTEXT_WINDOW": "1000000",
    }
    with patch.dict(os.environ, env_override, clear=False):
        config = EngineConfig(
            openai_api_key="", openai_model="gpt-5.6-terra",
            gemini_api_key="", anthropic_api_key="",
            max_tokens=8192, temperature=0.0, context_window=1000000,
        )
        assert config.max_tokens == 8192
        assert config.temperature == 0.0
        assert config.context_window == 1000000


def test_load_config_includes_advisor_env():
    env_override = {
        "ADVISOR_MODEL": "gpt-5.4",
        "ADVISOR_API_KEY": "advisor-key",
        "ADVISOR_MAX_CALLS": "7",
    }
    with patch.dict(os.environ, env_override, clear=False):
        config = load_config()
        assert config.advisor_model == "gpt-5.4"
        assert config.advisor_api_key == "advisor-key"
        assert config.advisor_max_calls == 7


def test_dotenv_override_disabled_preserves_spawn_key(tmp_path, monkeypatch):
    """VS Code SecretStorage key must not be clobbered by workspace .env."""
    import clawagents.config.config as cfg

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAWAGENTS_DOTENV_OVERRIDE", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "from-secret-storage")
    monkeypatch.delenv("CLAWAGENTS_ENV_FILE", raising=False)

    cfg._loaded = False
    cfg.env_file = None
    cfg._discover_env_file()
    assert os.environ["OPENAI_API_KEY"] == "from-secret-storage"


def test_dotenv_protects_secretstorage_provenance(tmp_path, monkeypatch):
    """Even with override=True, CLAW_KEY_SOURCES=SecretStorage keys are restored."""
    import json
    import clawagents.config.config as cfg

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAWAGENTS_DOTENV_OVERRIDE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "from-secret-storage")
    monkeypatch.setenv(
        "CLAW_KEY_SOURCES",
        json.dumps({"openai": "VS Code SecretStorage"}),
    )
    monkeypatch.delenv("CLAWAGENTS_ENV_FILE", raising=False)
    monkeypatch.delenv("CLAWAGENTS_SKIP_DOTENV", raising=False)

    cfg._loaded = False
    cfg.env_file = None
    cfg._discover_env_file()
    assert os.environ["OPENAI_API_KEY"] == "from-secret-storage"


def test_skip_dotenv_ignores_workspace_env(tmp_path, monkeypatch):
    import clawagents.config.config as cfg

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAWAGENTS_SKIP_DOTENV", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "from-secret-storage")
    monkeypatch.delenv("CLAWAGENTS_ENV_FILE", raising=False)

    cfg._loaded = False
    cfg.env_file = None
    cfg._discover_env_file()
    assert cfg.env_file is None
    assert os.environ["OPENAI_API_KEY"] == "from-secret-storage"
