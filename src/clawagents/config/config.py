import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_loaded = False
env_file = None


def _discover_env_file():
    """Discover .env file lazily on first access."""
    global _loaded, env_file
    if _loaded:
        return
    _loaded = True

    cwd = Path.cwd()
    _explicit = os.environ.get("CLAWAGENTS_ENV_FILE")
    local_env = cwd / ".env"
    parent_env = cwd.parent / ".env"

    if _explicit and Path(_explicit).exists():
        env_file = Path(_explicit)
    elif local_env.exists():
        env_file = local_env
    elif parent_env.exists():
        env_file = parent_env

    from dotenv import load_dotenv
    if env_file:
        load_dotenv(env_file, override=True)


class EngineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"
    openai_base_url: str = ""
    openai_api_version: str = ""
    openai_api_type: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    max_tokens: int = 8192
    temperature: float = 0.0
    context_window: int = 1000000
    streaming: bool = True
    gateway_api_key: str = ""
    claw_learn_model: str = ""


def load_config() -> EngineConfig:
    _discover_env_file()
    cfg = EngineConfig(_env_file=env_file) if env_file else EngineConfig()
    return cfg


def is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")


def is_anthropic_model(model: str) -> bool:
    return model.lower().startswith("claude") or model.lower().startswith("anthropic")


def get_default_model(config: EngineConfig) -> str:
    hint = os.getenv("PROVIDER", "").lower()
    if hint == "gemini" and config.gemini_api_key:
        return config.gemini_model
    if hint == "anthropic" and config.anthropic_api_key:
        return config.anthropic_model
    if hint == "openai" and config.openai_api_key:
        return config.openai_model
    if config.openai_api_key:
        return config.openai_model
    if config.gemini_api_key:
        return config.gemini_model
    if config.anthropic_api_key:
        return config.anthropic_model
    return config.openai_model
