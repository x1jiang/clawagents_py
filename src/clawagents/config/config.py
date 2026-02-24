import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Priority: clawagents_py/.env > parent dir .env (openclawVSdeepagents/.env)
cwd = Path.cwd()
local_env = cwd / ".env"
parent_env = cwd.parent / ".env"

env_file = None
if local_env.exists():
    env_file = local_env
elif parent_env.exists():
    env_file = parent_env

from dotenv import load_dotenv
if env_file:
    load_dotenv(env_file, override=True)


class EngineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-5-nano"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    max_tokens: int = 8192
    context_window: int = 128000
    streaming: bool = True


def load_config() -> EngineConfig:
    return EngineConfig()


def is_gemini_model(model: str) -> bool:
    """Infer provider from model name: 'gemini*' → True, everything else → False."""
    return model.lower().startswith("gemini")


def get_default_model(config: EngineConfig) -> str:
    """Pick the default model. PROVIDER env var is a hint when both API keys exist."""
    hint = os.getenv("PROVIDER", "").lower()
    if hint == "gemini" and config.gemini_api_key:
        return config.gemini_model
    if hint == "openai" and config.openai_api_key:
        return config.openai_model
    if config.openai_api_key:
        return config.openai_model
    if config.gemini_api_key:
        return config.gemini_model
    return config.openai_model
