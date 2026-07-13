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
    # Native Amazon Bedrock (IAM / instance role — no Anthropic API key).
    # Used when model IDs look like Bedrock (anthropic.claude-…:0, us.anthropic.…,
    # amazon.nova-…) or when PROVIDER=bedrock / profile=bedrock.
    aws_region: str = ""
    aws_profile: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    bedrock_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_tokens: int = 8192
    temperature: float = 0.0
    context_window: int = 1000000
    streaming: bool = True
    gateway_api_key: str = ""
    claw_learn_model: str = ""
    advisor_model: str = ""
    advisor_api_key: str = ""
    advisor_max_calls: int = 3


def load_config() -> EngineConfig:
    _discover_env_file()
    # Pydantic BaseSettings accepts ``_env_file`` as a runtime kwarg even though
    # it isn't a declared field — mypy doesn't see this dynamic surface.
    cfg = (
        EngineConfig(_env_file=env_file)  # type: ignore[call-arg]
        if env_file else EngineConfig()
    )
    return cfg


def is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")


def is_anthropic_model(model: str) -> bool:
    return model.lower().startswith("claude") or model.lower().startswith("anthropic")


def is_bedrock_model_id(model: str) -> bool:
    """True for Amazon Bedrock model / inference-profile IDs.

    Examples:
      - bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0  (explicit prefix)
      - anthropic.claude-3-5-sonnet-20241022-v2:0
      - us.anthropic.claude-sonnet-4-5-20250929-v1:0
      - amazon.nova-pro-v1:0
      - openai.gpt-oss-120b-1:0
    """
    lower = (model or "").lower().strip()
    if not lower:
        return False
    if lower.startswith("bedrock/"):
        return True
    # Cross-region / global inference profiles
    if lower.startswith(("us.", "eu.", "apac.", "global.")):
        return True
    # Foundation model IDs: provider.model-name-version
    for prefix in (
        "anthropic.",
        "amazon.",
        "meta.",
        "cohere.",
        "mistral.",
        "ai21.",
        "deepseek.",
        "openai.",  # Bedrock GPT-OSS
        "qwen.",
    ):
        if lower.startswith(prefix):
            return True
    return False


def strip_bedrock_prefix(model: str) -> str:
    lower = (model or "").strip()
    if lower.lower().startswith("bedrock/"):
        return lower[len("bedrock/") :]
    return lower


def get_default_model(config: EngineConfig) -> str:
    hint = os.getenv("PROVIDER", "").lower()
    if hint == "bedrock" or hint == "aws":
        return config.bedrock_model or "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
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
    if config.aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
        return config.bedrock_model or "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    return config.openai_model
