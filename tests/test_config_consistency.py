"""Verify that config fields, .env.example, and README stay in sync."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_example_vars() -> set[str]:
    """Extract uncommented variable names from .env.example."""
    env_file = ROOT / ".env.example"
    if not env_file.exists():
        return set()
    names = set()
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line or "=" not in line:
            continue
        names.add(line.split("=", 1)[0].strip())
    return names


def _commented_env_vars() -> set[str]:
    """Extract commented-out variable names from .env.example (the provider keys etc)."""
    env_file = ROOT / ".env.example"
    if not env_file.exists():
        return set()
    names = set()
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and "=" in stripped and not stripped.startswith("# ──"):
            var_part = stripped.lstrip("# ").split("=", 1)[0].strip()
            if var_part.isupper() or var_part.startswith("CLAW_"):
                names.add(var_part)
    return names


def _config_field_to_env() -> dict[str, str]:
    """Map EngineConfig field names to expected env var names."""
    return {
        "openai_api_key": "OPENAI_API_KEY",
        "openai_model": "OPENAI_MODEL",
        "openai_base_url": "OPENAI_BASE_URL",
        "openai_api_version": "OPENAI_API_VERSION",
        "gemini_api_key": "GEMINI_API_KEY",
        "gemini_model": "GEMINI_MODEL",
        "max_tokens": "MAX_TOKENS",
        "temperature": "TEMPERATURE",
        "context_window": "CONTEXT_WINDOW",
        "streaming": "STREAMING",
    }


def _readme_mentions(var_name: str) -> bool:
    """Check if a variable name appears in README.md."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return False
    return var_name in readme.read_text()


def test_all_config_fields_in_env_example():
    """Every EngineConfig field should have a corresponding env var in .env.example."""
    mapping = _config_field_to_env()
    active = _env_example_vars()
    commented = _commented_env_vars()
    all_vars = active | commented

    missing = []
    for field_name, env_name in mapping.items():
        if env_name not in all_vars:
            missing.append(f"{field_name} -> {env_name}")

    assert not missing, f"Config fields missing from .env.example: {missing}"


def test_all_config_fields_in_readme():
    """Every EngineConfig env var should be documented in README.md."""
    mapping = _config_field_to_env()
    missing = []
    for field_name, env_name in mapping.items():
        if not _readme_mentions(env_name):
            missing.append(env_name)

    assert not missing, f"Env vars missing from README.md: {missing}"


def test_ptrl_flags_in_env_example():
    """PTRL-related env vars should appear in .env.example."""
    ptrl_vars = {"CLAW_TRAJECTORY", "CLAW_RETHINK", "CLAW_LEARN",
                 "CLAW_PREVIEW_CHARS", "CLAW_RESPONSE_CHARS"}
    commented = _commented_env_vars()
    active = _env_example_vars()
    all_vars = active | commented

    missing = ptrl_vars - all_vars
    assert not missing, f"PTRL vars missing from .env.example: {missing}"


def test_ptrl_flags_in_readme():
    """PTRL-related env vars should be documented in README.md."""
    ptrl_vars = {"CLAW_TRAJECTORY", "CLAW_RETHINK", "CLAW_LEARN",
                 "CLAW_PREVIEW_CHARS", "CLAW_RESPONSE_CHARS"}
    missing = {v for v in ptrl_vars if not _readme_mentions(v)}
    assert not missing, f"PTRL vars missing from README.md: {missing}"


def test_env_example_exists():
    assert (ROOT / ".env.example").exists(), ".env.example not found in project root"


def test_clawagents_env_file_override_documented():
    """CLAWAGENTS_ENV_FILE should be mentioned in README."""
    assert _readme_mentions("CLAWAGENTS_ENV_FILE"), "CLAWAGENTS_ENV_FILE not documented in README.md"
