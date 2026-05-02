"""Named provider profile resolution.

Profiles are an additive convenience layer over the existing constructor/env
surface. They never replace explicit arguments: caller-provided ``model``,
``api_key``, ``base_url``, and ``api_version`` always win.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_version: str = ""


@dataclass(frozen=True)
class ResolvedProviderProfile:
    profile: str | None
    provider: str
    model: str | None
    api_key: str | None
    base_url: str | None
    api_version: str | None


BUILTIN_PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "openai": ProviderProfile("openai", "openai", "gpt-5-nano"),
    "gemini": ProviderProfile("gemini", "gemini", "gemini-3-flash-preview"),
    "anthropic": ProviderProfile("anthropic", "anthropic", "claude-sonnet-4-5"),
    "ollama": ProviderProfile("ollama", "openai", "llama3.1", "http://localhost:11434/v1", "ollama"),
}


def _profile_paths() -> list[Path]:
    return [
        Path.home() / ".clawagents" / "profiles.json",
        Path.cwd() / ".clawagents" / "profiles.json",
    ]


def load_provider_profiles(paths: list[Path] | None = None) -> dict[str, ProviderProfile]:
    profiles = dict(BUILTIN_PROVIDER_PROFILES)
    for path in paths or _profile_paths():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, json.JSONDecodeError):
            continue
        src = raw.get("profiles", raw) if isinstance(raw, dict) else {}
        if not isinstance(src, dict):
            continue
        for name, value in src.items():
            if not isinstance(name, str) or not isinstance(value, dict):
                continue
            profiles[name] = _profile_from_mapping(name, value)
    return profiles


def _profile_from_mapping(name: str, value: dict[str, Any]) -> ProviderProfile:
    provider = str(value.get("provider") or value.get("api_format") or "openai")
    model = str(value.get("model") or value.get("default_model") or "")
    base_url = str(value.get("base_url") or value.get("baseUrl") or "")
    api_key = str(value.get("api_key") or value.get("apiKey") or "")
    api_version = str(value.get("api_version") or value.get("apiVersion") or "")
    return ProviderProfile(name=name, provider=provider, model=model, base_url=base_url, api_key=api_key, api_version=api_version)


def resolve_provider_profile(
    profile: str | None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
) -> ResolvedProviderProfile:
    if not profile:
        return ResolvedProviderProfile(None, os.getenv("PROVIDER", "auto"), model, api_key, base_url, api_version)

    profiles = load_provider_profiles()
    selected = profiles.get(profile)
    if selected is None:
        raise ValueError(f"Unknown provider profile: {profile}")

    return ResolvedProviderProfile(
        profile=profile,
        provider=selected.provider,
        model=model if model is not None else selected.model,
        api_key=api_key if api_key is not None else (selected.api_key or None),
        base_url=base_url if base_url is not None else (selected.base_url or None),
        api_version=api_version if api_version is not None else (selected.api_version or None),
    )

