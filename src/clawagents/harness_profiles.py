"""Harness profiles — model-specific prompt/middleware bundles (DeepAgents 1.10.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HarnessProfile:
    name: str
    match_models: tuple[str, ...] = ()
    base_system_prompt: str = ""
    system_prompt_suffix: str = ""
    excluded_tools: tuple[str, ...] = ()
    compaction_headroom_ratio: float | None = None
    loop_detection_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


BUILTIN_HARNESS_PROFILES: dict[str, HarnessProfile] = {
    "anthropic-sonnet": HarnessProfile(
        name="anthropic-sonnet",
        match_models=("claude-sonnet", "claude-4.6-sonnet", "claude-4.5-sonnet"),
        system_prompt_suffix=(
            "Prefer concise tool use. When editing files, read before write. "
            "Batch independent reads in parallel when the runtime allows."
        ),
        compaction_headroom_ratio=0.75,
    ),
    "anthropic-opus": HarnessProfile(
        name="anthropic-opus",
        match_models=("claude-opus", "claude-opus-4"),
        system_prompt_suffix="Think step-by-step for multi-file refactors; verify with tests before claiming done.",
        compaction_headroom_ratio=0.8,
    ),
    "openai-codex": HarnessProfile(
        name="openai-codex",
        match_models=("gpt-5.3-codex", "gpt-5.1-codex", "gpt-5-codex", "codex"),
        system_prompt_suffix="Minimize scope. Surgical diffs only. Run verification commands before completion.",
        loop_detection_overrides={"critical_threshold": 5},
    ),
    "local-ollama": HarnessProfile(
        name="local-ollama",
        match_models=("llama", "gemma", "mistral", "qwen", "deepseek"),
        system_prompt_suffix="Keep responses short. One tool at a time when uncertain.",
        compaction_headroom_ratio=0.65,
    ),
}


def _profile_paths() -> list[Path]:
    return [
        Path.home() / ".clawagents" / "harness-profiles.json",
        Path.cwd() / ".clawagents" / "harness-profiles.json",
    ]


def load_harness_profiles() -> dict[str, HarnessProfile]:
    profiles = dict(BUILTIN_HARNESS_PROFILES)
    for path in _profile_paths():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        for name, spec in raw.items():
            if not isinstance(spec, dict):
                continue
            profiles[name] = HarnessProfile(
                name=name,
                match_models=tuple(spec.get("match_models", [])),
                base_system_prompt=str(spec.get("base_system_prompt", "")),
                system_prompt_suffix=str(spec.get("system_prompt_suffix", "")),
                excluded_tools=tuple(spec.get("excluded_tools", [])),
                compaction_headroom_ratio=spec.get("compaction_headroom_ratio"),
                loop_detection_overrides=dict(spec.get("loop_detection_overrides", {})),
                metadata=dict(spec.get("metadata", {})),
            )
    return profiles


def resolve_harness_profile(model: str | None, explicit: str | None = None) -> HarnessProfile | None:
    profiles = load_harness_profiles()
    if explicit and explicit in profiles:
        return profiles[explicit]
    if not model:
        return None
    model_lower = model.lower()
    for profile in profiles.values():
        for prefix in profile.match_models:
            if model_lower.startswith(prefix.lower()) or prefix.lower() in model_lower:
                return profile
    return None


def apply_harness_profile_to_prompt(base: str, profile: HarnessProfile | None) -> str:
    if not profile:
        return base
    if profile.base_system_prompt:
        base = profile.base_system_prompt
    if profile.system_prompt_suffix:
        base = f"{base.rstrip()}\n\n{profile.system_prompt_suffix.strip()}"
    return base
