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
    # Anthropic-style tool clearing knobs (micro-compact)
    clear_tool_keep: int | None = None
    clear_tool_trigger_ratio: float | None = None
    clear_tool_exclude: tuple[str, ...] = ()


BUILTIN_HARNESS_PROFILES: dict[str, HarnessProfile] = {
    # GPT-5.6 / Luna: huge tool schemas + multi-round search churn is the
    # dominant cost driver even when prompt-cache hit rates are excellent.
    "openai-gpt56": HarnessProfile(
        name="openai-gpt56",
        match_models=(
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-5.6-sol",
            "gpt-5.6",
            "openai.gpt-5.6",
        ),
        system_prompt_suffix=(
            "Efficiency rules (follow strictly):\n"
            "- When the user names exact file paths, call `read_file` on those "
            "paths first — do not grep/search the repo to rediscover them.\n"
            "- When the user names symbols/identifiers inside a file, prefer "
            "`grep`/`hashline_grep` then one bounded `read_file` (offset/limit). "
            "Do not page a large file sequentially just to find a symbol.\n"
            "- After you have enough facts to answer, stop and answer. Do not "
            "run extra exploratory tools.\n"
            "- Prefer one targeted read over multiple overlapping greps or reads.\n"
            "- Do not re-read the same file/range; reuse the prior tool result.\n"
            "- Optional tools (web, git, pty, …) stay hidden until you call "
            "`activate_tool_group`.\n"
            "- Do not load skills unless the task clearly needs a specialized workflow."
        ),
        # ~0.22 × 1.05M ≈ 231K — start clearing old tool dumps before Luna's
        # 272K long-context pricing cliff (see model_profiles).
        clear_tool_keep=2,
        clear_tool_trigger_ratio=0.22,
        compaction_headroom_ratio=0.7,
        # Soft warn on 2nd identical call; hard-stop on 3rd.
        loop_detection_overrides={
            "warning_threshold": 2,
            "critical_threshold": 3,
        },
    ),
    "anthropic-sonnet": HarnessProfile(
        name="anthropic-sonnet",
        match_models=("claude-sonnet", "claude-4.6-sonnet", "claude-4.5-sonnet"),
        system_prompt_suffix=(
            "Prefer concise tool use. When editing files, read before write. "
            "Batch independent reads in parallel when the runtime allows."
        ),
        compaction_headroom_ratio=0.75,
        clear_tool_keep=3,
        clear_tool_trigger_ratio=0.4,
    ),
    "anthropic-opus": HarnessProfile(
        name="anthropic-opus",
        match_models=("claude-opus", "claude-opus-4"),
        system_prompt_suffix="Think step-by-step for multi-file refactors; verify with tests before claiming done.",
        compaction_headroom_ratio=0.8,
        clear_tool_keep=4,
        clear_tool_trigger_ratio=0.45,
    ),
    "openai-codex": HarnessProfile(
        name="openai-codex",
        match_models=("gpt-5.3-codex", "gpt-5.1-codex", "gpt-5-codex", "codex"),
        system_prompt_suffix="Minimize scope. Surgical diffs only. Run verification commands before completion.",
        loop_detection_overrides={"critical_threshold": 5},
        clear_tool_keep=3,
    ),
    "local-ollama": HarnessProfile(
        name="local-ollama",
        match_models=("llama", "gemma", "mistral", "qwen", "deepseek"),
        system_prompt_suffix="Keep responses short. One tool at a time when uncertain.",
        compaction_headroom_ratio=0.65,
        clear_tool_keep=2,
        clear_tool_trigger_ratio=0.35,
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
                clear_tool_keep=spec.get("clear_tool_keep"),
                clear_tool_trigger_ratio=spec.get("clear_tool_trigger_ratio"),
                clear_tool_exclude=tuple(spec.get("clear_tool_exclude", [])),
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
