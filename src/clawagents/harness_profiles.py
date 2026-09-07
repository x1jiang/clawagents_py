"""Harness profiles — model-specific prompt/middleware bundles (DeepAgents 1.10.2)."""

from __future__ import annotations

import json
import re
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
    "gemma-agentic": HarnessProfile(
        name="gemma-agentic",
        match_models=("gemma4-agentic-v2", "gemma4-v2-", "gemma-4-12b-agentic-fable5-composer2.5-v2"),
        system_prompt_suffix="Delegate with explicit inputs and acceptance checks. Verify worker artifacts. Never treat incomplete work as success.",
        compaction_headroom_ratio=0.60,
        clear_tool_keep=8,
        clear_tool_trigger_ratio=0.60,
        loop_detection_overrides={"warning_threshold": 2, "critical_threshold": 3},
    ),
    # Muse-Glimmer-30B (SGLang). The provider applies Meta's documented
    # reasoning-strength directive; the loop recovers output-limit turns
    # and catches cross-tool stagnation. The model can retry a failing
    # shell approach with cosmetic changes, which the repeated-failure
    # escalation in the tool results addresses (see graph/loop_tracker.py).
    "meta-glimmer": HarnessProfile(
        name="meta-glimmer",
        match_models=("muse-glimmer-30b",),
        system_prompt_suffix=(
            "Tool efficiency:\n"
            "- Keep private reasoning short: decide in a few sentences, then act. "
            "Long deliberation hits the output limit and the turn is lost.\n"
            "- Read the exact paths named by the user before searching elsewhere.\n"
            "- Use native tool calls with the declared JSON arguments; never print tool-call markup.\n"
            "- Read, write and edit workspace files with read_file / write_file / "
            "edit_file, not with cat, echo or heredocs through execute. Use execute "
            "to run tests and commands from the workspace root with relative paths.\n"
            "- Reuse prior tool results instead of repeating identical calls.\n"
            "- If a tool fails, read the error and change approach. Never repeat a "
            "failing call unchanged, and never set unsandboxed=true.\n"
            "- Optional tools stay hidden until you call activate_tool_group.\n"
            "- After editing, run the relevant check once, then fix the failure or report the result concisely.\n"
            "- Stop when the requested result is verified."
        ),
        metadata={"initial_tools": [
            "read_file", "edit_file", "write_file", "execute", "ls", "glob",
            "grep", "activate_tool_group", "retrieve_tool_result", "ask_user",
        ]},
        compaction_headroom_ratio=0.70,
        clear_tool_keep=2,
        clear_tool_trigger_ratio=0.35,
        loop_detection_overrides={"warning_threshold": 2, "critical_threshold": 3, "progress_nudge_after": 8},
    ),
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
        # Token-anchored matching: "codellama" no longer hits via the "llama"
        # substring, so list it explicitly. Cloud-qualified ids (vendor.model,
        # geo prefixes) are excluded from this profile in resolve_harness_profile.
        match_models=("llama", "codellama", "gemma", "mistral", "qwen", "deepseek"),
        system_prompt_suffix="Keep responses short. One tool at a time when uncertain.",
        compaction_headroom_ratio=0.65,
        clear_tool_keep=2,
        clear_tool_trigger_ratio=0.35,
    ),
}


# Served-name aliases → harness profile name. Registered by create_claw_agent
# for named provider profiles (``meta`` / ``gemma-agentic``) so a custom
# deployment alias keeps its model-specific harness; every resolver in the loop
# (bootstrapper thresholds, micro-compact knobs, compaction headroom) goes
# through resolve_harness_profile, so one registration covers them all.
_MODEL_ALIASES: dict[str, str] = {}


def register_harness_alias(model: str, profile_name: str) -> None:
    key = str(model or "").strip().lower()
    if key and profile_name:
        _MODEL_ALIASES[key] = profile_name


def _profile_paths() -> list[Path]:
    return [
        Path.home() / ".clawagents" / "harness-profiles.json",
        Path.cwd() / ".clawagents" / "harness-profiles.json",
    ]


def _opt_int(value: Any, *, minimum: int = 1) -> int | None:
    """Positive int or None; strings such as ``"3"`` are accepted, junk is dropped."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= minimum else None


def _opt_ratio(value: Any) -> float | None:
    """Ratio in (0, 1]; anything else is dropped so downstream math stays sane."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out <= 0.0 or out > 1.0:  # NaN / non-positive / >100%
        return None
    return out


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if isinstance(v, (str, int, float)) and str(v).strip())
    return ()


def _harness_from_spec(name: str, spec: dict[str, Any]) -> HarnessProfile:
    overrides_raw = spec.get("loop_detection_overrides")
    overrides: dict[str, Any] = {}
    if isinstance(overrides_raw, dict):
        for key in ("warning_threshold", "critical_threshold", "progress_nudge_after"):
            coerced = _opt_int(overrides_raw.get(key), minimum=0 if key == "progress_nudge_after" else 1)
            if coerced is not None:
                overrides[key] = coerced
    metadata = spec.get("metadata")
    return HarnessProfile(
        name=name,
        match_models=_str_tuple(spec.get("match_models", [])),
        base_system_prompt=str(spec.get("base_system_prompt") or ""),
        system_prompt_suffix=str(spec.get("system_prompt_suffix") or ""),
        excluded_tools=_str_tuple(spec.get("excluded_tools", [])),
        compaction_headroom_ratio=_opt_ratio(spec.get("compaction_headroom_ratio")),
        loop_detection_overrides=overrides,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        # keep=0 means "clear nothing" downstream (``[-0:]`` keeps everything).
        clear_tool_keep=_opt_int(spec.get("clear_tool_keep"), minimum=1),
        clear_tool_trigger_ratio=_opt_ratio(spec.get("clear_tool_trigger_ratio")),
        clear_tool_exclude=_str_tuple(spec.get("clear_tool_exclude", [])),
    )


def load_harness_profiles() -> dict[str, HarnessProfile]:
    from clawagents.provider_profiles import skip_untrusted_workspace_file

    profiles = dict(BUILTIN_HARNESS_PROFILES)
    for path in _profile_paths():
        if skip_untrusted_workspace_file(path):
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        for name, spec in raw.items():
            if not isinstance(name, str) or not isinstance(spec, dict):
                continue
            profiles[name] = _harness_from_spec(name, spec)
    return profiles


# "vendor.model" (Bedrock / Mantle: ``deepseek.v3.2``, ``meta.llama3-70b``,
# ``mistral.mistral-large``) or "provider/model" ids are cloud deployments —
# never local Ollama tags like ``llama3.1`` or ``gemma4:e4b``.
_CLOUD_QUALIFIED_RE = re.compile(r"^[a-z][a-z0-9_-]*\.(?=[a-z])|/")
_GEO_PREFIX_RE = re.compile(r"^(global|us|eu|apac|ap|af|me|ca|sa)\.")


def _looks_cloud_qualified(model_lower: str) -> bool:
    return bool(_GEO_PREFIX_RE.match(model_lower) or _CLOUD_QUALIFIED_RE.search(model_lower))


def _pattern_matches(pattern: str, model_lower: str, normalized: str) -> bool:
    """Anchored start or token-boundary hit — ``"codex"`` matches ``gpt-5.3-codex``
    and ``openai.gpt-5.6`` but ``"llama"`` no longer matches ``codellama``."""
    p = pattern.strip().lower()
    if not p:
        return False
    if normalized.startswith(p) or model_lower.startswith(p):
        return True
    return re.search(r"(?:^|[-_/:.\s])" + re.escape(p), model_lower) is not None


def resolve_harness_profile(model: str | None, explicit: str | None = None) -> HarnessProfile | None:
    profiles = load_harness_profiles()
    if explicit and explicit in profiles:
        return profiles[explicit]
    if not model:
        return None
    model_lower = model.strip().lower()
    alias = _MODEL_ALIASES.get(model_lower)
    if alias and alias in profiles:
        return profiles[alias]
    from clawagents.graph.model_profiles import normalize_model_id

    normalized = normalize_model_id(model_lower)
    cloud = _looks_cloud_qualified(model_lower)
    # User-defined profiles first so a narrower custom entry can beat a
    # builtin whose pattern is a substring of the same id.
    ordered = [p for n, p in profiles.items() if n not in BUILTIN_HARNESS_PROFILES] + [
        p for n, p in profiles.items() if n in BUILTIN_HARNESS_PROFILES
    ]
    for profile in ordered:
        if profile.name == "local-ollama" and cloud:
            continue
        for pattern in profile.match_models:
            if _pattern_matches(pattern, model_lower, normalized):
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
