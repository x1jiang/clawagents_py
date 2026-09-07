"""Serving contract for Meta's Muse-Glimmer-30B on an SGLang endpoint.

Facts verified against a live deployment (2026-09-06):

- ``/v1/models`` reports ``max_model_len=196608``; the model profile keeps 20%
  headroom (see ``graph/model_profiles.py``).
- Reasoning is returned on a separate ``reasoning_content`` channel.
- Generic API switches are accepted but ignored by the tested deployment.
  Meta documents a system directive ``Reasoning strength: low|medium|high|xhigh``;
  live probes verify that it changes reasoning length. The harness maps its
  reasoning_effort setting to that directive (default medium).
- Every OpenAI-only request field the harness sends (``max_completion_tokens``,
  ``prompt_cache_key``, ``parallel_tool_calls``, ``store``) is accepted.

Reasoning strength is a soft model control, not a hard token cap. A larger
output budget and deadline-aware recovery handle genuinely truncated turns.
Source: https://huggingface.co/meta-models/Muse-Glimmer-30B#best-practices
"""

from __future__ import annotations

import re
from typing import Any

MODEL = "Muse-Glimmer-30B"
CANONICAL_PROFILE_KEY = "muse-glimmer-30b"
HARNESS_PROFILE = "meta-glimmer"
CONTEXT_WINDOW = 196_608
# Reasoning turns of 2K-6K tokens were observed on real coding tasks; 6144 cut
# four of twelve benchmark trials mid-thought. Explicit ``max_tokens`` wins.
MAX_OUTPUT_TOKENS = 16_384


def is_glimmer_model(model: str | None) -> bool:
    value = str(model or "").strip().lower()
    if not value:
        return False
    if "glimmer" in value or value == CANONICAL_PROFILE_KEY:
        return True
    from clawagents.harness_profiles import resolve_harness_profile

    profile = resolve_harness_profile(value)
    return profile is not None and profile.name == HARNESS_PROFILE


_STRENGTH_LINE = re.compile(r"(?im)^\s*reasoning (?:strength|effort)\s*:")


def reasoning_strength_messages(
    messages: list[dict[str, Any]], effort: str | None,
) -> list[dict[str, Any]]:
    """Apply Meta's documented control without mutating stored conversation.

    SGLang accepting ``reasoning_effort`` does not mean its template uses it.
    The model understands the system line directly. Preserve an explicit
    system directive and keep the prefix identical across a run's requests.
    """
    strength = str(effort or "medium").strip().lower()
    strength = {"none": "low", "minimal": "low", "max": "xhigh"}.get(strength, strength)
    if strength not in {"low", "medium", "high", "xhigh"}:
        strength = "medium"
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content", "")
        texts = [content] if isinstance(content, str) else [part.get("text", "") for part in content or [] if isinstance(part, dict)]
        if any(_STRENGTH_LINE.search(text) for text in texts if isinstance(text, str)):
            return messages
    directive = f"Reasoning strength: {strength}\n\n"
    result = list(messages)
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            content = message.get("content") or ""
            updated = directive + content if isinstance(content, str) else [{"type": "text", "text": directive}, *content]
            result[index] = {**message, "content": updated}
            return result
    return [{"role": "system", "content": directive.rstrip()}, *messages]
