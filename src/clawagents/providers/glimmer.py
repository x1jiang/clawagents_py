"""Serving contract for Meta's Muse-Glimmer-30B on an SGLang endpoint.

Facts verified against a live deployment (2026-09-06):

- ``/v1/models`` reports ``max_model_len=196608``; the model profile keeps 20%
  headroom (see ``graph/model_profiles.py``).
- Chain-of-thought is returned on a separate ``reasoning_content`` channel
  (SGLang reasoning parser) and counted in ``usage.completion_tokens`` plus a
  top-level ``usage.reasoning_tokens``. It is NOT switchable off:
  ``chat_template_kwargs={"enable_thinking": False}``, ``reasoning_effort`` and
  ``separate_reasoning`` are all accepted and ignored.
- Every OpenAI-only request field the harness sends (``max_completion_tokens``,
  ``prompt_cache_key``, ``parallel_tool_calls``, ``store``) is accepted.

Because reasoning cannot be capped server-side, the harness caps it from the
outside: a larger default output budget so a long think does not truncate the
tool call, and the loop's output-limit recovery when it still does.
"""

from __future__ import annotations

MODEL = "Muse-Glimmer-30B"
CANONICAL_PROFILE_KEY = "muse-glimmer-30b"
HARNESS_PROFILE = "meta-glimmer"
CONTEXT_WINDOW = 196_608
# Reasoning turns of 2K-6K tokens were observed on real coding tasks; 6144 cut
# four of twelve benchmark trials mid-thought. Explicit ``max_tokens`` wins.
MAX_OUTPUT_TOKENS = 16_384


def is_glimmer_model(model: str | None) -> bool:
    value = str(model or "").strip().lower()
    return bool(value) and ("glimmer" in value or value == CANONICAL_PROFILE_KEY)
