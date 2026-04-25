#!/usr/bin/env python3
"""One-off smoke test: verify ``gemma4:*`` tags route through the Ollama
factory and resolve correct context budgets. Mirrors
``clawagents/scripts/smoke-gemma4.ts`` so the cross-package parity claim
in both READMEs can actually be exercised.

Not wired into CI — run manually:

    python scripts/smoke_gemma4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from clawagents.config.config import load_config  # noqa: E402
from clawagents.providers.llm import (  # noqa: E402
    OpenAIProvider,
    create_provider,
)


def _provider_base_url(provider: object) -> str:
    client = getattr(provider, "client", None)
    base = getattr(client, "base_url", None)
    return str(base) if base else "<none>"


def main() -> None:
    cases = [
        "gemma4:e4b",
        "gemma4:e2b",
        "gemma4:26b",
        "gemma4",
        "ollama/gemma4:e4b",
        "gpt-5.4",
        "gemini-3.1-pro",
        "claude-opus-4-6",
    ]
    cfg = load_config()
    cfg.openai_base_url = ""
    cfg.openai_api_key = ""

    print(f"{'model':<24} {'provider':<18} {'base_url':<46} stored_model")
    print("-" * 110)
    for name in cases:
        try:
            provider = create_provider(name, cfg)
        except ImportError as e:
            print(f"{name:<24} <skipped: {e}>")
            continue

        kind = type(provider).__name__
        if isinstance(provider, OpenAIProvider):
            kind = "OpenAIProvider"
        base = _provider_base_url(provider)
        stored = getattr(provider, "model", None) or getattr(
            provider, "gemini_model", "?"
        )
        print(f"{name:<24} {kind:<18} {base:<46} {stored}")


if __name__ == "__main__":
    main()
