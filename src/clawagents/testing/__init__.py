"""Testing aids for clawagents.

Importable explicitly — NOT exposed via the top-level ``clawagents``
package on purpose. These modules exist for tests and offline e2e.

Currently exports:

    - :mod:`clawagents.testing.mock_provider` — deterministic fake LLM
      service, drop-in for ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL`` /
      ``GOOGLE_API_BASE_URL``.
"""

from clawagents.testing.mock_provider import (
    MockLLMService,
    Scenario,
    BUILTIN_SCENARIOS,
)

__all__ = ["MockLLMService", "Scenario", "BUILTIN_SCENARIOS"]
