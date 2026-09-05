"""Global test isolation for background memory features.

``memory_dream`` and ``smart_memory`` default ON and write session logs /
MEMORY.md under ``<cwd>/.clawagents`` — under pytest that's the repo checkout.
Left enabled, every test run appends session logs to the repo, and once the
dream time-gate (4h) opens, dream consolidation fires inside whatever test is
running at that moment and consumes a scripted mock-LLM response. That made
failures time-dependent (green in the morning, red in the afternoon).

Tests that exercise these features enable them explicitly via
``set_overrides({...})`` or a ``features=`` run parameter, both of which take
precedence over these env defaults.
"""

from __future__ import annotations

import os

# Hard-set (not setdefault): a shell exporting these =1 would reintroduce
# the nondeterminism for everyone running the suite from that shell.
os.environ["CLAW_FEATURE_MEMORY_DREAM"] = "0"
os.environ["CLAW_FEATURE_SMART_MEMORY"] = "0"

# Keep the suite hermetic to developer machines: config discovery walks up to
# a parent-directory ``.env`` (e.g. a workspace root two levels above the
# repo), whose CLAW_FEATURE_* lines silently flip flags mid-suite the first
# time any test touches EngineConfig. The sidecar's skip switch turns that
# discovery off entirely.
os.environ["CLAWAGENTS_SKIP_DOTENV"] = "1"

# With dotenv skipped, no real provider key is present. Modern provider SDKs
# (openai>=2) raise at *construction* on an empty key, so tests that build a
# real agent purely to inspect tool registration would crash offline. Inject
# inert placeholders so construction succeeds without ever reaching a network
# (CI's intent: "must not depend on real provider keys"). A dev exporting a
# real key, or a test's own monkeypatch.setenv/delenv, still wins. CI exports
# the keys as *empty strings* (ci.yml), which ``setdefault`` would leave in
# place — treat empty as unset, otherwise the OpenAI client raises
# "Missing credentials" at construction.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
    if not os.environ.get(_k):
        os.environ[_k] = "sk-test-placeholder-not-a-real-key"

from clawagents.config import features as _features  # noqa: E402

_features.reset()
