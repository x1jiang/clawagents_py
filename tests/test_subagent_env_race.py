"""Regression test for the subagent credential-proxy env race.

The OLD code mutated ``os.environ`` directly inside ``do_run()`` and restored it
in a ``finally`` block. With concurrent sub-agent invocations (``use_queue=False``
+ parallel calls + credential_proxy enabled) the second invocation captured the
FIRST one's overrides as its "original env", and its restore step then stamped
those overrides back into place — leaving stale ``OPENAI_BASE_URL`` /
``ANTHROPIC_BASE_URL`` pointing at proxy URLs that had already been stopped.

The fix wraps the env-mutate / run / env-restore window in an ``asyncio.Lock``
so concurrent runs serialise on the env mutation but still benefit from
parallelism in the no-proxy path.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import patch

import pytest

from clawagents.providers.llm import LLMProvider, LLMResponse, LLMMessage
from clawagents.tools.registry import ToolRegistry, ToolResult
from clawagents.tools.subagent import TaskTool, SubAgentSpec


class _CountingMockLLM(LLMProvider):
    name = "mock-counting"

    def __init__(self):
        self._idx = 0

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
        # Yield once so the two concurrent subagent runs can interleave inside
        # do_run() — without this, asyncio.gather may run them sequentially.
        await asyncio.sleep(0.01)
        return LLMResponse(content="done", model="mock", tokens_used=1)


class _StubProxy:
    """Stand-in for ``CredentialProxy`` that doesn't bind a real port."""

    instances: list["_StubProxy"] = []

    def __init__(self, headers):
        self.headers = headers
        self.started = False
        self.stopped = False
        _StubProxy.instances.append(self)

    def start(self) -> str:
        self.started = True
        return f"http://localhost:0/stub-{len(_StubProxy.instances)}"

    def stop(self) -> None:
        self.stopped = True


def _reset_features_cache():
    """Clear the cached feature flags so env-var changes take effect."""
    import clawagents.config.features as _ff
    _ff._resolved = None


@pytest.mark.asyncio
async def test_concurrent_subagents_with_credential_proxy_do_not_corrupt_env():
    """Two concurrent subagent runs with credential_proxy=True must restore the
    original env (real API keys, no leftover proxy BASE_URL)."""
    _StubProxy.instances.clear()

    sentinel_open = "real-open-key"
    sentinel_anth = "real-anth-key"

    env_overrides = {
        "CLAW_FEATURE_CREDENTIAL_PROXY": "1",
        "OPENAI_API_KEY": sentinel_open,
        "ANTHROPIC_API_KEY": sentinel_anth,
    }
    # Snapshot any pre-existing values so we can put them back at the end.
    _saved = {k: os.environ.get(k) for k in (
        "CLAW_FEATURE_CREDENTIAL_PROXY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
    )}
    try:
        for k, v in env_overrides.items():
            os.environ[k] = v
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        _reset_features_cache()

        spec = SubAgentSpec(
            name="worker", description="t", credential_proxy=True, max_iterations=1,
        )
        reg = ToolRegistry()
        llm = _CountingMockLLM()
        tool = TaskTool(llm=llm, tools=reg, subagents=[spec], use_queue=False)

        # Patch CredentialProxy at its source module so the lazy import inside
        # do_run() picks up the stub.
        with patch(
            "clawagents.sandbox.credential_proxy.CredentialProxy", _StubProxy
        ):
            results = await asyncio.gather(
                tool.execute({"description": "t1", "agent": "worker"}),
                tool.execute({"description": "t2", "agent": "worker"}),
            )

        # Both completed successfully
        assert all(r.success for r in results), f"results: {results}"

        # Both proxies were started and stopped
        assert len(_StubProxy.instances) == 2
        assert all(p.started and p.stopped for p in _StubProxy.instances), (
            f"started/stopped state: {[(p.started, p.stopped) for p in _StubProxy.instances]}"
        )

        # CRITICAL: env restored to sentinels, no stale proxy BASE_URL left over.
        assert os.environ.get("OPENAI_API_KEY") == sentinel_open, (
            f"OPENAI_API_KEY corrupted: {os.environ.get('OPENAI_API_KEY')!r} "
            f"(expected {sentinel_open!r})"
        )
        assert os.environ.get("ANTHROPIC_API_KEY") == sentinel_anth
        assert "OPENAI_BASE_URL" not in os.environ, (
            f"stale OPENAI_BASE_URL leaked: {os.environ.get('OPENAI_BASE_URL')!r}"
        )
        assert "ANTHROPIC_BASE_URL" not in os.environ
    finally:
        # Restore environment regardless of test outcome
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _reset_features_cache()
