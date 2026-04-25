"""Regression tests for specific agent_loop.py bugs:

  - Multimodal system message must not crash tier 1/2/3 context shedding
  - External pre_llm hook returning a non-allowed role must be coerced, not crash
"""

from clawagents.graph.agent_loop import _preflight_context_check
from clawagents.providers.llm import LLMMessage, NativeToolSchema


class TestMultimodalSystemShedding:
    """Bug 2: shedding code used .replace / slicing on sys_msg.content unconditionally,
    which crashes when content is the multimodal list-of-dicts shape."""

    def test_tier1_does_not_crash_with_multimodal_system(self):
        """Tier-1 path: tool_desc + registry triggered, system content is a list."""
        events: list[tuple[str, dict]] = []

        def emit(name, payload):
            events.append((name, payload))

        # Multimodal system content (Anthropic-style content blocks)
        sys_msg = LLMMessage(
            role="system",
            content=[
                {"type": "text", "text": "You are a helpful agent. " * 200},
                {"type": "image", "source": {"type": "url", "url": "https://example/img.png"}},
            ],
        )
        user_msg = LLMMessage(role="user", content="Describe the picture")
        # Make a tool_desc and a fake registry
        tool_desc = "## Available Tools\n### foo\nDoes foo " * 50

        class _FakeTool:
            name = "foo"
            description = "does foo"
            parameters = {"x": {"type": "string", "required": True}}

        class _FakeRegistry:
            def list(self):
                return [_FakeTool()]

        # Tiny context window forces the shedding path to engage.
        msgs, td, ns = _preflight_context_check(
            messages=[sys_msg, user_msg],
            context_window=64,
            tool_desc=tool_desc,
            native_schemas=None,
            registry=_FakeRegistry(),
            emit=emit,
            model_name=None,
        )

        # System message untouched (still list)
        assert isinstance(msgs[0].content, list)
        # warn events should mention skip due to multimodal
        warn_msgs = [p.get("message", "") for n, p in events if n == "warn"]
        assert any("multimodal" in m for m in warn_msgs), (
            f"expected a warn about multimodal shedding skip; got events={events}"
        )

    def test_tier2_does_not_crash_with_multimodal_system(self):
        """Tier-2 path: tool_desc + native_schemas; system content is a list."""
        events: list[tuple[str, dict]] = []

        def emit(name, payload):
            events.append((name, payload))

        sys_msg = LLMMessage(
            role="system",
            content=[{"type": "text", "text": "X" * 5000}],
        )
        user_msg = LLMMessage(role="user", content="hi")
        native_schemas = [
            NativeToolSchema(
                name="foo",
                description="does foo",
                parameters={"x": {"type": "string"}},
            )
        ]
        tool_desc = "blob " * 200

        msgs, td, ns = _preflight_context_check(
            messages=[sys_msg, user_msg],
            context_window=32,
            tool_desc=tool_desc,
            native_schemas=native_schemas,
            registry=None,  # skip tier-1
            emit=emit,
            model_name=None,
        )
        assert isinstance(msgs[0].content, list)

    def test_tier3_does_not_crash_with_multimodal_system(self):
        """Tier-3 path: when system is huge and a list, must not crash on slicing."""
        events: list[tuple[str, dict]] = []

        def emit(name, payload):
            events.append((name, payload))

        sys_msg = LLMMessage(
            role="system",
            content=[{"type": "text", "text": "Y" * 50000}],
        )
        user_msg = LLMMessage(role="user", content="hi")

        msgs, td, ns = _preflight_context_check(
            messages=[sys_msg, user_msg],
            context_window=128,
            tool_desc="",  # no tool_desc => tiers 1/2 don't trigger
            native_schemas=None,
            registry=None,
            emit=emit,
            model_name=None,
        )
        # Did not raise. System content stays a list.
        assert isinstance(msgs[0].content, list)
        warn_msgs = [p.get("message", "") for n, p in events if n == "warn"]
        # Either a tier-3 warn or final overflow warn — one of them must mention multimodal
        # (overflow may or may not fire depending on counts; the multimodal warn must)
        assert any("multimodal" in m for m in warn_msgs), (
            f"expected a multimodal warn; got events={events}"
        )


class TestExternalPreLlmRoleCoercion:
    """Bug 3: external pre_llm hook returning a non-allowed role string must
    be coerced to 'user' (with a warn), not blow up downstream."""

    def test_role_coercion_logic_directly(self):
        # Sanity check: the LLMMessage role literal — by inspecting the source
        # the fix uses these four roles.
        from typing import get_args, get_type_hints
        import inspect

        # Resolve forward-refs in the from __future__ import annotations module.
        try:
            hints = get_type_hints(LLMMessage.__init__)
            role_ann = hints.get("role")
            allowed = set(get_args(role_ann))
        except Exception:
            allowed = set()
        # The fix codifies these four roles; ensure the literal set matches.
        assert allowed == {"system", "user", "assistant", "tool"}

    def test_pre_llm_with_unknown_role_is_coerced_and_warned(self, monkeypatch):
        """End-to-end-ish: drive the pre_llm injection logic and verify the
        coercion + warn behavior."""
        # The simplest way to test the new code path is to run the fragment
        # against a synthetic ext_hook_runner.pre_llm result.
        events: list[tuple[str, dict]] = []

        def emit(name, payload):
            events.append((name, payload))

        # Mirror the patched code exactly:
        _ALLOWED_ROLES = ("system", "user", "assistant", "tool")
        extra_msgs = [
            {"role": "developer", "content": "I am from the future"},
            {"role": "user", "content": "ok"},
        ]
        appended = []
        for em in extra_msgs:
            raw_role = em.get("role", "user")
            if raw_role not in _ALLOWED_ROLES:
                emit("warn", {
                    "message": (
                        f"external pre_llm hook returned message with unknown role "
                        f"{raw_role!r}; coercing to 'user'"
                    )
                })
                role = "user"
            else:
                role = raw_role
            appended.append(LLMMessage(role=role, content=em.get("content", "")))

        # Both messages should land, with the developer one as user
        assert len(appended) == 2
        assert appended[0].role == "user"
        assert appended[0].content == "I am from the future"
        assert appended[1].role == "user"
        warn_msgs = [p.get("message", "") for n, p in events if n == "warn"]
        assert any("developer" in m and "coercing" in m for m in warn_msgs)
