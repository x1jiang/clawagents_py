"""Tests for :mod:`clawagents.commands`."""

from __future__ import annotations

import pytest

from clawagents.commands import (
    COMMAND_REGISTRY,
    CommandDef,
    all_command_names,
    format_help,
    list_commands,
    register_command,
    resolve_command,
)


def test_resolve_canonical_command():
    r = resolve_command("/steer please switch to Python")
    assert r is not None
    assert r.command.name == "steer"
    assert r.args == "please switch to Python"


def test_resolve_alias_routes_to_canonical():
    r = resolve_command("/q now do the next thing")
    assert r is not None
    assert r.command.name == "queue"
    assert r.args == "now do the next thing"


def test_resolve_no_args_yields_empty_string():
    r = resolve_command("/help")
    assert r is not None
    assert r.command.name == "help"
    assert r.args == ""


def test_resolve_strips_trailing_whitespace():
    r = resolve_command("/title  My Run   ")
    assert r is not None
    assert r.args == "My Run"


def test_resolve_unknown_returns_none():
    assert resolve_command("/notarealcommand") is None


def test_resolve_non_slash_returns_none():
    assert resolve_command("hello") is None
    assert resolve_command("") is None
    assert resolve_command("/") is None


def test_resolve_is_case_insensitive_on_command_head():
    r = resolve_command("/STEER do the thing")
    assert r is not None
    assert r.command.name == "steer"
    assert r.args == "do the thing"


def test_list_commands_filters_by_category():
    info = list_commands(category="Info")
    assert all(c.category == "Info" for c in info)
    assert any(c.name == "help" for c in info)
    assert all(c.name != "steer" for c in info)


def test_list_commands_filters_by_audience():
    cli_cmds = list_commands(audience="cli")
    # No gateway-only commands should leak into a CLI-only audience.
    assert all(not c.gateway_only for c in cli_cmds)


def test_format_help_groups_by_category():
    text = format_help()
    assert "=== Session ===" in text
    assert "=== Info ===" in text
    assert "/steer" in text
    assert "/help" in text


def test_format_help_with_category_filter():
    text = format_help(category="Permission")
    assert "=== Permission ===" in text
    assert "=== Session ===" not in text


def test_register_command_appends_then_overwrites():
    custom = CommandDef("zzz_test", "ephemeral test command", "Test")
    register_command(custom)
    try:
        r = resolve_command("/zzz_test foo")
        assert r is not None and r.args == "foo"

        # Re-register with new description; should *replace* not duplicate.
        replacement = CommandDef("zzz_test", "replaced description", "Test")
        register_command(replacement)
        matches = [c for c in COMMAND_REGISTRY if c.name == "zzz_test"]
        assert len(matches) == 1
        assert matches[0].description == "replaced description"
    finally:
        # Clean up so the registry is unchanged for other tests.
        COMMAND_REGISTRY[:] = [c for c in COMMAND_REGISTRY if c.name != "zzz_test"]
        from clawagents.commands import _rebuild_index  # type: ignore[attr-defined]
        _rebuild_index()


def test_register_command_alias_resolves():
    custom = CommandDef("zzz_alias_test", "ephemeral alias test", "Test",
                        aliases=("zzz_at",))
    register_command(custom)
    try:
        r = resolve_command("/zzz_at hello")
        assert r is not None and r.command.name == "zzz_alias_test"
    finally:
        COMMAND_REGISTRY[:] = [c for c in COMMAND_REGISTRY if c.name != "zzz_alias_test"]
        from clawagents.commands import _rebuild_index  # type: ignore[attr-defined]
        _rebuild_index()


def test_all_command_names_includes_aliases_by_default():
    names = all_command_names()
    assert "queue" in names
    assert "q" in names  # alias


def test_all_command_names_canonical_only():
    names = all_command_names(include_aliases=False)
    assert "queue" in names
    assert "q" not in names


def test_no_alias_collisions_in_default_registry():
    """Two CommandDefs must not share the same name or alias."""
    seen: dict[str, str] = {}
    for cmd in COMMAND_REGISTRY:
        for n in (cmd.name, *cmd.aliases):
            assert n not in seen, (
                f"alias/name collision: '{n}' is used by both "
                f"{seen[n]!r} and {cmd.name!r}"
            )
            seen[n] = cmd.name


def test_core_commands_present():
    """Sanity check: the documented set of core commands exists."""
    names = {c.name for c in COMMAND_REGISTRY}
    for required in {
        "new", "save", "compress", "stop",
        "steer", "queue", "background", "agents",
        "plan", "accept-edits", "default", "bypass",
        "help", "status", "version",
    }:
        assert required in names, f"missing core command: {required}"


# ── Cache impact + --now flag ─────────────────────────────────────────


def test_default_cache_impact_is_none():
    r = resolve_command("/help")
    assert r is not None
    assert r.command.cache_impact == "none"
    assert r.apply_now is False


def test_immediate_cache_impact_always_applies_now():
    r = resolve_command("/new")
    assert r is not None
    assert r.command.cache_impact == "immediate"
    assert r.apply_now is True


def test_immediate_cache_impact_ignores_now_flag_but_preserves_other_args():
    """`/compress focus --now keep` keeps --now for clarity, but also strips it."""
    r = resolve_command("/compress focus --now")
    assert r is not None
    assert r.command.name == "compress"
    assert r.apply_now is True  # immediate is always now
    assert "--now" not in r.args
    assert r.args == "focus"


def test_deferred_cache_impact_defers_by_default():
    r = resolve_command("/plan")
    assert r is not None
    assert r.command.cache_impact == "deferred"
    assert r.apply_now is False  # deferred defaults to next-session


def test_deferred_cache_impact_with_now_flag_applies_immediately():
    r = resolve_command("/plan --now")
    assert r is not None
    assert r.apply_now is True
    assert r.args == ""  # --now consumed


def test_now_flag_can_be_anywhere_in_args():
    r = resolve_command("/accept-edits --now")
    assert r is not None
    assert r.apply_now is True
    assert r.args == ""

    r2 = resolve_command("/bypass extra --now noted")
    assert r2 is not None
    assert r2.apply_now is True
    assert r2.args == "extra noted"


def test_deferred_cache_commands_are_marked():
    """Permission-mode commands should default to deferred cache impact."""
    by_name = {c.name: c for c in COMMAND_REGISTRY}
    for n in ("plan", "accept-edits", "default", "bypass"):
        assert by_name[n].cache_impact == "deferred", n


def test_now_flag_does_not_promote_none_command():
    """A `cache_impact='none'` command stays apply_now=False even with --now."""
    r = resolve_command("/help --now command")
    assert r is not None
    assert r.command.cache_impact == "none"
    assert r.apply_now is False
    # --now is still stripped from args because we don't want consumers to see it.
    assert "--now" not in r.args
    assert r.args == "command"
