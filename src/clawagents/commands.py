"""Central slash-command registry for clawagents.

Single source of truth for slash commands. Every consumer — CLI help, gateway
dispatch, autocomplete, REPL — derives its data from
:data:`COMMAND_REGISTRY`.

This module is intentionally tiny: it knows how to *describe* and *resolve*
slash commands, but does **not** wire them to the agent loop. Concrete
behaviour (e.g. ``/steer`` injecting a message) is implemented elsewhere
(see :mod:`clawagents.steer`); the registry just gives every consumer a
consistent vocabulary.

Usage::

    from clawagents.commands import resolve_command, format_help

    text = "/steer please switch to Python"
    resolved = resolve_command(text)
    if resolved is not None:
        cmd, args = resolved.command, resolved.args
        if cmd.name == "steer":
            run_context.steer_queue.append(args)

    print(format_help())               # full help
    print(format_help(category="Session"))  # filtered

To add a command: append a :class:`CommandDef` entry to
:data:`COMMAND_REGISTRY` (or use :func:`register_command` at runtime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional


CacheImpact = Literal["none", "deferred", "immediate"]
"""How a slash-command interacts with the LLM prompt cache.

- ``"none"`` — command does not mutate system-prompt state (most help/info
  commands, redaction toggles, history viewers). Safe to run mid-session.
- ``"deferred"`` — command **does** mutate system-prompt state (skills,
  permission mode, model, persona, …) but the change defaults to **next
  session** to preserve the active prompt cache. Use ``--now`` to opt
  into immediate invalidation. Mirrors Hermes' policy so that the common
  case (toggling permissions during a long-running run) does not blow
  away the prefix cache and force a full reload.
- ``"immediate"`` — command always invalidates the prompt cache. ``/new``,
  ``/clear``, and ``/compress`` are inherently immediate because they
  rewrite history.
"""


@dataclass(frozen=True)
class CommandDef:
    """Definition of a single slash command.

    Attributes:
        name: Canonical name without the leading slash (e.g. ``"steer"``).
        description: One-line human-readable description.
        category: Display category (``"Session"``, ``"Permission"``, ``"Info"``…).
        aliases: Alternative names. ``("bg",)`` lets ``/bg`` resolve to
            the same command.
        args_hint: Argument placeholder shown in help (e.g. ``"<prompt>"``).
        subcommands: Tab-completable subcommands (purely informational here).
        cli_only: Hide from gateway / messaging consumers.
        gateway_only: Hide from the local CLI consumer.
        cache_impact: Prompt-cache impact of this command. See
            :data:`CacheImpact`. Defaults to ``"none"``.
    """

    name: str
    description: str
    category: str = "Session"
    aliases: tuple[str, ...] = ()
    args_hint: str = ""
    subcommands: tuple[str, ...] = ()
    cli_only: bool = False
    gateway_only: bool = False
    cache_impact: CacheImpact = "none"


# Recognised forms of the "apply this change immediately" override.
# Only flag-shaped forms; bare ``now`` is reserved as a normal argument so that
# steer/queue prompts like ``/q now do the next thing`` are unaffected.
_NOW_FLAGS = frozenset({"--now", "-now"})


@dataclass(frozen=True)
class ResolvedCommand:
    """Result of :func:`resolve_command` for a parsed slash-command string.

    Attributes:
        command: The matched :class:`CommandDef`.
        args: Argument tail after the command name (with any
            ``--now`` flag stripped).
        apply_now: ``True`` when the change should be applied immediately
            and the prompt cache invalidated. Computed from
            ``command.cache_impact`` and an optional ``--now`` flag in
            the original arguments. Always ``True`` for
            ``cache_impact == "immediate"`` commands; ``True`` for
            ``"deferred"`` commands only when the user passed ``--now``.
            Always ``False`` for ``"none"`` commands.
    """

    command: CommandDef
    args: str
    apply_now: bool = False


# ─── Central registry ──────────────────────────────────────────────────────


COMMAND_REGISTRY: list[CommandDef] = [
    # Session control
    # /new and /clear rewrite history → must invalidate the cache.
    CommandDef("new", "Start a new session (fresh history + run id)", "Session",
               aliases=("reset",), cache_impact="immediate"),
    CommandDef("clear", "Clear screen and start a new session", "Session",
               cli_only=True, cache_impact="immediate"),
    CommandDef("history", "Show conversation history", "Session", cli_only=True),
    CommandDef("save", "Save the current conversation / trajectory", "Session"),
    CommandDef("retry", "Re-send the last user message to the agent", "Session"),
    CommandDef("undo", "Remove the last user/assistant exchange", "Session",
               cache_impact="immediate"),
    CommandDef("search", "Full-text search session history (SQLite sessions)", "Session",
               args_hint="<query>", cli_only=True),
    CommandDef("title", "Set a title for the current session", "Session",
               args_hint="[name]"),
    # /compress rewrites history → must invalidate the cache.
    CommandDef("compress", "Manually compress the conversation context", "Session",
               args_hint="[focus topic]", cache_impact="immediate"),
    CommandDef("stop", "Cancel the current run / kill background tasks", "Session"),

    # Mid-run nudges (see :mod:`clawagents.steer`).
    CommandDef("steer",
               "Inject guidance after the next tool call (does not interrupt)",
               "Steer", args_hint="<message>"),
    CommandDef("queue",
               "Queue a message for the next turn (does not interrupt)",
               "Steer", aliases=("q",), args_hint="<message>"),
    CommandDef("background", "Start a prompt running in the background",
               "Steer", aliases=("bg",), args_hint="<prompt>"),
    CommandDef("agents", "Show active background agents / tasks", "Steer",
               aliases=("tasks",)),

    # Permission / safety. These reshape the system-prompt-level rules and
    # so qualify as "deferred" by default — pass ``--now`` to invalidate
    # the cache and apply immediately. Mirrors Hermes prompt-cache policy.
    CommandDef("plan", "Switch to read-only plan mode (blocks write tools)",
               "Permission", cache_impact="deferred"),
    CommandDef("accept-edits", "Auto-approve write-class edits this run",
               "Permission", aliases=("accept",), cache_impact="deferred"),
    CommandDef("default", "Restore the default permission mode",
               "Permission", cache_impact="deferred"),
    CommandDef("bypass", "Disable all permission gates (DANGEROUS, opt-in)",
               "Permission", cache_impact="deferred"),
    CommandDef("redact", "Show or change output redaction (on/off/warn)",
               "Permission", args_hint="[on|off|warn]"),

    # Info / diagnostics
    CommandDef("help", "Show this help (optionally for one command)", "Info",
               args_hint="[command]"),
    CommandDef("status", "Show run status, model, token usage", "Info"),
    CommandDef("profile", "Show active profile name and home directory", "Info"),
    CommandDef("version", "Show clawagents version", "Info"),
    CommandDef("tools", "List currently registered tools", "Info"),
    CommandDef("models", "List known model profiles and routing", "Info"),
    CommandDef("trace", "Show the most recent trajectory turn", "Info"),
]


# ─── Mutable index (rebuilt on register_command) ───────────────────────────


_INDEX: dict[str, CommandDef] = {}


def _rebuild_index() -> None:
    _INDEX.clear()
    for cmd in COMMAND_REGISTRY:
        _INDEX[cmd.name] = cmd
        for alias in cmd.aliases:
            _INDEX[alias] = cmd


_rebuild_index()


# ─── Public API ────────────────────────────────────────────────────────────


def register_command(cmd: CommandDef) -> None:
    """Append a custom :class:`CommandDef` to the registry.

    Idempotent: re-registering the same canonical name overwrites the prior
    entry (and refreshes alias mappings).
    """
    # Drop existing entry with same canonical name.
    for i, existing in enumerate(list(COMMAND_REGISTRY)):
        if existing.name == cmd.name:
            COMMAND_REGISTRY[i] = cmd
            _rebuild_index()
            return
    COMMAND_REGISTRY.append(cmd)
    _rebuild_index()


def _strip_now_flag(args: str) -> tuple[str, bool]:
    """Pop a ``--now`` (or ``-now`` / ``now``) token from ``args``.

    The flag may appear anywhere in the argument list. Returns the
    cleaned-up arguments and a flag indicating whether ``--now`` was
    found.
    """
    if not args:
        return "", False
    tokens = args.split()
    rest = [t for t in tokens if t.lower() not in _NOW_FLAGS]
    found = len(rest) != len(tokens)
    return " ".join(rest), found


def resolve_command(text: str) -> Optional[ResolvedCommand]:
    """Parse ``text`` as a slash command.

    Returns ``None`` if ``text`` does not begin with ``/`` or names an
    unknown command. Trailing whitespace around the argument tail is stripped.
    A trailing ``--now`` flag is consumed (not returned in ``args``) and
    used to populate :attr:`ResolvedCommand.apply_now` for ``"deferred"``
    cache-impact commands; ``"immediate"`` commands are always
    ``apply_now=True``; ``"none"`` commands are always ``apply_now=False``.
    """
    if not text or not text.startswith("/"):
        return None
    body = text[1:].strip()
    if not body:
        return None
    head, _, tail = body.partition(" ")
    cmd = _INDEX.get(head.lower())
    if cmd is None:
        return None

    cleaned, now_flag = _strip_now_flag(tail.strip())

    if cmd.cache_impact == "immediate":
        apply_now = True
    elif cmd.cache_impact == "deferred":
        apply_now = now_flag
    else:  # "none"
        apply_now = False

    return ResolvedCommand(command=cmd, args=cleaned, apply_now=apply_now)


def list_commands(
    *, category: Optional[str] = None, audience: Optional[str] = None
) -> list[CommandDef]:
    """Return all registered commands, optionally filtered.

    Args:
        category: If given, only include commands in this category.
        audience: ``"cli"`` hides ``gateway_only`` commands; ``"gateway"``
            hides ``cli_only`` commands. ``None`` returns everything.
    """
    out: list[CommandDef] = []
    for cmd in COMMAND_REGISTRY:
        if category is not None and cmd.category != category:
            continue
        if audience == "cli" and cmd.gateway_only:
            continue
        if audience == "gateway" and cmd.cli_only:
            continue
        out.append(cmd)
    return out


def format_help(
    *, category: Optional[str] = None, audience: Optional[str] = None
) -> str:
    """Render a categorized help string for the registry.

    The output is plain text suitable for printing to a terminal or sending
    over a chat gateway. Commands are grouped by ``category`` in the order
    they first appear in :data:`COMMAND_REGISTRY`.
    """
    cmds = list_commands(category=category, audience=audience)
    if not cmds:
        return "(no commands)"

    # Preserve registration order of categories.
    seen: dict[str, list[CommandDef]] = {}
    for cmd in cmds:
        seen.setdefault(cmd.category, []).append(cmd)

    lines: list[str] = []
    for cat, group in seen.items():
        lines.append(f"=== {cat} ===")
        # Compute padding for tidy alignment.
        max_left = max(
            len(_format_left(cmd)) for cmd in group
        ) if group else 0
        for cmd in group:
            left = _format_left(cmd)
            lines.append(f"  {left.ljust(max_left)}  {cmd.description}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_left(cmd: CommandDef) -> str:
    """Return the left column for ``format_help`` (``/name [aliases] hint``)."""
    parts = [f"/{cmd.name}"]
    if cmd.aliases:
        parts.append("(" + ", ".join("/" + a for a in cmd.aliases) + ")")
    if cmd.args_hint:
        parts.append(cmd.args_hint)
    return " ".join(parts)


def all_command_names(*, include_aliases: bool = True) -> list[str]:
    """Return every recognised command name (handy for autocomplete)."""
    if include_aliases:
        return sorted(_INDEX.keys())
    return sorted(cmd.name for cmd in COMMAND_REGISTRY)


__all__ = [
    "CacheImpact",
    "CommandDef",
    "ResolvedCommand",
    "COMMAND_REGISTRY",
    "register_command",
    "resolve_command",
    "list_commands",
    "format_help",
    "all_command_names",
]
