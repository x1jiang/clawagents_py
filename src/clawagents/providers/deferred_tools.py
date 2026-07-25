"""Cache-preserving mid-run tool activation.

Activating a tool group mid-session normally appends new schemas to the tool
list. The tool list sits in the cached prompt *prefix*, so appending rewrites
it and the whole context is re-billed at full input rate — and it recurs every
time the agent discovers it needs another group.

The fix (ported from the Pi harness) is to leave the prefix byte-identical and
introduce the tool from the transcript instead:

* the schema is sent with ``defer_loading: true`` rather than as a prefix tool
* a ``tool_reference`` block is attached to the tool result that activated it

Only tools that were activated by a tool result *and* have not been called yet
are deferred; once the model actually calls one it becomes an ordinary tool.

This module is deliberately provider-agnostic — it decides *which* tools defer.
Emitting the provider wire format is the caller's job, because the shape (and
support) differs per provider and per model.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def _names_from_meta(meta: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for entry in meta or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            if name:
                out.append(str(name))
    return out


def split_deferred_tools(
    messages: Sequence[Any],
    schemas: Sequence[Any],
    *,
    enabled: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Partition ``schemas`` into ``(immediate, deferred_by_name)``.

    A schema defers when some tool result in ``messages`` reports having added
    it (``added_tool_names``) and no assistant turn has called it since. When
    ``enabled`` is false every schema is immediate, which is the pre-existing
    behaviour and the safe fallback for providers/models without support.
    """
    unique: dict[str, Any] = {}
    for schema in schemas or []:
        name = getattr(schema, "name", None)
        if name:
            unique[str(name)] = schema

    if not enabled:
        return list(unique.values()), {}

    # Single forward pass, matching the reference implementation: a tool is
    # deferred if it had not already been called at the point its activation
    # appears. Deciding this once, in transcript order, is what keeps the set
    # stable as the conversation grows — a tool that flip-flopped between
    # deferred and prefix would invalidate the cached prefix on every flip,
    # which is the exact cost this mechanism exists to avoid.
    used: set[str] = set()
    added: set[str] = set()
    for message in messages or []:
        role = getattr(message, "role", "")
        if role == "assistant":
            used.update(_names_from_meta(getattr(message, "tool_calls_meta", None)))
        elif role == "tool":
            for name in getattr(message, "added_tool_names", None) or []:
                if str(name) not in used:
                    added.add(str(name))

    immediate: list[Any] = []
    deferred: dict[str, Any] = {}
    for name, schema in unique.items():
        if name in added:
            deferred[name] = schema
        else:
            immediate.append(schema)
    return immediate, deferred


def tool_reference_blocks(
    message: Any,
    deferred_names: set[str],
    already_referenced: set[str],
) -> list[dict[str, str]]:
    """``tool_reference`` blocks for a tool result, in Anthropic's shape.

    Mutates ``already_referenced`` so each tool is introduced exactly once —
    a second reference to the same tool is rejected.
    """
    blocks: list[dict[str, str]] = []
    for raw in getattr(message, "added_tool_names", None) or []:
        name = str(raw)
        if name not in deferred_names or name in already_referenced:
            continue
        already_referenced.add(name)
        blocks.append({"type": "tool_reference", "tool_name": name})
    return blocks


# Models that reject client-side ``tool_reference`` blocks. Pi excludes Haiku
# for the same reason; keep this list conservative and additive.
_NO_TOOL_REFERENCE_SUBSTRINGS: tuple[str, ...] = ("haiku",)

# Substrings that identify a provider rejection *caused by* the deferred-tool
# wire shape, as opposed to any other 400. Used to self-heal exactly once.
_DEFERRED_REJECTION_MARKERS: tuple[str, ...] = (
    # Anthropic Messages
    "defer_loading",
    "tool_reference",
    "tool_name",
    # OpenAI Responses — the same mechanism, different wire nouns. Both
    # providers must be represented here or one path silently loses its
    # fail-soft retry.
    "tool_search_call",
    "tool_search_output",
    "tool_search",
)


def model_supports_tool_references(model: str) -> bool:
    """True when ``model`` accepts ``defer_loading`` / ``tool_reference``."""
    m = (model or "").strip().lower()
    if not m:
        return False
    return not any(bad in m for bad in _NO_TOOL_REFERENCE_SUBSTRINGS)


def is_deferred_tool_rejection(exc: BaseException) -> bool:
    """True when an exception looks like the API rejecting the deferred shape.

    The wire format for ``defer_loading`` / ``tool_reference`` cannot be
    verified offline, so callers use this to disable deferral and retry once
    with the ordinary full tool list rather than failing the turn. A false
    negative just surfaces the original error; a false positive costs one
    harmless retry with a shape that always works.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None and int(status or 0) not in (400, 422):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    if status is None and "400" not in text and "invalid_request" not in text:
        return False
    return any(marker in text for marker in _DEFERRED_REJECTION_MARKERS)


__all__ = [
    "is_deferred_tool_rejection",
    "model_supports_tool_references",
    "split_deferred_tools",
    "tool_reference_blocks",
]
