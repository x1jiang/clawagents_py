"""Accessibility-tree snapshot → text representation with ``@eN`` refs.

The agent never sees raw HTML or CSS selectors. Instead, after every
navigation/interaction we capture Playwright's accessibility tree
(``page.accessibility.snapshot()``) and serialize it as indented text
where every interactive node gets a stable ``@e1`` / ``@e2`` ref.
The agent then targets elements by ref:

    @e1 button "Sign in"
    @e2 link "Forgot password?"
    @e3 textbox "Email"

Because refs are derived from a depth-first traversal of the same
snapshot we hand to the agent, the click/type tools can resolve them
deterministically without re-running JS in the page.

Refs are scoped per snapshot: every new snapshot resets the counter,
so older refs from a stale snapshot will fail with
``ElementNotFoundError``. The agent should re-snapshot after any
state change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# Playwright accessibility roles that produce visible text-based refs.
# Anything that is interactive or carries semantic meaning gets a ref.
_REFFED_ROLES: frozenset[str] = frozenset({
    "button", "link", "textbox", "searchbox", "checkbox", "radio",
    "combobox", "listbox", "menuitem", "tab", "switch", "slider",
    "spinbutton", "treeitem", "option", "menuitemcheckbox",
    "menuitemradio", "image",
})

# Roles that are kept in the indented output but don't get a ``@eN`` ref.
_INFORMATIONAL_ROLES: frozenset[str] = frozenset({
    "heading", "paragraph", "text", "list", "listitem", "table",
    "row", "cell", "columnheader", "rowheader", "main", "navigation",
    "banner", "contentinfo", "form", "region", "article", "section",
    "complementary", "dialog", "alertdialog", "alert", "status",
    "tooltip", "group", "separator", "presentation", "img",
})


@dataclass
class SnapshotElement:
    """A single addressable element in a :class:`BrowserSnapshot`.

    The ``selector`` field is *not* a CSS selector — it's an opaque
    payload (currently a JSON path through the accessibility tree)
    that the session knows how to resolve back to a Playwright handle
    without re-rendering. Treat it as private to ClawAgents.
    """

    ref: str  # "@e1", "@e2", …
    role: str
    name: str
    value: Optional[str] = None
    selector: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrowserSnapshot:
    """Result of :meth:`BrowserSession.snapshot`.

    Attributes:
        url: Final URL after redirects.
        title: ``document.title``.
        text: Indented text representation passed to the LLM.
        elements: Mapping ``"@e1" -> SnapshotElement``.
        truncated: ``True`` when the tree was clipped to ``MAX_NODES``.
    """

    url: str
    title: str
    text: str
    elements: dict[str, SnapshotElement]
    truncated: bool = False

    def lookup(self, ref: str) -> SnapshotElement:
        """Resolve a ``@eN`` ref or raise :class:`ElementNotFoundError`."""
        from clawagents.browser.errors import ElementNotFoundError
        if ref in self.elements:
            return self.elements[ref]
        raise ElementNotFoundError(
            f"Ref {ref!r} not in current snapshot. "
            "Re-snapshot the page (refs reset on every snapshot)."
        )


# Hard cap on the number of nodes we render to keep prompts cheap.
# Hermes uses 3000; we start at 1500.
MAX_NODES = 1500


def _walk(
    node: dict[str, Any],
    depth: int,
    elements: dict[str, SnapshotElement],
    lines: list[str],
    counter: list[int],
    path: list[int],
) -> None:
    if counter[0] >= MAX_NODES:
        return

    role = node.get("role", "") or ""
    name = (node.get("name") or "").strip()
    value = node.get("value")

    children = node.get("children", []) or []
    indent = "  " * depth
    label = ""
    if role in _REFFED_ROLES:
        counter[0] += 1
        ref = f"@e{counter[0]}"
        elements[ref] = SnapshotElement(
            ref=ref,
            role=role,
            name=name,
            value=value if isinstance(value, str) else None,
            selector={"path": list(path)},
        )
        descriptor = f"{role} \"{name}\"" if name else role
        if isinstance(value, str) and value:
            descriptor += f" value=\"{value}\""
        label = f"{ref} {descriptor}"
    elif role in _INFORMATIONAL_ROLES:
        if name:
            label = f"{role}: {name}"
        elif role and not children:
            label = role
    elif role:
        # Unknown role — surface it for debug-ability but keep it simple.
        if name:
            label = f"{role}: {name}"

    if label:
        lines.append(f"{indent}{label}")
        next_depth = depth + 1
    else:
        next_depth = depth

    for i, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        path.append(i)
        _walk(child, next_depth, elements, lines, counter, path)
        path.pop()


def render_snapshot(
    tree: Optional[dict[str, Any]],
    *,
    url: str,
    title: str,
) -> BrowserSnapshot:
    """Convert a raw Playwright accessibility tree into a snapshot.

    Accepts ``None`` (page that doesn't have an accessibility tree, e.g.
    ``about:blank``) and returns an empty snapshot in that case.
    """
    elements: dict[str, SnapshotElement] = {}
    lines: list[str] = []
    counter = [0]

    if isinstance(tree, dict):
        _walk(tree, 0, elements, lines, counter, [])

    text = "\n".join(lines) if lines else "(empty page)"
    return BrowserSnapshot(
        url=url,
        title=title,
        text=text,
        elements=elements,
        truncated=counter[0] >= MAX_NODES,
    )


def iter_elements(snapshot: BrowserSnapshot) -> Iterator[SnapshotElement]:
    yield from snapshot.elements.values()
