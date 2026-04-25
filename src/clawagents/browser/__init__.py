"""Browser automation tools for ClawAgents (v6.6.0).

Lazy wrapper over Playwright (and optional cloud providers) that gives
agents a small, well-typed set of browser actions:

- ``browser_navigate(url)`` — go to a page after SSRF + scheme check.
- ``browser_snapshot()`` — accessibility-tree text representation
  with ``@e1``/``@e2`` refs the LLM can target.
- ``browser_click(ref)`` / ``browser_type(ref, text)`` — interact with
  ref-tagged elements from the most recent snapshot.
- ``browser_screenshot()`` — base64-encoded PNG payload.
- ``browser_wait_for(text, timeout_s)`` — wait for selector or text.
- ``browser_back()`` / ``browser_forward()`` / ``browser_close()``.

Playwright is an **optional** dependency::

    pip install clawagents[browser]
    playwright install chromium

Importing :mod:`clawagents.browser` works without Playwright; only
calling :class:`BrowserSession` raises :class:`MissingPlaywrightError`.

Storage of per-session state goes under ::

    ~/.clawagents/<profile>/browser/sessions/<session_id>/

Mirrored in ``clawagents/src/browser/`` (TypeScript port). Public
symbols are kept in 1:1 parity across ports.
"""

from clawagents.browser.config import BrowserConfig
from clawagents.browser.errors import (
    BrowserError,
    ElementNotFoundError,
    MissingPlaywrightError,
    NavigationBlockedError,
    SnapshotError,
)
from clawagents.browser.providers import (
    BrowserbaseProviderStub,
    BrowserUseProviderStub,
    CloudBrowserProvider,
    LocalProvider,
    get_provider,
)
from clawagents.browser.session import BrowserHandle, BrowserSession
from clawagents.browser.snapshot import BrowserSnapshot, SnapshotElement
from clawagents.browser.tools import (
    browser_back_tool,
    browser_click_tool,
    browser_close_tool,
    browser_evaluate_tool,
    browser_forward_tool,
    browser_hover_tool,
    browser_navigate_tool,
    browser_screenshot_tool,
    browser_select_option_tool,
    browser_snapshot_tool,
    browser_type_tool,
    browser_wait_for_tool,
    create_browser_tools,
)

__all__ = [
    "BrowserConfig",
    "BrowserError",
    "BrowserHandle",
    "BrowserSession",
    "BrowserSnapshot",
    "BrowserbaseProviderStub",
    "BrowserUseProviderStub",
    "CloudBrowserProvider",
    "ElementNotFoundError",
    "LocalProvider",
    "MissingPlaywrightError",
    "NavigationBlockedError",
    "SnapshotElement",
    "SnapshotError",
    "browser_back_tool",
    "browser_click_tool",
    "browser_close_tool",
    "browser_evaluate_tool",
    "browser_forward_tool",
    "browser_hover_tool",
    "browser_navigate_tool",
    "browser_screenshot_tool",
    "browser_select_option_tool",
    "browser_snapshot_tool",
    "browser_type_tool",
    "browser_wait_for_tool",
    "create_browser_tools",
    "get_provider",
]
