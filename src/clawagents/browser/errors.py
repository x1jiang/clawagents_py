"""Browser error types.

Kept in a tiny module so it imports cleanly without Playwright present —
callers should be able to ``except BrowserError`` without forcing the
optional dependency to be installed.
"""

from __future__ import annotations


class BrowserError(Exception):
    """Base class for all browser-related failures."""


class MissingPlaywrightError(BrowserError):
    """Raised when a browser action is attempted without Playwright installed.

    Install with::

        pip install clawagents[browser]
        playwright install chromium
    """

    def __init__(
        self,
        msg: str = (
            "Playwright is not installed. Install browser support with "
            "`pip install clawagents[browser]` and then run "
            "`playwright install chromium`."
        ),
    ) -> None:
        super().__init__(msg)


class NavigationBlockedError(BrowserError):
    """Raised when SSRF / scheme / policy guards refuse a URL."""


class SnapshotError(BrowserError):
    """Raised when accessibility-tree snapshot fails."""


class ElementNotFoundError(BrowserError):
    """Raised when a ``@eN`` ref does not resolve to an element."""
