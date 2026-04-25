"""``BrowserSession`` — lazy-loaded Playwright wrapper.

This module imports Playwright **only inside methods** so that simply
importing :mod:`clawagents.browser` works without the optional
dependency installed. The first call to :meth:`BrowserSession.start`
triggers the import; if it fails we raise
:class:`MissingPlaywrightError` with the install command.

Concurrency: each session owns one Chromium browser, one context, and
one page (single-tab). Multiple sessions can run in parallel and are
isolated by Playwright's per-context state. Per-run isolation in the
agent is handled by the existing ``RunContext`` plumbing — a forked
subagent that creates its own session gets its own Chromium.

Element targeting: ``ref`` strings (``@e1``, ``@e2``) come from the
most recent :class:`BrowserSnapshot`. The session resolves a ref by
re-running the same accessibility-tree traversal and following the
stored path index list. We do not invoke arbitrary JavaScript or
build CSS selectors — keeps the surface predictable and avoids
prompt-injection-via-selector pitfalls.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from clawagents.browser._safety import check_url
from clawagents.browser.config import BrowserConfig
from clawagents.browser.errors import (
    BrowserError,
    ElementNotFoundError,
    MissingPlaywrightError,
    NavigationBlockedError,
    SnapshotError,
)
from clawagents.browser.snapshot import (
    BrowserSnapshot,
    SnapshotElement,
    render_snapshot,
)
from clawagents.paths import get_clawagents_home

logger = logging.getLogger(__name__)


@dataclass
class BrowserHandle:
    """Lightweight serializable reference to a session.

    Useful for logging / tracing without leaking Playwright internals.
    """

    session_id: str
    provider: str
    started_at: float = field(default_factory=time.time)


def _import_playwright() -> Any:
    """Lazy import; raises :class:`MissingPlaywrightError` cleanly."""
    try:
        from playwright.async_api import async_playwright
        return async_playwright
    except ImportError as e:
        raise MissingPlaywrightError() from e


class BrowserSession:
    """A single browser session: ``async with`` to drive it.

    Typical usage::

        cfg = BrowserConfig(headless=True)
        async with BrowserSession(cfg) as bs:
            await bs.navigate("https://example.com")
            snap = await bs.snapshot()
            await bs.click(snap.elements["@e1"])

    Outside ``async with``, you must call :meth:`start` and
    :meth:`close` explicitly.

    The session owns:

    - one ``Playwright`` runtime (the value returned from
      ``async_playwright().start()``),
    - one ``Browser`` (Chromium),
    - one ``BrowserContext``,
    - one ``Page``.

    Reusing a session across multiple navigations preserves cookies
    and local storage, which is usually what the agent wants.
    """

    def __init__(
        self,
        config: Optional[BrowserConfig] = None,
        *,
        session_id: Optional[str] = None,
        provider: Optional[Any] = None,
    ) -> None:
        self.config = config or BrowserConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._provider = provider
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._last_snapshot: Optional[BrowserSnapshot] = None

    @property
    def handle(self) -> BrowserHandle:
        return BrowserHandle(
            session_id=self.session_id,
            provider=getattr(self._provider, "name", self.config.provider),
        )

    @property
    def state_dir(self) -> Path:
        """Per-session directory under ``~/.clawagents/<profile>/browser/``."""
        d = (
            get_clawagents_home(create=True)
            / "browser"
            / "sessions"
            / self.session_id
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def start(self) -> None:
        """Launch the browser if it isn't already running.

        Idempotent. Cloud providers route through their ``open`` hook;
        local provider goes straight to Playwright.
        """
        if self._page is not None:
            return

        if self.config.provider != "local":
            from clawagents.browser.providers import get_provider
            provider = self._provider or get_provider(self.config.provider)
            self._provider = provider
            await provider.open(self.config)
            return

        async_playwright = _import_playwright()
        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                args=list(self.config.chromium_args) or None,
                proxy=({"server": self.config.proxy} if self.config.proxy else None),
            )
        except Exception as e:
            await self._playwright.stop()
            self._playwright = None
            if "Executable doesn't exist" in str(e):
                raise MissingPlaywrightError(
                    "Chromium binary not found. Run `playwright install chromium` "
                    "after `pip install clawagents[browser]`."
                ) from e
            raise

        self._context = await self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            user_agent=self.config.user_agent,
            accept_downloads=self.config.accept_downloads,
        )
        self._context.set_default_timeout(self.config.timeout_ms)
        self._page = await self._context.new_page()
        logger.debug("BrowserSession %s started (provider=local)", self.session_id)

    async def close(self) -> None:
        """Tear down browser, context, and Playwright runtime.

        Idempotent. Safe to call from a ``finally`` block even when
        :meth:`start` failed mid-way.
        """
        try:
            if self._page is not None:
                try:
                    await self._page.close()
                except Exception:
                    pass
            if self._context is not None:
                try:
                    await self._context.close()
                except Exception:
                    pass
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    pass
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
        finally:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            self._last_snapshot = None

    async def __aenter__(self) -> "BrowserSession":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    def _require_page(self) -> Any:
        if self._page is None:
            raise BrowserError("Session not started; call start() or use `async with`.")
        return self._page

    # ── Navigation ────────────────────────────────────────────────

    async def navigate(self, url: str, *, wait_until: str = "load") -> str:
        """Navigate to *url* after SSRF / scheme validation."""
        err = check_url(url, allow_private=self.config.allow_private_network)
        if err is not None:
            raise NavigationBlockedError(err)
        page = self._require_page()
        await page.goto(url, wait_until=wait_until, timeout=self.config.timeout_ms)
        return page.url

    async def back(self) -> str:
        page = self._require_page()
        await page.go_back(timeout=self.config.timeout_ms)
        return page.url

    async def forward(self) -> str:
        page = self._require_page()
        await page.go_forward(timeout=self.config.timeout_ms)
        return page.url

    # ── Snapshot ──────────────────────────────────────────────────

    async def snapshot(self) -> BrowserSnapshot:
        """Capture the accessibility tree and return a :class:`BrowserSnapshot`."""
        page = self._require_page()
        try:
            tree = await page.accessibility.snapshot(interesting_only=True)
        except Exception as e:
            raise SnapshotError(f"Accessibility snapshot failed: {e}") from e

        try:
            title = await page.title()
        except Exception:
            title = ""
        snap = render_snapshot(tree, url=page.url, title=title)
        self._last_snapshot = snap
        return snap

    # ── Element resolution ────────────────────────────────────────

    async def _resolve(self, target: SnapshotElement | str) -> Any:
        """Resolve a ref string or :class:`SnapshotElement` to a Locator.

        Walks the page accessibility tree following the ``path`` indices
        we stored when building the snapshot, then converts the role+name
        into a Playwright ``role=`` locator.
        """
        page = self._require_page()
        if isinstance(target, str):
            if self._last_snapshot is None:
                raise ElementNotFoundError(
                    "No snapshot taken yet — call snapshot() first."
                )
            element = self._last_snapshot.lookup(target)
        else:
            element = target

        # Use Playwright's role-based locator. This is more robust than
        # CSS selectors against pages that change class names every
        # render. ``name`` is matched against accessible name (case-
        # sensitive substring by default).
        if element.name:
            locator = page.get_by_role(
                element.role, name=element.name, exact=True
            )
        else:
            locator = page.get_by_role(element.role)

        # Disambiguate to the first match. Snapshot already enumerated in
        # DFS order so this matches what the agent saw.
        return locator.first

    # ── Interactions ──────────────────────────────────────────────

    async def click(self, target: SnapshotElement | str) -> None:
        loc = await self._resolve(target)
        await loc.click(timeout=self.config.timeout_ms)

    async def type(
        self,
        target: SnapshotElement | str,
        text: str,
        *,
        submit: bool = False,
        clear: bool = True,
    ) -> None:
        loc = await self._resolve(target)
        if clear:
            await loc.fill("", timeout=self.config.timeout_ms)
        await loc.type(text, timeout=self.config.timeout_ms)
        if submit:
            await loc.press("Enter", timeout=self.config.timeout_ms)

    async def hover(self, target: SnapshotElement | str) -> None:
        loc = await self._resolve(target)
        await loc.hover(timeout=self.config.timeout_ms)

    async def select_option(
        self, target: SnapshotElement | str, value: str
    ) -> None:
        loc = await self._resolve(target)
        await loc.select_option(value=value, timeout=self.config.timeout_ms)

    async def wait_for(self, text: str, timeout_s: float = 30) -> None:
        page = self._require_page()
        # ``page.get_by_text`` waits up to default timeout for the string.
        try:
            await page.get_by_text(text).first.wait_for(
                timeout=int(timeout_s * 1000)
            )
        except Exception as e:
            raise BrowserError(
                f"Timed out after {timeout_s}s waiting for text {text!r}: {e}"
            ) from e

    async def screenshot(self, *, full_page: bool = False) -> str:
        """Capture a PNG and return base64-encoded bytes."""
        page = self._require_page()
        png = await page.screenshot(full_page=full_page, type="png")
        return base64.b64encode(png).decode("ascii")

    async def evaluate(self, expression: str) -> Any:
        """Run JavaScript in the page (gated by ``allow_eval``)."""
        if not self.config.allow_eval:
            raise BrowserError(
                "browser_evaluate is disabled. Set BrowserConfig.allow_eval=True "
                "to enable. Be aware this exposes arbitrary JS execution to "
                "the agent — make sure you've audited your prompt boundary."
            )
        page = self._require_page()
        return await page.evaluate(expression)

    # ── Native dialogs ────────────────────────────────────────────

    async def install_dialog_handler(
        self, *, accept: bool = True, prompt_text: Optional[str] = None
    ) -> None:
        """Auto-handle the next ``alert``/``confirm``/``prompt`` dialog."""
        page = self._require_page()

        async def _handler(dialog: Any) -> None:
            try:
                if accept:
                    await dialog.accept(prompt_text or "")
                else:
                    await dialog.dismiss()
            except Exception:
                pass

        # ``page.once`` ensures we don't accumulate handlers per dialog.
        page.once("dialog", lambda d: asyncio.create_task(_handler(d)))


__all__ = ["BrowserHandle", "BrowserSession"]
