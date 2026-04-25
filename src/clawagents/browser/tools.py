"""Function-tool wrappers exposing a :class:`BrowserSession` to the agent.

Each public function here is callable by the LLM. We bind a single
:class:`BrowserSession` instance per :func:`create_browser_tools` call,
so a multi-agent run that wants isolated browsers should call this
once per agent (typically inside a ``forked_subagent``).

Tool naming follows Hermes' convention so prompts portable across
both frameworks work without rewrites:

    browser_navigate, browser_snapshot, browser_click, browser_type,
    browser_screenshot, browser_wait_for, browser_back, browser_forward,
    browser_close, browser_hover, browser_select_option, browser_evaluate.

The module-level ``browser_*_tool`` symbols at the bottom are *bound*
to a default global session — convenient for quick demos but not
recommended for production multi-agent code (use
:func:`create_browser_tools` instead).
"""

from __future__ import annotations

from typing import Any, Optional

from clawagents.browser._safety import check_url
from clawagents.browser.config import BrowserConfig
from clawagents.browser.errors import BrowserError, NavigationBlockedError
from clawagents.browser.session import BrowserSession
from clawagents.function_tool import FunctionTool, function_tool


def create_browser_tools(
    session: Optional[BrowserSession] = None,
    *,
    config: Optional[BrowserConfig] = None,
) -> list[FunctionTool]:
    """Build the standard set of browser tools bound to *session*.

    If *session* is ``None`` we lazily instantiate one with *config*
    (or a default :class:`BrowserConfig`). The session is auto-started
    on first use.

    Returns a list of :class:`FunctionTool` ready to register::

        agent.register_tools(create_browser_tools())
    """
    bs = session if session is not None else BrowserSession(config or BrowserConfig())

    async def _ensure_started() -> None:
        if bs._page is None:
            await bs.start()

    @function_tool(
        name="browser_navigate",
        description=(
            "Navigate the browser to a URL. Blocks SSRF/loopback/private "
            "addresses unless allow_private_network is enabled. Returns "
            "the final URL after redirects."
        ),
    )
    async def browser_navigate(url: str) -> str:
        """
        Args:
            url: Absolute http(s) URL to navigate to.
        """
        # Validate URL *before* spawning Chromium so SSRF rejection
        # never has to launch a browser.
        err = check_url(url, allow_private=bs.config.allow_private_network)
        if err is not None:
            raise NavigationBlockedError(err)
        await _ensure_started()
        return await bs.navigate(url)

    @function_tool(
        name="browser_snapshot",
        description=(
            "Capture the page's accessibility tree as a text outline. "
            "Each interactive element gets a @eN ref you can pass to "
            "browser_click / browser_type. Refs reset every snapshot."
        ),
    )
    async def browser_snapshot() -> str:
        await _ensure_started()
        snap = await bs.snapshot()
        header = f"# {snap.title}\nURL: {snap.url}\n"
        body = snap.text
        if snap.truncated:
            body += "\n\n…(snapshot truncated; page exceeds 1500 nodes)"
        return header + "\n" + body

    @function_tool(
        name="browser_click",
        description="Click the element identified by a @eN ref from the latest snapshot.",
    )
    async def browser_click(ref: str) -> str:
        """
        Args:
            ref: Element ref like '@e1' from browser_snapshot.
        """
        await _ensure_started()
        await bs.click(ref)
        return f"Clicked {ref}"

    @function_tool(
        name="browser_type",
        description=(
            "Type text into the input element identified by a @eN ref. "
            "By default replaces existing content; pass submit=true to "
            "press Enter after typing."
        ),
    )
    async def browser_type(
        ref: str, text: str, submit: bool = False, clear: bool = True
    ) -> str:
        """
        Args:
            ref: Element ref like '@e1'.
            text: Text to type.
            submit: Press Enter after typing.
            clear: Clear the field before typing (default true).
        """
        await _ensure_started()
        await bs.type(ref, text, submit=submit, clear=clear)
        return f"Typed into {ref}" + (" and submitted" if submit else "")

    @function_tool(
        name="browser_hover",
        description="Hover the cursor over an element by ref.",
    )
    async def browser_hover(ref: str) -> str:
        await _ensure_started()
        await bs.hover(ref)
        return f"Hovered {ref}"

    @function_tool(
        name="browser_select_option",
        description="Select an option in a <select>-style combobox by ref + value.",
    )
    async def browser_select_option(ref: str, value: str) -> str:
        await _ensure_started()
        await bs.select_option(ref, value)
        return f"Selected {value!r} in {ref}"

    @function_tool(
        name="browser_screenshot",
        description=(
            "Capture a PNG of the current viewport (or full page) and "
            "return base64-encoded bytes. Pass full_page=true for a "
            "scroll-the-whole-page screenshot."
        ),
    )
    async def browser_screenshot(full_page: bool = False) -> str:
        await _ensure_started()
        return await bs.screenshot(full_page=full_page)

    @function_tool(
        name="browser_wait_for",
        description="Wait until visible text appears on the page (or timeout).",
    )
    async def browser_wait_for(text: str, timeout_s: float = 30.0) -> str:
        """
        Args:
            text: Text that must appear on the page.
            timeout_s: Maximum seconds to wait. Default 30s.
        """
        await _ensure_started()
        await bs.wait_for(text, timeout_s=timeout_s)
        return f"Found text {text!r}"

    @function_tool(
        name="browser_back",
        description="Go back one step in browser history.",
    )
    async def browser_back() -> str:
        await _ensure_started()
        return await bs.back()

    @function_tool(
        name="browser_forward",
        description="Go forward one step in browser history.",
    )
    async def browser_forward() -> str:
        await _ensure_started()
        return await bs.forward()

    @function_tool(
        name="browser_evaluate",
        description=(
            "Run JavaScript in the page (DISABLED unless allow_eval=true "
            "in BrowserConfig). Returns the JSON-serialised result."
        ),
    )
    async def browser_evaluate(expression: str) -> str:
        # Gate first — don't spend time launching Chromium just to refuse.
        if not bs.config.allow_eval:
            raise BrowserError(
                "browser_evaluate is disabled. Set BrowserConfig.allow_eval=True "
                "to enable."
            )
        await _ensure_started()
        result = await bs.evaluate(expression)
        return repr(result)

    @function_tool(
        name="browser_close",
        description="Close the browser session and free Chromium resources.",
    )
    async def browser_close() -> str:
        await bs.close()
        return "Browser closed."

    return [
        browser_navigate,
        browser_snapshot,
        browser_click,
        browser_type,
        browser_hover,
        browser_select_option,
        browser_screenshot,
        browser_wait_for,
        browser_back,
        browser_forward,
        browser_evaluate,
        browser_close,
    ]


# ── Module-level tool instances bound to a singleton session ────────
#
# Convenience for one-off scripts. Production code should call
# :func:`create_browser_tools` per agent so multi-agent runs don't
# share Chromium state.

_default_session: Optional[BrowserSession] = None


def _get_default_session() -> BrowserSession:
    global _default_session
    if _default_session is None:
        _default_session = BrowserSession()
    return _default_session


_default_tools = create_browser_tools(session=_get_default_session())
(
    browser_navigate_tool,
    browser_snapshot_tool,
    browser_click_tool,
    browser_type_tool,
    browser_hover_tool,
    browser_select_option_tool,
    browser_screenshot_tool,
    browser_wait_for_tool,
    browser_back_tool,
    browser_forward_tool,
    browser_evaluate_tool,
    browser_close_tool,
) = _default_tools


__all__ = [
    "create_browser_tools",
    "browser_navigate_tool",
    "browser_snapshot_tool",
    "browser_click_tool",
    "browser_type_tool",
    "browser_hover_tool",
    "browser_select_option_tool",
    "browser_screenshot_tool",
    "browser_wait_for_tool",
    "browser_back_tool",
    "browser_forward_tool",
    "browser_evaluate_tool",
    "browser_close_tool",
]
