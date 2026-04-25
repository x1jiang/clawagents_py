"""Hermetic tests for the browser module.

We verify three things WITHOUT requiring Playwright to be installed:

1. The module imports cleanly even when ``playwright`` is missing.
2. URL safety checks block SSRF / non-http(s) schemes correctly.
3. Snapshot rendering converts an accessibility tree into the
   ``@e1``/``@e2`` text format we promise the agent.

We also verify that the function tools have the right names so prompts
that target ``browser_navigate`` / ``browser_snapshot`` etc. line up.
"""

from __future__ import annotations

import asyncio

import pytest


def test_module_imports_without_playwright() -> None:
    """Importing must work even on machines without Playwright."""
    import clawagents.browser as bm
    assert bm.BrowserConfig().headless is True
    assert bm.MissingPlaywrightError.__bases__[0] is bm.BrowserError


def test_default_config_is_safe() -> None:
    from clawagents.browser import BrowserConfig
    cfg = BrowserConfig()
    assert cfg.headless is True
    assert cfg.allow_eval is False
    assert cfg.allow_private_network is False
    assert cfg.timeout_ms == 30_000


def test_config_with_overrides_is_immutable_copy() -> None:
    from clawagents.browser import BrowserConfig
    cfg = BrowserConfig()
    cfg2 = cfg.with_overrides(headless=False, allow_eval=True)
    assert cfg.headless is True and cfg.allow_eval is False
    assert cfg2.headless is False and cfg2.allow_eval is True
    assert cfg is not cfg2


def test_url_safety_rejects_loopback() -> None:
    from clawagents.browser._safety import check_url
    err = check_url("http://127.0.0.1:8080/", allow_private=False)
    assert err is not None and "private" in err.lower()


def test_url_safety_rejects_file_scheme_by_default() -> None:
    from clawagents.browser._safety import check_url
    err = check_url("file:///etc/passwd", allow_private=False)
    assert err is not None and "scheme" in err.lower()


def test_url_safety_allows_file_when_private_allowed() -> None:
    from clawagents.browser._safety import check_url
    err = check_url("file:///tmp/test.html", allow_private=True)
    assert err is None


def test_url_safety_rejects_unknown_scheme() -> None:
    from clawagents.browser._safety import check_url
    assert check_url("ftp://example.com/", allow_private=True) is not None
    assert check_url("javascript:alert(1)", allow_private=True) is not None


def test_url_safety_accepts_public_https() -> None:
    from clawagents.browser._safety import check_url
    assert check_url("https://example.com/path", allow_private=False) is None


def test_snapshot_rendering_produces_refs() -> None:
    from clawagents.browser.snapshot import render_snapshot

    tree = {
        "role": "WebArea",
        "name": "Example",
        "children": [
            {
                "role": "heading",
                "name": "Welcome",
                "children": [],
            },
            {
                "role": "textbox",
                "name": "Email",
                "children": [],
            },
            {
                "role": "button",
                "name": "Sign in",
                "children": [],
            },
        ],
    }
    snap = render_snapshot(tree, url="https://example.com", title="Example")
    assert snap.url == "https://example.com"
    assert "@e1" in snap.elements
    assert "@e2" in snap.elements
    assert snap.elements["@e1"].role == "textbox"
    assert snap.elements["@e1"].name == "Email"
    assert snap.elements["@e2"].role == "button"
    assert snap.elements["@e2"].name == "Sign in"
    assert "@e1 textbox" in snap.text
    assert "@e2 button" in snap.text


def test_snapshot_rendering_handles_none_tree() -> None:
    from clawagents.browser.snapshot import render_snapshot

    snap = render_snapshot(None, url="about:blank", title="")
    assert snap.url == "about:blank"
    assert snap.elements == {}
    assert snap.text == "(empty page)"


def test_snapshot_lookup_raises_on_unknown_ref() -> None:
    from clawagents.browser.errors import ElementNotFoundError
    from clawagents.browser.snapshot import render_snapshot

    snap = render_snapshot(None, url="about:blank", title="")
    with pytest.raises(ElementNotFoundError):
        snap.lookup("@e99")


def test_create_browser_tools_naming_and_count() -> None:
    """Tools are named exactly what the prompt expects."""
    from clawagents.browser import create_browser_tools

    tools = create_browser_tools()
    names = sorted(t.name for t in tools)
    assert names == sorted([
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_hover",
        "browser_select_option",
        "browser_screenshot",
        "browser_wait_for",
        "browser_back",
        "browser_forward",
        "browser_evaluate",
        "browser_close",
    ])
    nav = next(t for t in tools if t.name == "browser_navigate")
    assert "url" in nav.parameters and nav.parameters["url"]["required"] is True


def test_navigate_blocks_ssrf_without_starting_browser() -> None:
    """Calling browser_navigate on a private IP must reject before launching Chromium."""
    from clawagents.browser import BrowserConfig, BrowserSession, create_browser_tools

    cfg = BrowserConfig(allow_private_network=False)
    session = BrowserSession(cfg)
    tools = create_browser_tools(session=session)
    nav = next(t for t in tools if t.name == "browser_navigate")

    async def go() -> None:
        result = await nav.execute({"url": "http://127.0.0.1/"})
        assert result.success is False
        assert "private" in (result.error or "").lower()

    asyncio.run(go())
    assert session._page is None


def test_evaluate_disabled_by_default() -> None:
    from clawagents.browser import BrowserConfig, BrowserSession, create_browser_tools

    cfg = BrowserConfig(allow_eval=False)
    session = BrowserSession(cfg)
    tools = create_browser_tools(session=session)
    ev = next(t for t in tools if t.name == "browser_evaluate")

    async def go() -> None:
        # Even before start(), the gate must trigger before Playwright launch.
        result = await ev.execute({"expression": "1+1"})
        assert result.success is False
        assert "allow_eval" in (result.error or "")

    asyncio.run(go())


def test_session_state_dir_is_under_clawagents_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWAGENTS_HOME", str(tmp_path))
    from clawagents.browser import BrowserSession

    bs = BrowserSession(session_id="test-sess")
    d = bs.state_dir
    assert d.exists()
    assert "browser" in d.parts
    assert "test-sess" in d.parts
