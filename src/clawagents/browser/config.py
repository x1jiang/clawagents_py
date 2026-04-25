"""``BrowserConfig`` — dataclass capturing knobs the agent can tune.

Defaults are conservative: headless, no JS eval, no private-IP nav,
no extra Chromium args. Agents that need wider permission must opt in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BrowserConfig:
    """Configuration for a :class:`BrowserSession`.

    Attributes:
        headless: Run Chromium without a visible window. Default ``True``.
        viewport_width / viewport_height: Initial viewport size.
        user_agent: Override Chromium's UA string. ``None`` keeps default.
        proxy: Optional ``http://user:pass@host:port`` proxy string.
        accept_downloads: Whether the page may trigger file downloads
            (handled in-process, not yet exposed to the agent).
        timeout_ms: Default action timeout passed to Playwright. 30s.
        allow_private_network: When ``False`` (default), navigation to
            loopback / RFC1918 / link-local / metadata IPs is rejected
            with :class:`NavigationBlockedError`.
        allow_eval: When ``False`` (default), ``browser_evaluate(js)``
            is disabled. Set to ``True`` to expose arbitrary JS execution
            to the agent. **Security-sensitive — leave off unless you
            have audited the prompt boundary.**
        provider: One of ``"local"`` (Playwright Chromium),
            ``"browserbase"``, ``"browser-use"``. Cloud providers are
            stubs in v6.6 and raise on use.
        chromium_args: Extra command-line flags forwarded to Chromium.
        downloads_dir: Override the default downloads location.
    """

    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    accept_downloads: bool = False
    timeout_ms: int = 30_000
    allow_private_network: bool = False
    allow_eval: bool = False
    provider: str = "local"
    chromium_args: list[str] = field(default_factory=list)
    downloads_dir: Optional[str] = None

    def with_overrides(self, **kwargs: Any) -> "BrowserConfig":
        """Return a copy with the given attributes overridden."""
        from dataclasses import replace
        return replace(self, **kwargs)
