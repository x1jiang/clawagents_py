"""URL safety checks for browser navigation.

Re-uses :func:`clawagents.tools.web._is_private_address` so we are
consistent with ``web_fetch``: the same hosts that are SSRF-blocked
in the fetch tool are blocked here.

We also enforce the scheme allow-list (``http``, ``https``, plus
``file`` and ``data`` for tests when ``allow_private_network=True``).
"""

from __future__ import annotations

from urllib.parse import urlparse

from clawagents.tools.web import _is_private_address

_SAFE_SCHEMES = {"http", "https"}
_TEST_SCHEMES = {"file", "data", "about"}


def check_url(url: str, *, allow_private: bool) -> str | None:
    """Validate *url* before navigating to it.

    Returns an error message if the URL is rejected, ``None`` if it's safe.
    Mirrors the behavior of ``clawagents.tools.web._validate_hop`` but is
    aware of browser-only schemes (``file://`` etc.) used by tests.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return f"Invalid URL: {url}"

    scheme = (parsed.scheme or "").lower()
    if not scheme:
        return f"Invalid URL: {url}"

    if scheme in _TEST_SCHEMES:
        if not allow_private:
            return (
                f"Refusing scheme {scheme!r}: only allowed when "
                "BrowserConfig.allow_private_network is True."
            )
        return None

    if scheme not in _SAFE_SCHEMES:
        return (
            f"Refusing scheme {scheme!r}: browser_navigate only allows "
            "http and https (or file/data when allow_private_network=True)."
        )

    if not allow_private:
        host = parsed.hostname or ""
        if not host or _is_private_address(host):
            return (
                f"Refusing to navigate to {host or url!r}: resolves to a "
                "private/loopback/link-local/reserved address. Set "
                "BrowserConfig.allow_private_network=True to override."
            )
    return None
