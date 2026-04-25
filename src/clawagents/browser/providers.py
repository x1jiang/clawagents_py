"""Cloud provider extension point for browser sessions.

In v6.6 the only fully-implemented provider is ``LocalProvider``
(Playwright + Chromium). Cloud providers (Browserbase, Browser Use)
ship as stubs that raise :class:`NotImplementedError` so the API
shape is fixed; full implementations land in subsequent point
releases when there is demand.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from clawagents.browser.config import BrowserConfig


@runtime_checkable
class CloudBrowserProvider(Protocol):
    """Protocol every cloud provider must implement.

    Implementations are typically tiny — a few hundred lines that
    translate a :class:`BrowserConfig` into the provider's session
    creation API and return an object compatible with the same
    ``BrowserSession`` surface. See ``providers.py`` in Hermes Agent
    for reference implementations of Browserbase / Browser Use.
    """

    name: str

    async def open(self, cfg: BrowserConfig) -> Any: ...
    async def close(self, session: Any) -> None: ...


class LocalProvider:
    """Default provider: spawn a local Playwright Chromium.

    Doesn't implement :class:`CloudBrowserProvider` directly because
    :class:`BrowserSession` already handles local Playwright — this
    class exists mainly so the provider registry has a uniform shape.
    """

    name = "local"


class _UnimplementedCloudProvider:
    """Base for cloud-provider stubs."""

    name: str
    install_extra: str

    async def open(self, cfg: BrowserConfig) -> Any:  # pragma: no cover
        raise NotImplementedError(
            f"Cloud provider {self.name!r} is a stub in v6.6. "
            f"Install with `pip install clawagents[{self.install_extra}]` and "
            "wait for the full implementation in a follow-up release, or "
            "implement CloudBrowserProvider yourself and pass the instance "
            "to BrowserSession(provider=...)."
        )

    async def close(self, session: Any) -> None:  # pragma: no cover
        return None


class BrowserbaseProviderStub(_UnimplementedCloudProvider):
    name = "browserbase"
    install_extra = "browser-browserbase"


class BrowserUseProviderStub(_UnimplementedCloudProvider):
    name = "browser-use"
    install_extra = "browser-cloud"


_BUILTIN_PROVIDERS: dict[str, type[Any]] = {
    "local": LocalProvider,
    "browserbase": BrowserbaseProviderStub,
    "browser-use": BrowserUseProviderStub,
}


def get_provider(name: str) -> Any:
    """Return a provider instance for *name*, or raise ``KeyError``."""
    cls = _BUILTIN_PROVIDERS.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown browser provider {name!r}. Built-ins: "
            f"{sorted(_BUILTIN_PROVIDERS)}. Pass a CloudBrowserProvider "
            "instance directly for custom providers."
        )
    return cls()
