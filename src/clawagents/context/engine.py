"""Pluggable Context Engine — allows alternative context management strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from clawagents.providers.llm import LLMProvider, LLMMessage


@dataclass
class ContextEngineConfig:
    context_window: int = 1_000_000
    model_name: Optional[str] = None
    budget_ratio: float = 0.75
    soft_trim_ratio: float = 0.60


class ContextEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    async def bootstrap(self, config: ContextEngineConfig) -> None:
        pass

    @abstractmethod
    async def after_turn(
        self,
        messages: list[LLMMessage],
        llm: LLMProvider,
        config: ContextEngineConfig,
    ) -> list[LLMMessage]: ...

    async def compact(
        self,
        messages: list[LLMMessage],
        llm: LLMProvider,
        config: ContextEngineConfig,
    ) -> list[LLMMessage] | None:
        return None

    async def cleanup(self) -> None:
        pass


class DefaultContextEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "default"

    async def after_turn(
        self,
        messages: list[LLMMessage],
        llm: LLMProvider,
        config: ContextEngineConfig,
    ) -> list[LLMMessage]:
        return messages


_registered_engines: dict[str, type[ContextEngine]] = {}


def register_context_engine(name: str, engine_cls: type[ContextEngine]) -> None:
    _registered_engines[name] = engine_cls


def resolve_context_engine(name: str | None = None) -> ContextEngine:
    if not name or name == "default":
        return DefaultContextEngine()
    cls = _registered_engines.get(name)
    if not cls:
        available = ", ".join(_registered_engines.keys()) or "default"
        raise ValueError(f'Unknown context engine: "{name}". Available: {available}')
    return cls()


def list_context_engines() -> list[str]:
    return ["default", *_registered_engines.keys()]
