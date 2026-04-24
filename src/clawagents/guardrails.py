"""Input / output guardrails.

A guardrail is an async function that inspects either the incoming user
task (``InputGuardrail``) or a final assistant message
(``OutputGuardrail``) and returns a :class:`GuardrailResult`. The agent
loop enforces the result using the behavior enum:

* ``ALLOW`` — proceed unchanged (default when a guardrail passes).
* ``REJECT_CONTENT`` — replace the offending payload with the guardrail's
  ``replacement_output`` and stop the loop.
* ``RAISE_EXCEPTION`` — raise :class:`GuardrailTripwireTriggered`.

Both input and output guardrails can be attached to an agent; they fire
in registration order and short-circuit on the first non-ALLOW decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Generic, TypeVar

from clawagents.run_context import RunContext

TContext = TypeVar("TContext")


class GuardrailBehavior(str, Enum):
    ALLOW = "allow"
    REJECT_CONTENT = "reject_content"
    RAISE_EXCEPTION = "raise_exception"


@dataclass
class GuardrailResult:
    """Return value from a guardrail function."""
    behavior: GuardrailBehavior = GuardrailBehavior.ALLOW
    replacement_output: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> "GuardrailResult":
        return cls(behavior=GuardrailBehavior.ALLOW)

    @classmethod
    def reject(cls, replacement: str, *, message: str = "") -> "GuardrailResult":
        return cls(
            behavior=GuardrailBehavior.REJECT_CONTENT,
            replacement_output=replacement,
            message=message,
        )

    @classmethod
    def raise_exc(cls, message: str) -> "GuardrailResult":
        return cls(behavior=GuardrailBehavior.RAISE_EXCEPTION, message=message)


class GuardrailTripwireTriggered(Exception):
    """Raised when a guardrail chooses :attr:`GuardrailBehavior.RAISE_EXCEPTION`."""

    def __init__(
        self,
        guardrail_name: str,
        where: str,
        result: GuardrailResult,
    ):
        self.guardrail_name = guardrail_name
        self.where = where
        self.result = result
        super().__init__(
            f"[{where} guardrail '{guardrail_name}'] {result.message or 'tripwire triggered'}"
        )


InputGuardrailFn = Callable[
    [RunContext[TContext], str],
    Awaitable[GuardrailResult],
]
OutputGuardrailFn = Callable[
    [RunContext[TContext], str],
    Awaitable[GuardrailResult],
]


@dataclass
class InputGuardrail(Generic[TContext]):
    """Named wrapper around an input guardrail function."""
    name: str
    guardrail_fn: InputGuardrailFn[TContext]

    async def run(self, context: RunContext[TContext], task: str) -> GuardrailResult:
        return await self.guardrail_fn(context, task)


@dataclass
class OutputGuardrail(Generic[TContext]):
    """Named wrapper around an output guardrail function."""
    name: str
    guardrail_fn: OutputGuardrailFn[TContext]

    async def run(self, context: RunContext[TContext], output: str) -> GuardrailResult:
        return await self.guardrail_fn(context, output)


def input_guardrail(name: str | None = None):
    """Decorator: turn a function into an :class:`InputGuardrail`.

    Example::

        @input_guardrail("profanity")
        async def profanity_check(ctx, task):
            if "bad_word" in task:
                return GuardrailResult.raise_exc("profanity detected")
            return GuardrailResult.allow()
    """
    def decorator(fn: InputGuardrailFn[TContext]) -> InputGuardrail[TContext]:
        return InputGuardrail(name=name or fn.__name__, guardrail_fn=fn)
    return decorator


def output_guardrail(name: str | None = None):
    """Decorator: turn a function into an :class:`OutputGuardrail`."""
    def decorator(fn: OutputGuardrailFn[TContext]) -> OutputGuardrail[TContext]:
        return OutputGuardrail(name=name or fn.__name__, guardrail_fn=fn)
    return decorator
