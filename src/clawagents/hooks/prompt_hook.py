"""PromptHook — LLM-evaluated guardrail.

A ``PromptHook`` is configured in code or in ``settings.json`` like this::

    PromptHook(
        prompt="Block tool calls that write files outside the project root.",
        model="claude-haiku-4-5",
    )

When the runtime evaluates the hook, it sends a small JSON-shaped prompt to
the configured cheap model and parses a verdict:

    {"ok": true | false, "reason": "..."}

If ``ok=false``, the hooked action is blocked with the model's stated reason
fed back into the agent's transcript. This lets users write natural-language
guardrails without writing Python.

Inspired by `claude-code-main/src/utils/hooks/execPromptHook.ts`. Reuses the
same cheap-LLM handle that the existing advisor pattern uses.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptHookVerdict:
    """The decision a PromptHook produced for one event."""

    ok: bool
    reason: Optional[str] = None
    raw_response: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class PromptHook:
    """An LLM-evaluated hook.

    Attributes
    ----------
    prompt:
        Natural-language description of when the hook should BLOCK. The
        runtime wraps it in a strict-JSON instruction.
    model:
        Cheap model spec passed to ``_resolve_model`` (e.g. ``"claude-haiku-4-5"``,
        ``"gpt-4o-mini"``, ``"gemini-flash-latest"``). Defaults to whatever
        ``ADVISOR_MODEL`` env var resolves to so users get one cheap-model
        config across both PromptHook and the advisor.
    timeout_s:
        Max seconds to wait for a verdict. On timeout, the hook FAILS OPEN
        (allows the action) and emits a warn event — a noisy hook must not
        deadlock the agent.
    """

    prompt: str
    model: Optional[str] = None
    timeout_s: float = 8.0

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("PromptHook.prompt must be non-empty")

    async def evaluate(
        self,
        payload: dict[str, Any],
        *,
        llm_resolver: Any = None,
    ) -> PromptHookVerdict:
        """Evaluate the hook against ``payload`` and return a verdict.

        ``payload`` is the event-shape dict the runtime would otherwise pass
        to a shell-based hook. ``llm_resolver`` is a callable that, given a
        model spec, returns a runnable :class:`LLMProvider`. If ``None``,
        falls back to ``clawagents.providers.llm._resolve_model`` with the
        agent's existing config.

        Always returns a verdict — never raises. On any error it logs and
        FAILS OPEN.
        """
        import asyncio

        try:
            llm = await self._resolve_llm(llm_resolver)
        except Exception as e:
            logger.warning("PromptHook: failed to resolve model %r: %s", self.model, e)
            return PromptHookVerdict(ok=True, reason=f"failed-open (no model): {e}")

        full_prompt = self._render_prompt(payload)

        from clawagents.providers.llm import LLMMessage

        messages = [
            LLMMessage(role="system", content=_VERDICT_SYSTEM),
            LLMMessage(role="user", content=full_prompt),
        ]

        try:
            response = await asyncio.wait_for(
                llm.chat(messages, on_chunk=None, cancel_event=None, tools=None),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("PromptHook: timed out after %.1fs — failing open", self.timeout_s)
            return PromptHookVerdict(ok=True, reason="failed-open (timeout)")
        except Exception as e:
            logger.warning("PromptHook: llm error %s — failing open", e)
            return PromptHookVerdict(ok=True, reason=f"failed-open (error): {e}")

        return _parse_verdict(response.content or "")

    async def _resolve_llm(self, llm_resolver: Any):
        if llm_resolver is not None:
            res = llm_resolver(self.model)
            # Allow either sync or async resolvers
            try:
                import inspect
                if inspect.isawaitable(res):
                    return await res
            except Exception:
                pass
            return res

        from clawagents.config.config import load_config
        from clawagents.providers.llm import _resolve_model  # type: ignore[attr-defined]
        config = load_config()
        return _resolve_model(
            self.model or "",
            streaming=False,
            api_key_override=None,
            context_window_override=None,
        )

    def _render_prompt(self, payload: dict[str, Any]) -> str:
        try:
            payload_json = json.dumps(payload, default=str, indent=2)[:6000]
        except Exception:
            payload_json = repr(payload)[:6000]
        return (
            f"Rule:\n{self.prompt.strip()}\n\n"
            f"Event payload (JSON):\n```json\n{payload_json}\n```\n\n"
            "Reply with ONLY a single JSON object: "
            '{"ok": true|false, "reason": "..."}.\n'
            "  - ok=true means ALLOW the action.\n"
            "  - ok=false means BLOCK; reason will be shown to the agent."
        )


# ─── Verdict parsing ──────────────────────────────────────────────────────

_VERDICT_SYSTEM = (
    "You are a strict JSON-output evaluator. Read a rule and an event "
    "payload, decide whether to allow or block, and reply with a single "
    "JSON object: {\"ok\": true|false, \"reason\": \"...\"}. Never include "
    "any other text, code fences, or commentary."
)

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _parse_verdict(text: str) -> PromptHookVerdict:
    """Parse the model's text into a :class:`PromptHookVerdict`.

    Tolerates code fences, leading/trailing prose, and missing fields. If we
    can't extract a clear verdict, FAIL OPEN (ok=true) and include the raw
    response in the reason so users can debug their hook.
    """
    if not text:
        return PromptHookVerdict(ok=True, reason="failed-open (empty response)", raw_response=text)

    # Strip code fences, take the largest JSON object substring.
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"```\s*$", "", candidate).strip()

    match = _JSON_OBJ_RE.search(candidate)
    if not match:
        return PromptHookVerdict(ok=True, reason="failed-open (no JSON found)", raw_response=text)
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return PromptHookVerdict(ok=True, reason="failed-open (bad JSON)", raw_response=text)

    ok = bool(obj.get("ok", True))
    reason = obj.get("reason")
    return PromptHookVerdict(
        ok=ok,
        reason=str(reason) if reason is not None else None,
        raw_response=text,
    )
