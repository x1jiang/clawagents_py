"""Structured output helpers — map one json_schema to provider wire formats.

Grok Build parity: Chat Completions response_format, Responses text.format,
Anthropic output_config.format.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Literal, Sequence, TypeVar

T = TypeVar("T")

STRUCTURED_OUTPUT_NAME = "structured_output"

_SCHEMA_UNSUPPORTED_MARKERS: tuple[str, ...] = (
    "response_format",
    "json_schema",
    "output_config",
    "response_schema",
    "structured_output",
)


def schema_from_output_type(output_type: Any) -> dict[str, Any] | None:
    """Best-effort JSON Schema from a Pydantic model / dataclass / dict schema."""
    if output_type is None:
        return None
    if isinstance(output_type, dict):
        return output_type
    # Pydantic v2
    if hasattr(output_type, "model_json_schema"):
        try:
            return output_type.model_json_schema()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(output_type, "schema"):
        try:
            return output_type.schema()
        except Exception:
            pass
    return None


def openai_chat_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": STRUCTURED_OUTPUT_NAME,
            "schema": schema,
            "strict": True,
        },
    }


def openai_responses_text_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": STRUCTURED_OUTPUT_NAME,
            "schema": schema,
            "strict": True,
        }
    }


def anthropic_output_format(schema: dict[str, Any]) -> dict[str, Any]:
    # Anthropic Messages: bare schema under output_config.format
    return {
        "type": "json_schema",
        "schema": schema,
    }


def gemini_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini response_schema (subset — pass through JSON Schema loosely)."""
    return schema


def verdict_json_schema() -> dict[str, Any]:
    """Strict JSON Schema for guardrail / PromptHook verdicts."""
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["ok", "reason"],
        "additionalProperties": False,
    }


def choice_json_schema(choices: Sequence[str]) -> dict[str, Any]:
    """Strict JSON Schema for blind anonymous candidate selection."""
    allowed = sorted({str(c).strip() for c in choices if str(c).strip()})
    choice_prop: dict[str, Any] = {"type": "string"}
    if allowed:
        choice_prop["enum"] = allowed
    return {
        "type": "object",
        "properties": {
            "why": {"type": "string"},
            "choice": choice_prop,
        },
        "required": ["why", "choice"],
        "additionalProperties": False,
    }


def local_engine_structured_args(
    schema_type: Literal["json_schema", "regex", "grammar", "structural_tag"],
    schema: dict[str, Any] | str,
    *,
    engine: Literal["deepinfer", "vllm", "sglang"] = "deepinfer",
) -> dict[str, Any]:
    """Build grammar-constrained decoding args for local inference engines.

    Inspired by DeepInfer's XGrammar ``so_args`` contract
    (``{"type": "json_schema" | "regex" | "grammar" | "structural_tag", "schema": ...}``),
    alongside vLLM and SGLang guided-decoding fields.
    """
    schema_str = (
        json.dumps(schema, sort_keys=True)
        if isinstance(schema, dict)
        else str(schema)
    )
    if engine == "deepinfer":
        return {
            "so_args": {
                "type": schema_type,
                "schema": schema_str,
            }
        }
    if engine == "vllm":
        if schema_type == "json_schema":
            return {"guided_json": schema if isinstance(schema, dict) else schema_str}
        if schema_type == "regex":
            return {"guided_regex": schema_str}
        return {"guided_grammar": schema_str}
    if engine == "sglang":
        if schema_type == "json_schema":
            return {"json_schema": schema_str}
        if schema_type == "regex":
            return {"regex": schema_str}
        return {"ebnf": schema_str}
    raise ValueError(f"Unsupported local engine {engine!r}")


async def invoke_structured_control(
    llm: Any,
    messages: list[Any],
    *,
    schema: dict[str, Any],
    parse_fn: Callable[[str], T],
    timeout_s: float | None = None,
) -> tuple[T, Any]:
    """Invoke a control-plane LLM call with grammar-constrained schema decoding.

    Temporarily attaches ``_structured_json_schema`` to ``llm`` (restoring any
    prior value on exit). If the provider endpoint explicitly rejects structured
    output wire parameters, retries once without ``_structured_json_schema`` so
    prompt-level parsing continues cleanly.
    """
    had_attr = hasattr(llm, "_structured_json_schema")
    prev_schema = getattr(llm, "_structured_json_schema", None)
    schema_applied = False
    try:
        setattr(llm, "_structured_json_schema", schema)
        schema_applied = True
    except (AttributeError, TypeError):
        schema_applied = False

    async def _do_call() -> Any:
        coro = llm.chat(messages, on_chunk=None, cancel_event=None, tools=None)
        if timeout_s is not None and timeout_s > 0:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        return await coro

    try:
        try:
            response = await _do_call()
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            msg_lower = str(exc).lower()
            if schema_applied and any(m in msg_lower for m in _SCHEMA_UNSUPPORTED_MARKERS):
                try:
                    setattr(llm, "_structured_json_schema", None)
                except (AttributeError, TypeError):
                    pass
                response = await _do_call()
            else:
                raise
    finally:
        if schema_applied:
            try:
                if had_attr:
                    setattr(llm, "_structured_json_schema", prev_schema)
                else:
                    delattr(llm, "_structured_json_schema")
            except (AttributeError, TypeError):
                try:
                    setattr(llm, "_structured_json_schema", prev_schema)
                except (AttributeError, TypeError):
                    pass

    raw_text = str(getattr(response, "content", "") or "")
    return parse_fn(raw_text), response


__all__ = [
    "STRUCTURED_OUTPUT_NAME",
    "schema_from_output_type",
    "openai_chat_response_format",
    "openai_responses_text_format",
    "anthropic_output_format",
    "gemini_response_schema",
    "verdict_json_schema",
    "choice_json_schema",
    "local_engine_structured_args",
    "invoke_structured_control",
]
