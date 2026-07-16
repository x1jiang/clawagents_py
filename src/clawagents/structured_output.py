"""Structured output helpers — map one json_schema to provider wire formats.

Grok Build parity: Chat Completions response_format, Responses text.format,
Anthropic output_config.format.
"""

from __future__ import annotations

from typing import Any, Optional


STRUCTURED_OUTPUT_NAME = "structured_output"


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


__all__ = [
    "STRUCTURED_OUTPUT_NAME",
    "schema_from_output_type",
    "openai_chat_response_format",
    "openai_responses_text_format",
    "anthropic_output_format",
    "gemini_response_schema",
]
