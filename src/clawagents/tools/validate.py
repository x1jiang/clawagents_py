"""Tool Parameter Validation + Lenient Type Coercion.

Validates tool arguments against their declared parameter schemas before
execution, catching type mismatches and missing required params early.
Includes lenient coercion (e.g. "42" → 42) to handle common LLM output quirks.

Inspired by ToolUniverse's BaseTool.validate_parameters().
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


class _ToolLike(Protocol):
    name: str
    parameters: Dict[str, Dict[str, Any]]


@dataclass
class ValidationError:
    param: str
    message: str


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    coerced: Dict[str, Any] = field(default_factory=dict)


def _coerce_number(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _coerce_integer(v: Any) -> Optional[int]:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, str):
        try:
            f = float(v)
            if f == int(f):
                return int(f)
        except ValueError:
            pass
    return None


def _coerce_boolean(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lower = v.lower()
        if lower in ("true", "1", "yes"):
            return True
        if lower in ("false", "0", "no"):
            return False
    if isinstance(v, (int, float)):
        return v != 0
    return None


def _coerce_string(v: Any) -> Optional[str]:
    if isinstance(v, str):
        return v
    if v is not None:
        return str(v)
    return None


def _coerce_array(v: Any) -> Optional[list]:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _coerce_object(v: Any) -> Optional[dict]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


_COERCERS: Dict[str, Callable[[Any], Any]] = {
    "number": _coerce_number,
    "integer": _coerce_integer,
    "boolean": _coerce_boolean,
    "string": _coerce_string,
    "array": _coerce_array,
    "object": _coerce_object,
}


def validate_tool_args(tool: _ToolLike, args: Dict[str, Any]) -> ValidationResult:
    """Validate and coerce tool arguments against the tool's declared parameter schema."""
    errors: List[ValidationError] = []
    coerced = dict(args)
    params = tool.parameters

    for name, info in params.items():
        if info.get("required") and (args.get(name) is None):
            errors.append(ValidationError(
                param=name,
                message=f'required parameter "{name}" is missing',
            ))

    for name, value in args.items():
        schema = params.get(name)
        if not schema:
            continue

        expected_type = (schema.get("type") or "string").lower()
        coercer = _COERCERS.get(expected_type)

        if coercer:
            coerced_value = coercer(value)
            if coerced_value is None and value is not None:
                errors.append(ValidationError(
                    param=name,
                    message=f'expected type "{expected_type}" but got {type(value).__name__}: {str(value)[:80]}',
                ))
            elif coerced_value is not None:
                coerced[name] = coerced_value

    return ValidationResult(valid=len(errors) == 0, errors=errors, coerced=coerced)


def format_validation_errors(errors: List[ValidationError]) -> str:
    return "\n".join(f"- {e.param}: {e.message}" for e in errors)
