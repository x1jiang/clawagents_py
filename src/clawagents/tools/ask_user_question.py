"""AskUserQuestion — structured human-in-the-loop multiple-choice tool.

Inspired by Claude Code's `AskUserQuestionTool`. The agent emits a small
batch (1-3) of structured questions; the host UI is responsible for
collecting answers via a callback supplied by the embedder.

Public entry point:

    from clawagents import ask_user_question_tool

    async def my_ui(questions):
        # Render `questions`, collect answers from a TUI/web/Telegram, return:
        return {
            q["header"]: {"question": q["question"], "answer": "Yes"}
            for q in questions
        }

    tool = ask_user_question_tool(on_ask=my_ui)

If ``on_ask`` is ``None`` the tool still validates input but returns an
error explaining that no UI is registered (rather than hanging on
``input()``). This keeps the tool safe to install in headless / channel
gateways where there is no human to answer.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from clawagents.tools.registry import Tool, ToolResult


# ─── Spec types ────────────────────────────────────────────────────────────


class QuestionSpec(TypedDict, total=False):
    question: str
    header: str
    options: List[str]
    multiSelect: bool


class AnswerSpec(TypedDict, total=False):
    question: str
    answer: str
    free_text: str


OnAskCallback = Callable[[List[QuestionSpec]], Awaitable[Dict[str, AnswerSpec]]]


# ─── Constants ─────────────────────────────────────────────────────────────

QUESTION_MAX_CHARS = 256
HEADER_MAX_CHARS = 80
MIN_QUESTIONS = 1
MAX_QUESTIONS = 3
MIN_OPTIONS = 2
MAX_OPTIONS = 4
OTHER_OPTION = "Other (please specify)"


# ─── Validation ────────────────────────────────────────────────────────────


def _validate(questions: Any) -> Optional[str]:
    if not isinstance(questions, list):
        return "`questions` must be an array"
    if not (MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS):
        return f"must provide {MIN_QUESTIONS}-{MAX_QUESTIONS} questions, got {len(questions)}"

    seen_headers: set[str] = set()
    for idx, q in enumerate(questions):
        if not isinstance(q, dict):
            return f"question[{idx}] must be an object"

        question = q.get("question")
        header = q.get("header")
        options = q.get("options")

        if not isinstance(question, str) or not question.strip():
            return f"question[{idx}].question must be a non-empty string"
        if len(question) > QUESTION_MAX_CHARS:
            return f"question[{idx}].question exceeds {QUESTION_MAX_CHARS} characters"

        if not isinstance(header, str) or not header.strip():
            return f"question[{idx}].header must be a non-empty string"
        if len(header) > HEADER_MAX_CHARS:
            return f"question[{idx}].header exceeds {HEADER_MAX_CHARS} characters"
        if header in seen_headers:
            return f"duplicate header: {header!r}"
        seen_headers.add(header)

        if not isinstance(options, list):
            return f"question[{idx}].options must be an array"
        if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
            return (
                f"question[{idx}].options must have {MIN_OPTIONS}-{MAX_OPTIONS} "
                f"items, got {len(options)}"
            )
        if not all(isinstance(o, str) and o.strip() for o in options):
            return f"question[{idx}].options entries must be non-empty strings"
        if len(set(options)) != len(options):
            return f"question[{idx}].options entries must be unique"

        ms = q.get("multiSelect", False)
        if not isinstance(ms, bool):
            return f"question[{idx}].multiSelect must be a boolean"

    return None


def _inject_other(questions: List[Dict[str, Any]]) -> List[QuestionSpec]:
    """Append the implicit 'Other (please specify)' option to each question.

    If the user already supplied an option matching ``OTHER_OPTION`` we leave
    it alone (avoids creating a duplicate that would fail validation later).
    """
    out: List[QuestionSpec] = []
    for q in questions:
        opts = list(q.get("options") or [])
        if OTHER_OPTION not in opts:
            opts.append(OTHER_OPTION)
        out.append(
            {
                "question": q["question"],
                "header": q["header"],
                "options": opts,
                "multiSelect": bool(q.get("multiSelect", False)),
            }
        )
    return out


# ─── Tool ──────────────────────────────────────────────────────────────────


class AskUserQuestionTool:
    name = "ask_user_question"
    description = (
        "Ask the user 1-3 structured multiple-choice questions in a single batch. "
        "Use when you need clarification with a small, well-defined option set. "
        "Each question must have a short header (≤80 chars), the question text "
        "(≤256 chars), and 2-4 options. An implicit 'Other (please specify)' "
        "option is always appended so the user can break out of the menu."
    )
    parameters: Dict[str, Dict[str, Any]] = {
        "questions": {
            "type": "array",
            "description": (
                "Array of 1-3 question objects, each with `question` (string), "
                "`header` (string), `options` (array of 2-4 unique strings) and "
                "optional `multiSelect` (boolean, default false). Headers must be "
                "unique across the batch."
            ),
            "required": True,
            "items": {"type": "object"},
        }
    }

    def __init__(self, on_ask: Optional[OnAskCallback] = None):
        self._on_ask = on_ask

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        questions = args.get("questions")
        err = _validate(questions)
        if err is not None:
            return ToolResult(success=False, output="", error=err)

        prepared = _inject_other(questions)  # type: ignore[arg-type]

        if self._on_ask is None:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "ask_user_question: no UI registered. Pass an `on_ask` "
                    "callback to ask_user_question_tool() to enable HITL prompts."
                ),
            )

        try:
            answers = await self._on_ask(prepared)
        except Exception as exc:  # surface the UI failure as a tool error
            return ToolResult(success=False, output="", error=f"ask_user_question UI error: {exc}")

        if not isinstance(answers, dict):
            return ToolResult(
                success=False, output="",
                error="ask_user_question: on_ask callback must return a dict keyed by header",
            )

        # Filter to known headers and pass through whatever shape the UI returned
        # (allowing optional `free_text`).
        out: Dict[str, Dict[str, Any]] = {}
        for q in prepared:
            header = q["header"]
            ans = answers.get(header)
            if not isinstance(ans, dict):
                out[header] = {"question": q["question"], "answer": "", "free_text": ""}
                continue
            entry: Dict[str, Any] = {
                "question": q["question"],
                "answer": str(ans.get("answer", "")),
            }
            if "free_text" in ans and ans["free_text"]:
                entry["free_text"] = str(ans["free_text"])
            out[header] = entry

        return ToolResult(success=True, output=json.dumps(out, ensure_ascii=False))


def ask_user_question_tool(on_ask: Optional[OnAskCallback] = None) -> Tool:
    """Factory that returns a configured AskUserQuestionTool.

    Pass an async callable that renders the structured questions through your
    UI (TUI, web, channel adapter, …) and returns a dict mapping each
    ``header`` to ``{"question": str, "answer": str, "free_text"?: str}``.
    """
    return AskUserQuestionTool(on_ask=on_ask)


__all__ = [
    "AskUserQuestionTool",
    "ask_user_question_tool",
    "QuestionSpec",
    "AnswerSpec",
    "OnAskCallback",
    "OTHER_OPTION",
]
