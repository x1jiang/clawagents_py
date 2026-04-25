"""Tests for AskUserQuestion structured HITL tool."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from clawagents import ask_user_question_tool
from clawagents.tools.ask_user_question import OTHER_OPTION


def _good_question(**overrides):
    base = {
        "question": "Which framework?",
        "header": "Framework",
        "options": ["FastAPI", "Flask"],
    }
    base.update(overrides)
    return base


# ─── Validation paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_zero_questions_rejected():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute({"questions": []})
    assert res.success is False
    assert "1-3 questions" in (res.error or "")


@pytest.mark.asyncio
async def test_four_questions_rejected():
    tool = ask_user_question_tool(on_ask=None)
    questions = [_good_question(header=f"H{i}") for i in range(4)]
    res = await tool.execute({"questions": questions})
    assert res.success is False
    assert "1-3 questions" in (res.error or "")


@pytest.mark.asyncio
async def test_one_option_rejected():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute({"questions": [_good_question(options=["only"])]})
    assert res.success is False
    assert "2-4" in (res.error or "")


@pytest.mark.asyncio
async def test_five_options_rejected():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute(
        {"questions": [_good_question(options=["a", "b", "c", "d", "e"])]}
    )
    assert res.success is False
    assert "2-4" in (res.error or "")


@pytest.mark.asyncio
async def test_duplicate_headers_rejected():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute(
        {
            "questions": [
                _good_question(header="Same"),
                _good_question(header="Same", question="Different question?"),
            ]
        }
    )
    assert res.success is False
    assert "duplicate header" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_duplicate_options_rejected():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute(
        {"questions": [_good_question(options=["A", "A"])]}
    )
    assert res.success is False
    assert "unique" in (res.error or "")


@pytest.mark.asyncio
async def test_question_too_long_rejected():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute(
        {"questions": [_good_question(question="x" * 257)]}
    )
    assert res.success is False
    assert "256" in (res.error or "")


@pytest.mark.asyncio
async def test_header_too_long_rejected():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute(
        {"questions": [_good_question(header="h" * 81)]}
    )
    assert res.success is False
    assert "80" in (res.error or "")


@pytest.mark.asyncio
async def test_no_callback_returns_error():
    tool = ask_user_question_tool(on_ask=None)
    res = await tool.execute({"questions": [_good_question()]})
    assert res.success is False
    assert "no UI registered" in (res.error or "")


# ─── Happy path ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_successful_call_with_fake_callback():
    seen: Dict[str, List[Dict[str, Any]]] = {"questions": []}

    async def on_ask(questions):
        seen["questions"] = list(questions)
        return {
            "Framework": {"question": "Which framework?", "answer": "FastAPI"},
            "DB": {
                "question": "Which DB?",
                "answer": OTHER_OPTION,
                "free_text": "DuckDB",
            },
        }

    tool = ask_user_question_tool(on_ask=on_ask)
    res = await tool.execute(
        {
            "questions": [
                _good_question(),
                _good_question(
                    header="DB",
                    question="Which DB?",
                    options=["Postgres", "SQLite"],
                ),
            ]
        }
    )
    assert res.success is True, res.error
    parsed = json.loads(res.output)
    assert parsed["Framework"] == {
        "question": "Which framework?",
        "answer": "FastAPI",
    }
    assert parsed["DB"]["answer"] == OTHER_OPTION
    assert parsed["DB"]["free_text"] == "DuckDB"
    # callback received the prepared questions
    assert len(seen["questions"]) == 2


# ─── "Other" injection ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_other_option_appended_implicitly():
    captured: Dict[str, Any] = {}

    async def on_ask(questions):
        captured["q"] = questions
        return {q["header"]: {"question": q["question"], "answer": q["options"][0]}
                for q in questions}

    tool = ask_user_question_tool(on_ask=on_ask)
    res = await tool.execute({"questions": [_good_question()]})
    assert res.success is True

    sent = captured["q"]
    assert sent[0]["options"][-1] == OTHER_OPTION
    # original 2 + injected 1 = 3
    assert len(sent[0]["options"]) == 3


@pytest.mark.asyncio
async def test_other_not_double_injected():
    captured: Dict[str, Any] = {}

    async def on_ask(questions):
        captured["q"] = questions
        return {q["header"]: {"question": q["question"], "answer": q["options"][0]}
                for q in questions}

    tool = ask_user_question_tool(on_ask=on_ask)
    res = await tool.execute(
        {
            "questions": [
                _good_question(options=["A", "B", OTHER_OPTION]),
            ]
        }
    )
    assert res.success is True
    # No duplicate "Other" — already present, count stays at 3.
    assert captured["q"][0]["options"].count(OTHER_OPTION) == 1


# ─── Callback failure surfaces as tool error ───────────────────────────────


@pytest.mark.asyncio
async def test_callback_exception_surfaces_as_error():
    async def boom(_questions):
        raise RuntimeError("ui crashed")

    tool = ask_user_question_tool(on_ask=boom)
    res = await tool.execute({"questions": [_good_question()]})
    assert res.success is False
    assert "ui crashed" in (res.error or "")
