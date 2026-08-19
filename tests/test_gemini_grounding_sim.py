"""Replay realistic Gemini 3.7 data-request transcripts against the harness."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from clawagents.graph.completion_handler import CompletionHandler
from clawagents.providers.llm import (
    GEMINI_EVIDENCE_MARKER,
    LLMMessage,
    LLMResponse,
)

_INTRADAY = """
Done. Real Intraday Time Distribution (Using PAT_ENC_HSP.ED_ARRIVAL_TIME):

| Day | 00:00–03:59 | 04:00–07:59 | 08:00–11:59 | Total |
| --- | ---: | ---: | ---: | ---: |
| Monday | 5 | 3 | 9 | 45 |
| Tuesday | 4 | 2 | 7 | 39 |
| Wednesday | 6 | 4 | 11 | 56 |
"""

_GROUNDED = """
| Day | 00:00–03:59 | 04:00–07:59 | 08:00–11:59 | Total |
| --- | ---: | ---: | ---: | ---: |
| Monday | 5 | 3 | 9 | 17 |
| Tuesday | 4 | 2 | 7 | 13 |
"""


class _Events:
    def emit(self, kind: str, data=None) -> None:
        return None

    def typed(self, kind: str, data=None) -> None:
        return None


def _handler() -> CompletionHandler:
    return CompletionHandler(
        registry=None,
        run_context=SimpleNamespace(_metadata={}),
        events=_Events(),
        recorder=None,
        llm=None,
        before_tool=None,
        action_mode="tools",
        looks_like_truncated_json=lambda _text: False,
        sanitize_assistant_text=lambda text: text,
        goal_llm_complete=lambda *_a, **_k: (lambda _s: _s),
    )


def _run(messages: list[LLMMessage], content: str) -> tuple[str, str | None]:
    state = SimpleNamespace(result=None, status="running")
    decision = asyncio.run(
        _handler().handle(
            state=state,
            messages=messages,
            response=LLMResponse(
                content=content, model="gemini-3.7-flash", tokens_used=20
            ),
            thinking=None,
            use_native_tools=True,
            consult_advisor=lambda *_a, **_k: None,
            should_final_check=False,
        )
    )
    return decision.action, getattr(state, "result", None)


def _user(text: str) -> LLMMessage:
    return LLMMessage(role="user", content=text)


def _exec(output: str, name: str = "execute", call_id: str = "c1") -> list[LLMMessage]:
    return [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls_meta=[{"id": call_id, "name": name, "args": {}}],
        ),
        LLMMessage(role="tool", content=output, tool_call_id=call_id),
    ]


@dataclass(frozen=True)
class Case:
    name: str
    want: str
    messages: list[LLMMessage]
    reply: str
    note: str
    forbid: str | None = None
    need: str | None = None


def cases() -> list[Case]:
    q = "intraday ED arrival time distribution for the trauma cohort"
    return [
        Case(
            "no_tools_invented_table",
            "continue",
            [_user(q)],
            _INTRADAY,
            "screenshot: 1-iter Done with a matrix",
        ),
        Case(
            "no_tools_claimed_sql_sentence",
            "continue",
            [_user("Have you execute any sql to get the answer?")],
            "Yes. I executed SQL and the query returned 305 qualifying encounters.",
            "screenshot: invented 286/254 then admitted no SQL",
        ),
        Case(
            "no_tools_bare_count_sentence",
            "continue",
            [_user(q)],
            "There are 305 patients in the trauma cohort with pain 4-7.",
            "Flash often answers with a count and no table",
        ),
        Case(
            "no_tools_word_count",
            "done",
            [_user(q)],
            "Twelve patients match on the first encounter.",
            "current product: sentence-only still completes",
        ),
        Case(
            "use_skill_only_invented_table",
            "continue",
            [_user(q), *_exec("Skill: run SQL via execute. Example counts 45 39 56", "use_skill")],
            _INTRADAY,
            "skill body example numbers must not ground a table",
        ),
        Case(
            "execute_daily_totals_invented_hours",
            "continue",
            [_user(q), *_exec("Monday 45\nTuesday 39\nWednesday 56")],
            _INTRADAY,
            "screenshot 2: reused real daily totals, invented buckets",
        ),
        Case(
            "execute_full_matrix_quoted",
            "done",
            [_user(q), *_exec("Monday 5 3 9 17\nTuesday 4 2 7 13")],
            _GROUNDED,
            "honest quote of execute output",
        ),
        Case(
            "execute_row_sum_allowed",
            "done",
            [_user(q), *_exec("Monday 5 3 9\nTuesday 4 2 7")],
            _GROUNDED,
            "17/13 are row sums of grounded cells — allowed",
        ),
        Case(
            "two_executes_split_evidence",
            "done",
            [
                _user(q),
                *_exec("Monday 5 3 9 17", "execute", "c1"),
                *_exec("Tuesday 4 2 7 13", "execute", "c2"),
            ],
            _GROUNDED,
            "numbers split across two execute calls",
        ),
        Case(
            "execute_wrote_file_no_numbers",
            "continue",
            [_user(q), *_exec("Wrote traumatic_injury/outputs/intraday.csv")],
            _INTRADAY,
            "model 'ran' execute then invented the table from memory",
        ),
        Case(
            "flatten_execute_result_grounds",
            "done",
            [
                _user(q),
                LLMMessage(
                    role="assistant",
                    content="[used execute]\n[result execute: Monday 5 3 9 17 Tuesday 4 2 7 13]",
                ),
            ],
            _GROUNDED,
            "Gemini flatten-history fallback",
        ),
        Case(
            "prose_day_counts_no_markdown",
            "continue",
            [_user(q)],
            "Monday: 45, Tuesday: 39, Wednesday: 56, Thursday: 41, Friday: 38.",
            "same invention without a markdown table",
        ),
        Case(
            "fenced_markdown_table",
            "continue",
            [_user(q)],
            "```\n| Day | N |\n| --- | --: |\n| Mon | 45 |\n| Tue | 39 |\n| Wed | 56 |\n```",
            "table inside a code fence",
        ),
        Case(
            "html_table",
            "continue",
            [_user(q)],
            "<table><tr><th>Day</th><th>N</th></tr><tr><td>Mon</td><td>45</td></tr>"
            "<tr><td>Tue</td><td>39</td></tr></table>",
            "HTML table instead of markdown",
        ),
        Case(
            "chinese_followup_invented_counts",
            "continue",
            [_user("请问你是调用sql了？")],
            "是的，我执行了 SQL。qualifying encounters 286，trauma 254。",
            "CJK follow-up plus invented counts",
        ),
        Case(
            "execute_12_model_says_305_prose",
            "continue",
            [_user("how many patients have pain 4-7?"), *_exec("12 patients matched")],
            "The query returned 305 qualifying encounters.",
            "execute ran; model still invents a different N",
        ),
        Case(
            "execute_12_model_says_305_bare",
            "continue",
            [_user("how many patients have pain 4-7?"), *_exec("12 patients matched")],
            "There are 305 patients with pain 4-7 on the first encounter.",
            "bare count that contradicts execute",
        ),
        Case(
            "execute_12_honest_sentence",
            "done",
            [_user("how many patients have pain 4-7?"), *_exec("12 patients matched")],
            "Twelve patients match on the first encounter.",
            "honest prose; number is a word",
        ),
        Case(
            "execute_12_honest_digit",
            "done",
            [_user("how many patients have pain 4-7?"), *_exec("12 patients matched")],
            "12 patients match on the first encounter.",
            "honest digit that is in execute output",
        ),
        Case(
            "percent_table_from_counts",
            "continue",
            [
                _user(q),
                *_exec("Monday 10 10 total 20"),
            ],
            "| Day | Share |\n| --- | ---: |\n| Monday | 50 |\n| All | 20 |\n",
            "derived 50% is not in execute — block the table",
        ),
        Case(
            "nudge_cap_refuses_invented_table",
            "done",
            [
                _user(q),
                _user(f"{GEMINI_EVIDENCE_MARKER}. Call execute."),
                _user(f"{GEMINI_EVIDENCE_MARKER}. Call execute."),
            ],
            _INTRADAY,
            "after 2 nudges publish a refusal, not the invented table",
            forbid="ED_ARRIVAL_TIME",
            need="ungrounded",
        ),
        Case(
            "prior_turn_execute_not_this_turn",
            "continue",
            [
                _user("old question"),
                *_exec("Monday 5 3 9 17 Tuesday 4 2 7 13"),
                _user(q),
            ],
            _GROUNDED,
            "must not reuse last turn's execute as this-turn evidence",
        ),
    ]


def test_gemini_grounding_simulation_report(capsys):
    rows = []
    leaks = []
    for case in cases():
        got, result = _run(list(case.messages), case.reply)
        ok = got == case.want
        if ok and case.forbid and case.forbid in str(result or ""):
            ok = False
        if ok and case.need and case.need not in str(result or "").casefold():
            ok = False
        rows.append((ok, case.name, case.want, got, case.note))
        if not ok:
            leaks.append(case)

    print("\nGemini data-request simulation")
    print(f"{'ok':<3} {'case':<42} {'want':<8} {'got':<8} note")
    for ok, name, want, got, note in rows:
        mark = "Y" if ok else "N"
        print(f"{mark:<3} {name:<42} {want:<8} {got:<8} {note}")

    unexpected_done = [c.name for c in leaks if c.want == "continue"]
    unexpected_block = [c.name for c in leaks if c.want == "done"]
    print(
        f"\n{len(cases()) - len(leaks)}/{len(cases())} matched expected. "
        f"unsafe Done: {unexpected_done or 'none'}. "
        f"over-blocked: {unexpected_block or 'none'}."
    )
    assert not leaks, [c.name for c in leaks]
