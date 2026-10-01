"""Integrity checks for the standalone, blinded report benchmark."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from clawagents.config.features import temporary_overrides
from clawagents.providers.llm import LLMProvider, LLMResponse, NativeToolCall

SPEC = importlib.util.spec_from_file_location(
    "hip_benchmark", Path(__file__).parents[1] / "examples/hip_dislocation/agent.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def prediction(label="obturator", review=False):
    return {
        "label": label,
        "needs_review": review,
        "obturator_screen": "possible",
        "evidence": "obturator dislocation",
        "reason": "Explicit subtype.",
    }


def test_submit_requires_read_and_literal_evidence():
    async def scenario():
        reader = module.ReadReport("Right obturator\n dislocation.")
        submit = module.SubmitLabel(reader)
        assert not (await submit.execute(prediction())).success
        await reader.execute({})
        invalid = {**prediction(), "evidence": "posterior dislocation"}
        assert not (await submit.execute(invalid)).success
        accepted = await submit.execute(prediction())
        assert accepted.success and accepted.return_direct

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        {"label": "unknown"},
        {"needs_review": "false"},
        {"label": "other_unknown", "needs_review": False},
        {"obturator_screen": "confirmed"},
        {"extra": "gold label"},
    ],
)
def test_invalid_submission_is_not_scored(change):
    async def scenario():
        reader = module.ReadReport("Right obturator dislocation.")
        await reader.execute({})
        submit = module.SubmitLabel(reader)
        assert not (await submit.execute({**prediction(), **change})).success
        assert submit.prediction is None

    asyncio.run(scenario())


def test_scoring_keeps_review_and_failures_in_denominator():
    rows = [
        {"reference": "obturator", "prediction": prediction()},
        {"reference": "obturator", "prediction": prediction("other_unknown", True)},
        {"reference": "posterior", "prediction": None},
    ]
    s = module.score(rows)
    assert s["n"] == 3 and s["correct"] == 1 and s["failures"] == 1
    assert s["coverage"] == 1 / 3 and s["agreement_all"] == 1 / 3
    assert s["selective_agreement"] == 1
    assert s["per_class"]["obturator"]["recall"] == 0.5
    assert s["obturator_screen_capture"] == 1


def test_record_aggregation_resolves_weak_but_abstains_on_conflict():
    def row(rid, ref, p, source_row):
        return {
            "record_id": rid,
            "reference": ref,
            "prediction": p,
            "source_row": source_row,
        }

    rows = [
        row(1, "obturator", prediction("other_unknown", True), 2),
        row(1, "obturator", prediction(), 3),
        row(2, "obturator", prediction(), 4),
        row(2, "obturator", prediction("posterior"), 5),
    ]
    records = module.aggregate_records(rows)
    assert records[0]["prediction"]["label"] == "obturator"
    assert not records[0]["prediction"]["needs_review"]
    assert records[1]["prediction"]["label"] == "other_unknown"
    assert records[1]["conflicting_supported_labels"]


def test_real_claw_loop_reads_then_submits_without_extra_tools(monkeypatch, tmp_path):
    observed = []

    class FakeProvider(LLMProvider):
        name = "openai"
        model = module.MODEL

        def __init__(self, config):
            async def close():
                pass

            assert config.openai_wire_api == "responses"
            self.client = SimpleNamespace(
                base_url="https://api.openai.com/v1", close=close
            )
            self.calls = 0

        async def chat(
            self, messages, on_chunk=None, cancel_event=None, tools=None, **kwargs
        ):
            observed.extend(messages)
            assert {t.name for t in tools} == {"read_report", "submit_label"}
            assert all("PRIVATE_GOLD_LABEL" not in str(m.content) for m in messages)
            self.calls += 1
            call = (
                NativeToolCall("read_report", {}, "read1")
                if self.calls == 1
                else NativeToolCall("submit_label", prediction(), "submit1")
            )
            if self.calls == 2:
                assert any(
                    m.role == "tool"
                    and "Right obturator dislocation." in str(m.content)
                    for m in messages
                )
            return LLMResponse(
                "", module.MODEL, 20, tool_calls=[call], prompt_tokens=10
            )

    monkeypatch.setattr(module, "OpenAIProvider", FakeProvider)
    with temporary_overrides(module.FEATURES):
        result = asyncio.run(
            module.annotate("Right obturator dislocation.", "not-a-real-key", tmp_path)
        )
    assert result["status"] == "done"
    assert result["prediction"]["label"] == "obturator"
    assert result["tool_calls"] == 2
    assert result["usage"]["requests"] == 2
    assert not any("PRIVATE_GOLD_LABEL" in str(m.content) for m in observed)
