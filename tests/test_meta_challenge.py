"""Each frozen grader must reject its starter and accept a working reference."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from benchmark_meta_challenge import CASES, clean_completion, grade, prepare


@pytest.mark.parametrize("task", list(CASES))
def test_challenge_negative_and_positive_controls(tmp_path, task):
    case = CASES[task]
    prepare(case, tmp_path)
    assert not grade(case, tmp_path)["passed"]
    prepare(case, tmp_path, reference=True)
    result = grade(case, tmp_path)
    assert result["passed"], result
    assert result["checks"] >= 10


def test_budget_exhaustion_is_not_a_clean_pass():
    assert clean_completion("done", "Finished")
    assert not clean_completion("done", "Reached maximum of 32 tool rounds.")
    assert not clean_completion("error", "Finished")
    assert not clean_completion("done", "[cancelled]")


@pytest.mark.asyncio
async def test_shell_can_read_fixture_but_not_private_grader(tmp_path):
    import shutil
    import shlex

    if not shutil.which("sandbox-exec"):
        pytest.skip("macOS sandbox-exec required")
    from benchmark_meta_challenge import ChallengeSandbox

    (tmp_path / "fixture.txt").write_text("public fixture")
    backend = ChallengeSandbox(tmp_path)
    ok = await backend.exec("cat fixture.txt", cwd=str(tmp_path))
    assert ok.exit_code == 0 and "public fixture" in ok.stdout
    private = Path(__file__).parents[1] / "scripts" / "meta_challenge_cases.py"
    denied = await backend.exec("cat " + shlex.quote(str(private)), cwd=str(tmp_path))
    assert denied.exit_code != 0
    assert "BENCHMARK_PRIVATE_ORACLE" not in denied.stdout
    write = await backend.exec("printf test > local.txt", cwd=str(tmp_path))
    assert write.exit_code == 0
    assert (tmp_path / "local.txt").read_text() == "test"


def test_integrity_detector_has_positive_control():
    from benchmark_meta_challenge import grader_leaked
    from clawagents.providers.llm import LLMMessage

    assert grader_leaked([LLMMessage(role="tool", content="BENCHMARK_PRIVATE_ORACLE")])
    assert not grader_leaked([LLMMessage(role="tool", content="ordinary test output")])
