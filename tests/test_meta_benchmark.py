"""Negative controls for the benchmark's independent outcome validators."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "meta_benchmark", Path(__file__).parents[1] / "scripts" / "benchmark_meta_glimmer.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_read_validator_rejects_wrong_retry_count(tmp_path):
    (tmp_path / "config.json").write_text(BENCH.FIXTURES["config.json"])
    assert BENCH.check("read", tmp_path, "Version 3.17.9; retries 7")
    assert not BENCH.check("read", tmp_path, "Version 3.17.9; retries 27")


def test_aggregate_validator_rejects_self_report_and_wrong_artifact(tmp_path):
    assert not BENCH.check("aggregate", tmp_path, "Done, correct!")
    (tmp_path / "totals.json").write_text('{"Ada":22,"Lin":809}')
    assert not BENCH.check("aggregate", tmp_path, "Done")
    (tmp_path / "totals.json").write_text('{"Ada":22,"Lin":9}')
    assert BENCH.check("aggregate", tmp_path, "Done")


def test_repair_validator_rejects_original_bug(tmp_path):
    (tmp_path / "ranges.py").write_text(BENCH.FIXTURES["ranges.py"])
    assert not BENCH.check("repair", tmp_path, "All tests pass")
    (tmp_path / "ranges.py").write_text('def inclusive_sum(start, end):\n    return sum(range(start, end+1))\n')
    assert BENCH.check("repair", tmp_path, "Done")
