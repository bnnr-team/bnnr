"""Regression guards for published benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "results.json"
RESNET50_RESULTS_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "results_resnet50.json"
REQUIRED_CONDITIONS = frozenset({"no_bnnr", "randaugment", "bnnr_branch_search"})


@pytest.fixture(name="benchmark_results")
def fixture_benchmark_results() -> dict:
    # results.json and results_resnet50.json are generated files (gitignored).
    # Skip rather than fail when they're absent — they only exist after a local run.
    for path in (RESULTS_PATH, RESNET50_RESULTS_PATH):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    pytest.skip("no local results.json or results_resnet50.json — run benchmarks first")
    raise AssertionError("unreachable: pytest.skip always raises")


def test_results_json_has_runs_for_all_conditions(benchmark_results: dict) -> None:
    runs = benchmark_results.get("runs")
    assert isinstance(runs, list) and runs, "runs must be a non-empty list"

    for index, run in enumerate(runs):
        assert isinstance(run, dict), f"run at index {index} must be an object"
        assert "condition" in run and run["condition"], (
            f"run at index {index} is missing required 'condition'"
        )
        assert "val_metric" in run and run["val_metric"] is not None, (
            f"run at index {index} is missing required 'val_metric'"
        )

    conditions = {run["condition"] for run in runs}
    missing = REQUIRED_CONDITIONS - conditions
    assert not missing, f"missing conditions in results.json: {sorted(missing)}"
