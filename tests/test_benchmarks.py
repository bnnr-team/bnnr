"""Regression guards for published benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "results.json"
REQUIRED_CONDITIONS = frozenset({"no_bnnr", "randaugment", "bnnr_branch_search"})


@pytest.fixture(name="benchmark_results")
def fixture_benchmark_results() -> dict:
    assert RESULTS_PATH.is_file(), f"missing {RESULTS_PATH}"
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


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
