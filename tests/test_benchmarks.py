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

    conditions = {run.get("condition") for run in runs if isinstance(run, dict)}
    missing = REQUIRED_CONDITIONS - conditions
    assert not missing, f"missing conditions in results.json: {sorted(missing)}"
