from __future__ import annotations

import json
from pathlib import Path


RESULTS_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "results.json"
REQUIRED_CONDITIONS = {"no_bnnr", "randaugment", "bnnr_branch_search"}


def test_benchmark_results_cover_all_conditions() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    runs = data.get("runs") or []

    assert runs, "benchmarks/results.json should include at least one run"

    seen_conditions = {run["condition"] for run in runs}
    assert REQUIRED_CONDITIONS.issubset(seen_conditions)
