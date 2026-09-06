"""Tests for ``benchmarks/stats.py`` — the extracted paired statistics.

Closed-form values where they exist, a float-tie fixture built from the
denominators this repo actually compares, and the #390 regression reproduced
from the committed Oxford Pets records. All CPU, no downloads.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
PETS_JSON = BENCHMARKS_DIR / "results_pets_scratch.json"


def _load(name: str) -> ModuleType:
    """Load a module from ``benchmarks/`` with the directory on ``sys.path``.

    The path insert is what lets ``summarize_*.py`` do ``import stats``; without
    it the sibling import fails under pytest even though it works when the
    scripts are run directly.
    """
    if str(BENCHMARKS_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS_DIR))
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def st() -> ModuleType:
    return _load("stats")


# --------------------------------------------------------------------------- #
# Wilcoxon: exact p
# --------------------------------------------------------------------------- #
def test_exact_p_all_positive_n10(st: ModuleType) -> None:
    """W- = 0 at n = 10 with no ties: p = 2 * (1/2**10) = 2/1024."""
    res = st.wilcoxon_signed_rank(np.arange(1.0, 11.0))
    assert res.method == "exact"
    assert res.w_minus == 0.0
    assert res.p_value == pytest.approx(2 / 1024, abs=1e-12)


def test_null_counts_are_a_distribution(st: ModuleType) -> None:
    for n in range(1, 13):
        counts = st._wilcoxon_null_counts(n)
        assert int(counts.sum()) == 2**n
        assert len(counts) == n * (n + 1) // 2 + 1
        assert int(counts[0]) == 1  # only the all-positive assignment gives W- = 0
        # the null is symmetric about max_W / 2
        assert list(counts) == list(counts[::-1])


def test_exact_p_is_two_sided_and_capped(st: ModuleType) -> None:
    # a perfectly balanced split cannot yield p > 1
    assert st.wilcoxon_exact_p(st._wilcoxon_null_counts(6).size // 2, 6) <= 1.0
    assert st.wilcoxon_exact_p(0.0, 5) == pytest.approx(2 / 32, abs=1e-12)


def test_approx_branch_above_n_exact_max(st: ModuleType) -> None:
    d = np.arange(1.0, float(st.N_EXACT_MAX) + 2.0)
    assert st.wilcoxon_p_method(d) == "approx"
    assert st.wilcoxon_signed_rank(d).method == "approx"


# --------------------------------------------------------------------------- #
# Wilcoxon: signed effect size
# --------------------------------------------------------------------------- #
def test_rank_biserial_reaches_both_extremes(st: ModuleType) -> None:
    d = np.arange(1.0, 11.0)
    assert st.wilcoxon_signed_rank(d).rank_biserial_r == pytest.approx(1.0)
    assert st.wilcoxon_signed_rank(-d).rank_biserial_r == pytest.approx(-1.0)


def test_rank_biserial_zero_for_balanced_set(st: ModuleType) -> None:
    d = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    assert st.wilcoxon_signed_rank(d).rank_biserial_r == pytest.approx(0.0, abs=1e-12)


def test_rank_biserial_is_signed_when_median_is_zero(st: ModuleType) -> None:
    """The T20 headline shape: median paired difference exactly 0, r still signed.

    The defective expression ``1 - 2*min(W+,W-)/(n(n+1))`` returned +0.5278 here;
    the matched-pairs value is -0.0556. Guards against a regression to a formula
    that cannot be negative.
    """
    d = np.array([0.0, 0.001469, 0.002745, -0.000826, -0.003110,
                  0.002059, 0.0, 0.001051, -0.003668, -0.001330])
    res = st.wilcoxon_signed_rank(d)
    assert np.median(d) == 0.0
    assert res.n == 8 and res.n_dropped == 2
    assert (res.w_plus, res.w_minus) == (17.0, 19.0)
    assert res.rank_biserial_r == pytest.approx(-2 / 36, abs=1e-12)
    assert res.rank_biserial_r < 0.0
    assert res.p_value == pytest.approx(0.9453125, abs=1e-6)


def test_defective_formula_is_not_what_we_ship(st: ModuleType) -> None:
    """``2*r_printed - 1`` recovers only |r|, which is why we regenerate."""
    d = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    res = st.wilcoxon_signed_rank(d)
    n, w = res.n, res.w_statistic
    r_defective = 1.0 - 2.0 * w / (n * (n + 1))
    assert r_defective == pytest.approx(1.0)          # cannot go negative
    assert res.rank_biserial_r == pytest.approx(-1.0)  # the true value
    assert 2 * r_defective - 1 == pytest.approx(abs(res.rank_biserial_r))


# --------------------------------------------------------------------------- #
# Zeros and ties
# --------------------------------------------------------------------------- #
def test_zeros_are_dropped_and_reported(st: ModuleType) -> None:
    res = st.wilcoxon_signed_rank(np.array([0.0, 1.0, -2.0, 0.0, 3.0]))
    assert (res.n, res.n_dropped) == (3, 2)
    assert "zeros dropped" in res.method_label


def test_all_zero_differences(st: ModuleType) -> None:
    res = st.wilcoxon_signed_rank(np.zeros(5))
    assert res.n == 0 and res.p_value == 1.0 and res.rank_biserial_r == 0.0


def test_float_near_tie_is_detected(st: ModuleType) -> None:
    """Two 'one out of 642' differences are mathematically equal, not bit-equal.

    On this denominator ``(k+1)/642 - k/642`` takes 11 distinct float values
    depending on k, spread 1.11e-16. Exact-equality tie detection misses that,
    claims ``exact`` where the null assumption is violated, and splits a tied
    rank block.
    """
    a = 322 / 642 - 321 / 642
    b = 323 / 642 - 322 / 642
    assert a != b and abs(a - b) < 1e-15  # the fixture is a genuine near-tie

    d = np.array([a, -b, 5 / 642, -9 / 642])
    assert st.wilcoxon_p_method(d) == "approx"
    res = st.wilcoxon_signed_rank(d)
    assert res.w_plus == pytest.approx(4.5)   # the tied block shares rank 1.5
    assert res.w_minus == pytest.approx(5.5)


def test_exact_ties_still_force_approx(st: ModuleType) -> None:
    d = np.array([1.0, -1.0, 2.0])
    assert st.wilcoxon_p_method(d) == "approx"


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_is_the_paired_quantity(st: ModuleType) -> None:
    """Median of paired differences, not difference of medians.

    Constructed so the two estimands disagree: every pair moves by +1 while the
    marginal medians are far apart.
    """
    x = np.array([1.0, 2.0, 3.0, 50.0, 51.0])
    y = x - 1.0
    lo, hi, width = st.bootstrap_median_diff_ci(x - y, n_boot=500, seed=0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert width == pytest.approx(0.0)


def test_bootstrap_is_deterministic_and_ordered(st: ModuleType) -> None:
    d = np.array([0.1, -0.3, 0.2, 0.05, -0.15, 0.4])
    first = st.bootstrap_median_diff_ci(d, n_boot=500, seed=3)
    second = st.bootstrap_median_diff_ci(d, n_boot=500, seed=3)
    assert first == second
    assert first[0] <= first[1]
    assert first[2] == pytest.approx(first[1] - first[0])


def test_bootstrap_empty_input(st: ModuleType) -> None:
    lo, hi, width = st.bootstrap_median_diff_ci(np.array([]))
    assert math.isnan(lo) and math.isnan(hi) and math.isnan(width)


# --------------------------------------------------------------------------- #
# Holm and Wilson
# --------------------------------------------------------------------------- #
def test_holm_known_family(st: ModuleType) -> None:
    assert all(a == pytest.approx(0.030, abs=1e-9) for a in st.holm_bonferroni([0.006] * 5))


def test_holm_is_monotone_and_capped(st: ModuleType) -> None:
    raw = [0.01, 0.02, 0.04, 0.5, 0.9]
    adj = st.holm_bonferroni(raw)
    by_raw = [a for _, a in sorted(zip(raw, adj))]
    assert by_raw == sorted(by_raw)
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj[0] == pytest.approx(0.05)


def test_wilson_known_value(st: ModuleType) -> None:
    lo, hi = st.wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3)
    assert hi == pytest.approx(0.943, abs=1e-3)


def test_wilson_edges(st: ModuleType) -> None:
    assert st.wilson_ci(0, 10)[0] == 0.0
    assert st.wilson_ci(10, 10)[1] == 1.0
    assert all(math.isnan(v) for v in st.wilson_ci(0, 0))


# --------------------------------------------------------------------------- #
# #390 regression, from committed records
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not PETS_JSON.is_file(), reason="pets records not present")
def test_issue_390_reproduction(st: ModuleType) -> None:
    """Reproduce the numbers #390 reports, from the committed Oxford Pets data.

    #390: W+ = 14, W- = 1, printed r = +0.9333, correct r = +0.8667, paired
    median Δ = +1.589 pp against a difference of medians of +2.291 pp.
    """
    runs = json.loads(PETS_JSON.read_text(encoding="utf-8"))["runs"]
    by: dict[str, dict[int, float]] = {}
    for r in runs:
        metric = r.get("held_out_test_metric")
        if metric is None:
            metric = r.get("val_metric")
        if metric is not None:
            by.setdefault(r["condition"], {})[r["seed"]] = float(metric)

    seeds = sorted(set(by["bnnr_xai"]) & set(by["bnnr_random"]))
    x = np.array([by["bnnr_xai"][s] for s in seeds])
    y = np.array([by["bnnr_random"][s] for s in seeds])
    res = st.wilcoxon_signed_rank(x - y)

    assert len(seeds) == 7
    assert (res.n, res.n_dropped) == (5, 2)
    assert (res.w_plus, res.w_minus) == (14.0, 1.0)
    assert res.rank_biserial_r == pytest.approx(0.8667, abs=1e-4)

    r_defective = 1.0 - 2.0 * res.w_statistic / (res.n * (res.n + 1))
    assert r_defective == pytest.approx(0.9333, abs=1e-4)

    assert float(np.median(x - y)) * 100 == pytest.approx(1.589, abs=1e-3)
    assert float(np.median(x) - np.median(y)) * 100 == pytest.approx(2.291, abs=1e-3)


# --------------------------------------------------------------------------- #
# Cross-check against scipy where it happens to be installed
# --------------------------------------------------------------------------- #
def test_agrees_with_scipy_where_available(st: ModuleType) -> None:
    wilcoxon = pytest.importorskip("scipy.stats").wilcoxon
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = int(rng.integers(6, 20))
        d = rng.normal(size=n).round(6)
        res = st.wilcoxon_signed_rank(d)
        ref = wilcoxon(d, alternative="two-sided")
        assert res.w_statistic == pytest.approx(float(ref.statistic))
        assert res.p_value == pytest.approx(float(ref.pvalue), abs=1e-9)
