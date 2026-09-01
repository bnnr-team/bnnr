"""Acceptance oracle for ``benchmarks/stats.py`` (#398, source #390).

This file exists to be *hard to satisfy*. The rule it follows: every critical
quantity is checked against a **different computation path** than the one under
test, or against a property that no single golden value can encode. A test that
passes against the code it is meant to reject is theatre, and two of the tests
in ``test_benchmarks_stats.py`` were exactly that — see ``TIER 3`` below.

Layout:

* **Tier 1** — independent recomputation. Brute-force enumeration (the module
  uses subset-sum DP, so the test must not), scipy on the branches that actually
  carry risk, and a hand-rolled replay of the bootstrap.
* **Tier 2** — metamorphic properties. No golden values, so they cannot be
  back-fitted to whatever the implementation happens to return.
* **Tier 3** — the two #390 defects, with fixtures that can genuinely fail.
* **Tier 4** — tolerance brackets on ``_tie_groups``, on both sides.

scipy is imported unconditionally on purpose. It is guaranteed by the hard
``scikit-learn`` dependency, and it is the only external engine that can falsify
the estimators here — if it ever stops being installed, this file must fail
loudly rather than skip and report green.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.stats import wilcoxon as scipy_wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
PETS_JSON = BENCHMARKS_DIR / "results_pets_scratch.json"
WATERBIRDS_B15 = BENCHMARKS_DIR / "findings_t20" / "results_waterbirds_b15.json"


def _load(name: str) -> ModuleType:
    """Load a module from ``benchmarks/`` with the directory on ``sys.path``."""
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
# TIER 1 — independent recomputation
# --------------------------------------------------------------------------- #
def _brute_force_two_sided(n: int) -> list[int]:
    """All ``2**n`` values of ``min(W+, W-)``, by explicit enumeration.

    Deliberately naive: ``stats.py`` computes the null with a subset-sum DP, so a
    test that reuses the DP would only prove the DP is self-consistent.
    """
    ranks = range(1, n + 1)
    max_w = n * (n + 1) // 2
    out = []
    for signs in itertools.product((0, 1), repeat=n):
        w_minus = sum(r for r, s in zip(ranks, signs) if s)
        out.append(min(w_minus, max_w - w_minus))
    return out


@pytest.mark.parametrize("n", range(1, 13))
def test_exact_p_matches_brute_force(st: ModuleType, n: int) -> None:
    """Exact two-sided p equals the enumerated tail, for every reachable w_min."""
    dist = _brute_force_two_sided(n)
    total = float(2**n)
    for w in range(0, n * (n + 1) // 2 // 2 + 1):
        expected = sum(1 for v in dist if v <= w) / total
        assert st.wilcoxon_exact_p(w, n) == pytest.approx(expected, abs=1e-12), (
            f"n={n} w_min={w}"
        )


def test_approx_matches_scipy_under_ties_and_zeros(st: ModuleType) -> None:
    """The approximation branch, on inputs that actually carry ties and zeros.

    ``test_agrees_with_scipy_where_available`` in ``test_benchmarks_stats.py``
    draws ``rng.normal(...).round(6)``, which puts 50 of its 50 cases on the
    exact branch with no dropped zeros — it validates only the branch that was
    never in doubt. Small integers guarantee both.
    """
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(400):
        n = int(rng.integers(6, 40))
        d = rng.integers(-3, 4, size=n).astype(float)
        if not np.any(d):
            continue
        res = st.wilcoxon_signed_rank(d)
        if res.method != "approx":
            continue
        ref = scipy_wilcoxon(
            d, alternative="two-sided", method="approx",
            zero_method="wilcox", correction=True,
        )
        assert res.p_value == pytest.approx(float(ref.pvalue), abs=1e-12)
        checked += 1
    assert checked > 100, f"fixture stopped exercising the approx branch ({checked} cases)"


def test_exact_matches_scipy_conditional_on_nonzeros(st: ModuleType) -> None:
    """Dropping zeros keeps exactness, checked against scipy rather than asserted.

    This is the only external check of the convention the module rests on: the
    signed-rank null is exact *conditional on* the number of non-zero
    differences. scipy's own ``method='auto'`` refuses its exact branch whenever
    zeros are present, so the comparison has to be made against scipy fed the
    already-dropped non-zeros.
    """
    rng = np.random.default_rng(19)
    checked = 0
    for _ in range(200):
        n = int(rng.integers(4, 20))
        d = rng.normal(size=n).round(6)
        d[rng.integers(0, n, size=int(rng.integers(1, 4)))] = 0.0
        nz = d[d != 0.0]
        if len(nz) < 2 or len(np.unique(np.abs(nz))) < len(nz):
            continue
        res = st.wilcoxon_signed_rank(d)
        assert res.method == "exact"
        ref = scipy_wilcoxon(nz, alternative="two-sided", method="exact")
        assert res.p_value == pytest.approx(float(ref.pvalue), abs=1e-12)
        assert res.n == len(nz)
        checked += 1
    assert checked > 50, f"fixture stopped producing zero-dropping cases ({checked})"


def test_bootstrap_matches_independent_replay(st: ModuleType) -> None:
    """Percentile bootstrap on the median, replayed with its own RNG stream."""
    d = np.array([0.1, -0.3, 0.2, 0.05, -0.15, 0.4])
    lo, hi, width = st.bootstrap_median_diff_ci(d, n_boot=5000, seed=5)

    rng = np.random.default_rng(5)
    meds = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(5000)])
    ref_lo, ref_hi = np.percentile(meds, [2.5, 97.5])

    assert lo == pytest.approx(float(ref_lo), abs=1e-12)
    assert hi == pytest.approx(float(ref_hi), abs=1e-12)
    assert width == pytest.approx(hi - lo, abs=1e-12)


# --------------------------------------------------------------------------- #
# TIER 2 — metamorphic properties
# --------------------------------------------------------------------------- #
def _random_diffs(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(2, 30))
    d = rng.normal(size=n).round(int(rng.integers(1, 7)))
    return d


def test_negation_flips_r_and_preserves_p(st: ModuleType) -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        d = _random_diffs(rng)
        a, b = st.wilcoxon_signed_rank(d), st.wilcoxon_signed_rank(-d)
        assert b.rank_biserial_r == pytest.approx(-a.rank_biserial_r, abs=1e-12)
        assert b.p_value == pytest.approx(a.p_value, abs=1e-12)
        assert b.w_statistic == pytest.approx(a.w_statistic, abs=1e-12)
        assert (b.n, b.n_dropped, b.method) == (a.n, a.n_dropped, a.method)


def test_pair_order_is_irrelevant(st: ModuleType) -> None:
    rng = np.random.default_rng(4)
    for _ in range(200):
        d = _random_diffs(rng)
        a = st.wilcoxon_signed_rank(d)
        b = st.wilcoxon_signed_rank(rng.permutation(d))
        assert b.p_value == pytest.approx(a.p_value, abs=1e-12)
        assert b.rank_biserial_r == pytest.approx(a.rank_biserial_r, abs=1e-12)


def test_positive_rescaling_changes_nothing(st: ModuleType) -> None:
    """``d -> c*d`` must not move p or r — catches an absolute tie tolerance."""
    rng = np.random.default_rng(5)
    for _ in range(100):
        d = _random_diffs(rng)
        base = st.wilcoxon_signed_rank(d)
        for c in (1e-6, 1e-3, 1e3, 1e6):
            scaled = st.wilcoxon_signed_rank(d * c)
            assert scaled.p_value == pytest.approx(base.p_value, abs=1e-12), f"c={c}"
            assert scaled.rank_biserial_r == pytest.approx(base.rank_biserial_r, abs=1e-12)


def test_rank_sums_and_r_range(st: ModuleType) -> None:
    rng = np.random.default_rng(6)
    for _ in range(300):
        d = _random_diffs(rng)
        res = st.wilcoxon_signed_rank(d)
        assert res.w_plus + res.w_minus == pytest.approx(res.n * (res.n + 1) / 2, abs=1e-9)
        assert -1.0 - 1e-12 <= res.rank_biserial_r <= 1.0 + 1e-12
        nz = d[d != 0.0]
        if len(nz) and np.all(nz > 0):
            assert res.rank_biserial_r == pytest.approx(1.0)
        elif len(nz) and np.all(nz < 0):
            assert res.rank_biserial_r == pytest.approx(-1.0)
        else:
            assert abs(res.rank_biserial_r) < 1.0 or res.n == 0


def test_r_sign_follows_the_rank_sums_not_the_median(st: ModuleType) -> None:
    """r is signed by ``W+ - W-``, which is not always the sign of the median.

    Guards a docstring that claimed the two always agree when the median is
    non-zero. They do not: ranks carry magnitude, the median only counts. Here
    the median is +1 while the two large negative differences own ranks 4 and 5.
    """
    d = np.array([1.0, 2.0, 3.0, -100.0, -101.0])
    res = st.wilcoxon_signed_rank(d)
    assert np.median(d) > 0.0
    assert (res.w_plus, res.w_minus) == (6.0, 9.0)
    assert res.rank_biserial_r == pytest.approx(-0.2)
    assert np.sign(res.rank_biserial_r) != np.sign(np.median(d))


def test_holm_properties(st: ModuleType) -> None:
    rng = np.random.default_rng(8)
    for _ in range(200):
        raw = list(rng.uniform(0, 1, size=int(rng.integers(1, 12))))
        adj = st.holm_bonferroni(raw)
        assert len(adj) == len(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adj, raw)), "adjusted must not shrink a p"
        assert all(0.0 <= a <= 1.0 for a in adj)
        by_raw = [a for _, a in sorted(zip(raw, adj))]
        assert by_raw == sorted(by_raw), "must be monotone in the raw order"
    assert st.holm_bonferroni([0.031]) == pytest.approx([0.031])
    assert st.holm_bonferroni([]) == []


def test_wilson_properties(st: ModuleType) -> None:
    """Contains the point estimate, and is symmetric under ``k -> n - k``.

    The containment check carries a 1-ulp slack: at ``k == n`` the closed form
    reaches 1 only in exact arithmetic (``wilson_ci(642, 642)[1]`` lands on
    0.9999999999999999), which the ``min(1.0, ...)`` clip cannot lift.
    """
    for n in (1, 7, 10, 214, 642, 3929):
        for k in {0, 1, n // 3, n // 2, n - 1, n}:
            lo, hi = st.wilson_ci(k, n)
            assert 0.0 <= lo <= k / n + 1e-12, f"k={k} n={n}"
            assert k / n - 1e-12 <= hi <= 1.0, f"k={k} n={n}"
            mlo, mhi = st.wilson_ci(n - k, n)
            assert mlo == pytest.approx(1.0 - hi, abs=1e-12)
            assert mhi == pytest.approx(1.0 - lo, abs=1e-12)


# --------------------------------------------------------------------------- #
# TIER 3 — the two #390 defects, with fixtures that can fail
# --------------------------------------------------------------------------- #
def test_bootstrap_is_the_median_of_paired_differences(st: ModuleType) -> None:
    """A fixture on which the two estimands have **opposite signs**.

    The fixture in ``test_benchmarks_stats.py`` (``x=[1,2,3,50,51]``, ``y=x-1``)
    gives ``median(x-y) == median(x)-median(y) == 1.0``, so it passes verbatim
    against the difference-of-medians estimator #390 reported. This one does not.
    """
    # Six pairs. The point estimates disagree in sign (-2.5 against +0.5) *and*
    # the resample distributions separate, so the two estimands differ at the
    # interval level too -- which the old fixture did not achieve.
    x = np.array([13.0, 0.0, 14.0, 16.0, 3.0, 13.0])
    y = np.array([7.0, 17.0, 18.0, 11.0, 11.0, 14.0])
    assert float(np.median(x - y)) == pytest.approx(-2.5)
    assert float(np.median(x) - np.median(y)) == pytest.approx(+0.5)

    n_boot, seed, n = 2000, 0, len(x)

    def _replay(statistic) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        vals = np.array([statistic(rng.integers(0, n, n)) for _ in range(n_boot)])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    paired_ci = _replay(lambda i: np.median((x - y)[i]))
    unpaired_ci = _replay(lambda i: np.median(x[i]) - np.median(y[i]))
    assert paired_ci != unpaired_ci, "fixture no longer discriminates the two estimands"

    lo, hi, _ = st.bootstrap_median_diff_ci(x - y, n_boot=n_boot, seed=seed)
    assert (lo, hi) == pytest.approx(paired_ci, abs=1e-12)
    assert (lo, hi) != pytest.approx(unpaired_ci, abs=1e-12)


def test_bootstrap_reproduces_the_390_imagewoof_correction(st: ModuleType) -> None:
    """The published correction, on the records it was computed from.

    #390: the shipped interval was a bootstrap of the difference of medians and
    came out roughly twice as wide as the paired one. Both are recomputed here
    from the raw records, so the test fails if either estimand drifts.
    """
    records = BENCHMARKS_DIR / "results_imagewoof_scratch.json"
    assert records.is_file(), f"missing regression fixture: {records}"
    runs = json.loads(records.read_text(encoding="utf-8"))["runs"]
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
    n, n_boot, seed = len(x), 10_000, 42

    rng = np.random.default_rng(seed)
    unpaired = np.array(
        [np.median(x[i]) - np.median(y[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    )
    u_lo, u_hi = np.percentile(unpaired, [2.5, 97.5])

    lo, hi, width = st.bootstrap_median_diff_ci(x - y, n_boot=n_boot, seed=seed)

    assert (round(lo * 100, 2), round(hi * 100, 2)) == (-0.20, 0.15)
    assert (round(float(u_lo) * 100, 2), round(float(u_hi) * 100, 2)) == (-0.37, 0.27)
    assert width * 100 < 0.5 * (float(u_hi) - float(u_lo)) * 100 + 0.05


def test_rank_biserial_is_not_the_defective_formula(st: ModuleType) -> None:
    """``1 - 2W/(n(n+1))`` cannot go negative; the matched-pairs value must."""
    d = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    res = st.wilcoxon_signed_rank(d)
    defective = 1.0 - 2.0 * res.w_statistic / (res.n * (res.n + 1))
    assert defective == pytest.approx(1.0)
    assert res.rank_biserial_r == pytest.approx(-1.0)


def test_issue_390_reproduction_from_committed_records(st: ModuleType) -> None:
    """#398's Done-when clause: the summarizer agrees with #390's script.

    No ``skipif``. A missing fixture is a failure, not a silent pass — the whole
    point of this file is that green means checked.
    """
    assert PETS_JSON.is_file(), f"missing regression fixture: {PETS_JSON}"
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

    # The values #390 quotes, recomputed here by the independent path scipy provides.
    ref = scipy_wilcoxon(x, y, alternative="two-sided")
    assert (res.w_plus, res.w_minus) == (14.0, 1.0)
    assert res.w_statistic == pytest.approx(float(ref.statistic))
    assert res.p_value == pytest.approx(float(ref.pvalue), abs=1e-12)
    assert res.rank_biserial_r == pytest.approx(0.8667, abs=1e-4)
    assert 1.0 - 2.0 * res.w_statistic / (res.n * (res.n + 1)) == pytest.approx(0.9333, abs=1e-4)
    assert float(np.median(x - y)) * 100 == pytest.approx(1.589, abs=1e-3)
    assert float(np.median(x) - np.median(y)) * 100 == pytest.approx(2.291, abs=1e-3)


def test_2r_minus_1_conversion_rule_recovers_only_the_magnitude(st: ModuleType) -> None:
    """The published conversion ``true r = 2*printed - 1`` is sign-blind.

    Pins why the findings documents cannot simply be converted in place: the
    defective statistic threw the sign away, so no function of it recovers one.
    """
    for d in (np.arange(1.0, 6.0), -np.arange(1.0, 6.0)):
        res = st.wilcoxon_signed_rank(d)
        printed = 1.0 - 2.0 * res.w_statistic / (res.n * (res.n + 1))
        assert 2 * printed - 1 == pytest.approx(abs(res.rank_biserial_r))


# --------------------------------------------------------------------------- #
# TIER 4 — tolerance brackets on _tie_groups
# --------------------------------------------------------------------------- #
def test_tie_tolerance_is_bracketed_on_both_sides(st: ModuleType) -> None:
    r = st.TIE_RTOL
    assert len(np.unique(st._tie_groups(np.array([1.0, 1.0 + 0.5 * r])))) == 1
    assert len(np.unique(st._tie_groups(np.array([1.0, 1.0 + 2.0 * r])))) == 2


def test_tie_groups_do_not_chain(st: ModuleType) -> None:
    """Single-linkage chaining would merge values far beyond the tolerance.

    Comparing each value only against its predecessor makes the relation
    non-transitive: four values spaced 0.9*rtol apart span 2.7*rtol end to end
    and must not end up in one group.
    """
    r = st.TIE_RTOL
    a = np.array([1.0, 1.0 + 0.9 * r, 1.0 + 1.8 * r, 1.0 + 2.7 * r])
    assert (a[-1] - a[0]) / a[-1] > r, "fixture no longer spans more than the tolerance"
    assert len(np.unique(st._tie_groups(a))) > 1, "tolerance chained across a 2.7x span"


def test_real_near_tie_from_waterbirds_is_one_group(st: ModuleType) -> None:
    """The near-tie that moves a published p-value must be detected as a tie.

    ``bnnr_xai vs dfr`` on ``results_waterbirds_b15.json`` carries two |d| values
    7.2e-16 apart relative. They are mathematically equal, so the exact branch's
    no-ties precondition is violated and ``approx`` is the correct method.
    """
    a, b = 0.15420560747663548, 0.1542056074766356
    assert a != b and abs(a - b) / max(a, b) < 1e-15
    assert len(np.unique(st._tie_groups(np.array([a, b])))) == 1
    assert st.wilcoxon_p_method(np.array([a, -b, 0.2, -0.25])) == "approx"


def test_the_zero_and_tie_rationale_holds_on_the_real_records(st: ModuleType) -> None:
    """Pin the empirical claims the module docstring makes, so they are checked.

    The zero rule uses exact ``!= 0.0`` while ties use a tolerance. That rests on
    two measurable facts about this repository's records, not on an argument:
    equal paired values are bit-equal, and genuinely distinct differences sit
    many orders of magnitude above float64 rounding. A prose justification that
    no test reads is how "no SpuriousBench number changes" survived 1723 tests.
    """
    records = BENCHMARKS_DIR / "results_imagewoof_scratch.json"
    assert records.is_file(), f"missing regression fixture: {records}"
    runs = json.loads(records.read_text(encoding="utf-8"))["runs"]
    by: dict[int, dict[str, float]] = {}
    for r in runs:
        metric = r.get("held_out_test_metric")
        if metric is not None:
            by.setdefault(int(r["seed"]), {})[r["condition"]] = float(metric)

    seeds = sorted(s for s in by if {"bnnr_xai", "bnnr_random"} <= set(by[s]))
    pairs = [(by[s]["bnnr_xai"], by[s]["bnnr_random"]) for s in seeds]

    # 1. Tied pairs are bit-equal, so exact-zero detection loses nothing.
    tied = [(a, b) for a, b in pairs if a - b == 0.0]
    assert len(tied) == 2, f"expected 2 tied pairs in the T20 headline, got {len(tied)}"
    for a, b in tied:
        assert a == b and a.hex() == b.hex(), "a tied pair is not byte-identical"

    # 2. Genuinely distinct differences are nowhere near float64 rounding.
    nonzero = [abs(a - b) for a, b in pairs if a - b != 0.0]
    assert min(nonzero) > 1e-4, f"smallest non-zero |d| = {min(nonzero):.3e}"
    assert min(nonzero) / math.ulp(1e-2) > 1e12, "no longer orders of magnitude above ULP"

    # 3. Waterbirds worst-group accuracy really is quantised at 1/642.
    assert 1 / 642 == pytest.approx(1.558e-3, rel=1e-3)


def test_exact_branch_requires_no_ties(st: ModuleType) -> None:
    assert st.wilcoxon_p_method(np.array([1.0, -1.0, 2.0])) == "approx"
    assert st.wilcoxon_p_method(np.array([1.0, -2.0, 3.0])) == "exact"


# --------------------------------------------------------------------------- #
# Guards on inputs the engine of record should not answer for silently
# --------------------------------------------------------------------------- #
def test_non_finite_differences_are_rejected(st: ModuleType) -> None:
    """A NaN is neither dropped nor propagated, so it silently breaks the ranks.

    ``d > 0`` and ``d < 0`` are both False for NaN, so ``W+ + W-`` stops
    equalling ``n(n+1)/2`` and the rank-biserial denominator is wrong.
    """
    for bad in (np.array([1.0, -2.0, np.nan, 3.0]), np.array([1.0, np.inf, -2.0])):
        with pytest.raises(ValueError):
            st.wilcoxon_signed_rank(bad)


def test_exact_p_refuses_n_beyond_the_supported_range(st: ModuleType) -> None:
    """``_wilcoxon_null_counts`` is int64: at n >= 64 the counts silently wrap."""
    with pytest.raises(ValueError):
        st.wilcoxon_exact_p(0.0, st.N_EXACT_MAX + 1)


def test_null_counts_are_a_symmetric_distribution(st: ModuleType) -> None:
    for n in range(1, 13):
        counts = st._wilcoxon_null_counts(n)
        assert int(counts.sum()) == 2**n
        assert len(counts) == n * (n + 1) // 2 + 1
        assert int(counts[0]) == 1
        assert list(counts) == list(counts[::-1])


def test_all_zero_and_empty_inputs(st: ModuleType) -> None:
    res = st.wilcoxon_signed_rank(np.zeros(5))
    assert (res.n, res.n_dropped, res.p_value, res.rank_biserial_r) == (0, 5, 1.0, 0.0)
    lo, hi, width = st.bootstrap_median_diff_ci(np.array([]))
    assert math.isnan(lo) and math.isnan(hi) and math.isnan(width)
