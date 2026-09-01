"""Paired statistics for the benchmark summarizers — the single implementation.

Every estimator the benchmark reports quotes lives here, so that
``summarize_grand.py`` and ``summarize_spurious.py`` cannot drift apart. The
implementations are the ones reviewed in #389; the two defects reported in #390
are fixed here (see ``wilcoxon_signed_rank`` and ``bootstrap_median_diff_ci``).

Pure numpy on purpose. scipy is not a declared dependency of this project — it
is present in CI only transitively — so a scipy-preferring code path silently
changes which engine computed a published p-value depending on what else the
reader happened to install. One engine of record, deterministic, exact where
the data permits.

Conventions, applied uniformly:

* **Zeros.** Zero differences are dropped and ``n`` is reduced (the Wilcoxon
  convention). The ``n`` reported by :class:`WilcoxonResult` is the ``n`` the
  test ran on, not the number of pairs supplied — those differ whenever any
  pair is tied, and quoting the latter overstates the evidence.
* **Ties.** Tied ``|d|`` receive mid-ranks. Ties are detected with a relative
  tolerance, not exact equality, because the metrics compared here are discrete
  — Waterbirds ``worst_group_acc`` is quantised at ``1/642 ≈ 1.6e-3`` — and two
  mathematically equal differences between such values can land ~1e-16 apart in
  floating point. The live example is the ``bnnr_xai vs dfr`` contrast on
  ``results_waterbirds_b15.json``, where two ``|d|`` sit 7.2e-16 apart relative;
  exact equality missed the tie and the contrast claimed an exact branch whose
  no-ties precondition it violated.
* **Zeros are *not* tolerance-detected**, and the asymmetry with ties is
  deliberate. It rests on one assumption: both metrics in a pair share a
  denominator, so equal values are bit-equal and a mathematically zero
  difference is exactly ``0.0``. That is checked, not assumed: both tied pairs
  in the T20 headline contrast (seeds 42 and 48) hold byte-identical floats.
  A genuine difference is nowhere near float64 rounding either — the smallest
  non-zero ``|d|`` is 8.3e-4 over the nine contrasts the summarizer tests, and
  1.8e-4 over all 45 condition pairs, against ``ulp(8.3e-4) = 1.1e-19``. Under those
  equality is the right test and a tolerance would discard real observations.
  Where it weakens: metrics whose denominators differ *between the two
  conditions being compared* — ``worst_group_acc`` is the live candidate, since
  the worst group may be group 1 (n=642) for one condition and group 2 (n=214)
  for the other. No contrast in this repository does that today, and a test pins
  the behaviour.
* **Effect size.** ``(W⁺ − W⁻)/(W⁺ + W⁻)``, which is intrinsically signed. No
  sign is applied from outside.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

__all__ = [
    "N_EXACT_MAX",
    "TIE_RTOL",
    "WilcoxonResult",
    "bootstrap_median_diff_ci",
    "holm_bonferroni",
    "wilcoxon_exact_p",
    "wilcoxon_p_method",
    "wilcoxon_signed_rank",
    "wilson_ci",
]

N_EXACT_MAX = 25
"""Largest n for which the exact null is enumerated (DP, not 2**n)."""

TIE_RTOL = 1e-9
"""Relative tolerance for calling two |d| values tied.

Sized from measurements on this repo's own contrasts rather than from a round
number, and stated as the two quantities a *relative* tolerance sits between.

Below it: mathematically equal differences land ~1e-16 apart relatively —
observed 7.2e-16 on the Waterbirds b15 ``bnnr_xai vs dfr`` contrast, which is
the near-tie that moved a published p-value.

Above it: the smallest relative separation between two genuinely distinct
``|d|`` values is 4.0e-3, on the Imagewoof ``icd_aicd_fixed`` contrast.

So the tolerance has ~6 orders of magnitude of headroom above the noise and
~6.5 below the finest real distinction. (The smallest *absolute* non-zero
``|d|`` is 8.3e-4 over the nine contrasts the summarizer tests, but absolute
figures do not bound a relative tolerance and are not what sizes this constant.)
"""


class WilcoxonResult(NamedTuple):
    """Outcome of a two-sided paired Wilcoxon signed-rank test.

    Attributes:
        w_plus: Sum of ranks of positive differences.
        w_minus: Sum of ranks of negative differences.
        p_value: Two-sided p-value, by the method named in ``method``.
        rank_biserial_r: Matched-pairs rank-biserial correlation, signed.
        n: Number of differences the test ran on (zeros already dropped).
        n_dropped: Number of zero differences dropped.
        method: ``"exact"`` or ``"approx"``.
    """

    w_plus: float
    w_minus: float
    p_value: float
    rank_biserial_r: float
    n: int
    n_dropped: int
    method: str

    @property
    def w_statistic(self) -> float:
        """``min(W⁺, W⁻)``, the conventional two-sided test statistic.

        This is the quantity scipy reports as ``statistic``. It carries the
        magnitude of the effect but not its direction — read
        ``rank_biserial_r`` for direction.
        """
        return min(self.w_plus, self.w_minus)

    @property
    def method_label(self) -> str:
        """Human-readable method, noting dropped zeros when there were any."""
        if self.method == "exact" and self.n_dropped:
            return f"exact (conditional; {self.n_dropped} zeros dropped)"
        return self.method


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _tie_groups(a: np.ndarray, rtol: float = TIE_RTOL) -> np.ndarray:
    """Cluster near-equal magnitudes, returning a monotone group id per element.

    Each sorted value is compared against the **anchor** of the open group — the
    first value admitted to it — rather than against its immediate predecessor.
    That bound matters: comparing neighbours makes the relation non-transitive,
    so a chain of values each within ``rtol`` of the last merges into one group
    however far the chain runs. Anchoring caps a group's total width at ``rtol``
    relative to its anchor, which is what "these are the same measurement" is
    supposed to mean.

    Group ids increase with value, so ranking the ids yields exactly the
    mid-ranks that ranking the values would give if the near-ties were exact.
    """
    n = len(a)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(a, kind="mergesort")
    groups = np.empty(n, dtype=np.int64)
    gid = 0
    anchor = a[order[0]]
    groups[order[0]] = 0
    for i in range(1, n):
        cur = a[order[i]]
        if abs(cur - anchor) > rtol * max(abs(anchor), abs(cur)):
            gid += 1
            anchor = cur
        groups[order[i]] = gid
    return groups


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their rank block), 1-based."""
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _wilcoxon_null_counts(n: int) -> np.ndarray:
    """Exact null distribution of the signed-rank statistic for ``n`` pairs.

    Returns an integer array ``c`` of length n(n+1)/2 + 1 where ``c[w]`` is the
    number of the ``2**n`` equally likely sign assignments giving ``W⁻ == w``.
    Computed by subset-sum DP over ranks 1..n, which is exact and fast at
    n <= 25 where naive enumeration would be 33M assignments.
    """
    max_w = n * (n + 1) // 2
    counts = np.zeros(max_w + 1, dtype=np.int64)
    counts[0] = 1
    for r in range(1, n + 1):
        counts[r:] += counts[:-r].copy()
    return counts


def _split(diffs: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop zero differences, returning the survivors and how many went.

    Non-finite input is rejected rather than tolerated: ``d > 0`` and ``d < 0``
    are both False for ``NaN``, so a single one would leave it unranked while
    still counted in ``n`` — ``W⁺ + W⁻`` would stop equalling ``n(n+1)/2`` and
    both the rank-biserial denominator and the null moments would be wrong, with
    no visible symptom.
    """
    d = np.asarray(diffs, dtype=float)
    if d.size and not np.all(np.isfinite(d)):
        raise ValueError(
            "paired differences must be finite; got NaN or inf. Drop or impute "
            "the offending pairs before testing, so that n reflects a decision."
        )
    nz = d[d != 0.0]
    return nz, int(len(d) - len(nz))


def _method_for(d: np.ndarray, groups: np.ndarray) -> str:
    """Exact or approx, from already-split differences and their tie groups.

    Single source of truth, so :func:`wilcoxon_p_method` and
    :func:`wilcoxon_signed_rank` cannot disagree about which branch ran.
    """
    n = len(d)
    if n == 0 or n > N_EXACT_MAX:
        return "approx"
    return "approx" if len(np.unique(groups)) < n else "exact"


# --------------------------------------------------------------------------- #
# Public estimators
# --------------------------------------------------------------------------- #
def wilcoxon_exact_p(w_min: float, n: int) -> float:
    """Exact two-sided p for an observed ``min(W⁺, W⁻)`` under the null.

    Two-sided is ``P(W <= w_min) + P(W >= max_W - w_min)``, which by the
    symmetry of the null equals ``2 * P(W <= w_min)``, capped at 1. Valid only
    when the retained differences carry no ties in ``|d|``; the caller enforces
    that via :func:`wilcoxon_p_method`.

    Raises for ``n > N_EXACT_MAX``. The counts are int64 and their total is
    ``2**n``, so past n = 63 the DP wraps silently and returns a plausible
    p-value computed from a distribution that sums to zero.
    """
    if n <= 0:
        return 1.0
    if n > N_EXACT_MAX:
        raise ValueError(
            f"exact enumeration is supported up to n = {N_EXACT_MAX}, got n = {n}; "
            "use wilcoxon_signed_rank, which falls back to the approximation"
        )
    counts = _wilcoxon_null_counts(n)
    lo = float(counts[: int(math.floor(w_min)) + 1].sum())
    return min(1.0, 2.0 * lo / float(2 ** n))


def wilcoxon_p_method(diffs: np.ndarray) -> str:
    """Which p-value method :func:`wilcoxon_signed_rank` will use.

    Exact requires ``n <= N_EXACT_MAX`` after dropping zeros and no ties in
    ``|d|`` (tolerance ``TIE_RTOL``). Returns ``"exact"`` or ``"approx"``;
    report it alongside the p-value so the reader knows which is on the table.

    **Dropped zeros do not cost exactness.** Conditional on which pairs tied and
    on the retained ``|d|``, the null makes the signs of the survivors i.i.d.
    uniform, so enumerating their ``2**n`` assignments is the exact conditional
    null — Wilcoxon's own zero handling, as against Pratt's, which keeps the
    zeros and shifts the statistic. What "exact" then means is *exact given n*,
    which is why :attr:`WilcoxonResult.method_label` says so out loud.

    One caveat worth stating because it is easy to get wrong: this is **not**
    what scipy's ``method='auto'`` does. scipy routes away from its exact branch
    whenever a zero is present, to a permutation test that is exhaustive only up
    to 13 pairs and to the normal approximation above that. On
    ``d = [0, 0, 0, 1, 2, ..., 18]`` — 21 pairs, 3 zeros, no ties — this module
    returns 7.63e-06 and ``scipy.wilcoxon(d, method='auto')`` returns 1.96e-04,
    a factor of 26. The ratio is data-dependent, hence the explicit fixture.
    Feed scipy the already dropped non-zeros and it agrees with this module
    exactly.
    """
    d, _ = _split(diffs)
    return _method_for(d, _tie_groups(np.abs(d)))


def wilcoxon_signed_rank(diffs: np.ndarray) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Zeros are dropped and ``n`` reduced. Tied ``|d|`` take mid-ranks, with ties
    detected to within ``TIE_RTOL``.

    The effect size is the matched-pairs rank-biserial correlation
    ``(W⁺ − W⁻)/(W⁺ + W⁻)``, which ranges over [-1, +1] and is *intrinsically
    signed* — no external sign convention is applied. Its sign is the sign of
    ``W⁺ − W⁻`` and nothing else. In particular it is **not** the sign of the
    median paired difference: ranks carry magnitude while the median only
    counts, so ``d = [1, 2, 3, -100, -101]`` has median +1 and r = -0.2. The two
    agree whenever every retained difference shares a sign, and disagree exactly
    when a minority of large-magnitude differences outweighs a majority of small
    ones — which is information worth keeping, not a defect. When the median is
    exactly zero (the T20 headline contrast: median 0.000, W⁺ = 17, W⁻ = 19) the
    sign is still well defined.

    This replaces ``1 - 2W/(n(n+1))`` with ``W = min(W⁺, W⁻)``, reported in
    #390: that expression equals ``(1 + |r|)/2``, so it lived in [0.5, 1.0],
    could never be negative, and printed ≈ 0.50 for a perfect null.
    """
    d, n_dropped = _split(diffs)
    n = len(d)
    if n == 0:
        return WilcoxonResult(0.0, 0.0, 1.0, 0.0, 0, n_dropped, "approx")

    groups = _tie_groups(np.abs(d))
    ranks = _rankdata(groups)
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = w_plus + w_minus
    rbr = (w_plus - w_minus) / total if total > 0 else 0.0
    w_min = min(w_plus, w_minus)
    method = _method_for(d, groups)

    if method == "exact":
        p = wilcoxon_exact_p(w_min, n)
    else:
        mean_w = n * (n + 1) / 4.0
        _, counts = np.unique(groups, return_counts=True)
        tie_term = float((counts.astype(np.int64) ** 3 - counts).sum())
        var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
        if var_w <= 0:
            p = 1.0
        else:
            z = (w_min - mean_w + 0.5) / math.sqrt(var_w)
            p = min(1.0, max(0.0, 2.0 * _norm_cdf(z)))

    return WilcoxonResult(w_plus, w_minus, p, rbr, n, n_dropped, method)


def bootstrap_median_diff_ci(
    diffs: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on the **median of the paired differences**.

    Returns ``(lo, hi, width)`` in the units of ``diffs``.

    The quantity is ``median(x - y)``, matching the Wilcoxon test printed
    beside it. #390 reported that the shipped code bootstrapped
    ``median(x) - median(y)`` instead — a difference of medians, which is a
    different estimand and on the T20 headline contrast produced an interval
    roughly twice as wide (0.64 pp against 0.34 pp).
    """
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        nan = float("nan")
        return nan, nan, nan
    rng = np.random.default_rng(seed)
    n = len(d)
    # One (n_boot, n) draw, not n_boot draws of n. numpy fills row-major and
    # consumes the bit stream per element either way, so the index matrix is
    # identical to the loop's and every published interval is unchanged --
    # verified bit-for-bit on the T20 headline contrast before this was applied.
    meds = np.median(d[rng.integers(0, n, (n_boot, n))], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi), float(hi - lo)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Monotone by construction: the running maximum keeps an adjusted value from
    falling below one assigned to a smaller raw p. Each result is capped at 1.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n``.

    Preferred over the normal-approximation interval for the per-group
    accuracies reported here, where ``n`` is small (≈642 per Waterbirds test
    group) and the proportion can sit near 0 or 1. Returns ``(lo, hi)``
    clipped to [0, 1]; ``(nan, nan)`` when ``n == 0``.
    """
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)
