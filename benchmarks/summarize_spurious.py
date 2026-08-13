#!/usr/bin/env python3
"""Summarize a SpuriousBench results directory into a markdown report (T4).

Implements the frozen statistics contract (plan v1.5 §4):
  - WGA table: per-seed + median, per condition
  - Pairwise contrasts on WGA: paired two-sided Wilcoxon signed-rank on identical
    seed sets, Holm-Bonferroni over the pairwise family, bootstrap 95% CI on the
    paired median difference (width stated in prose), matched-pairs rank-biserial
    r = (W+ - W-)/(W+ + W-) reported CONSISTENT with the printed p
  - Faithfulness trio: EBPG + IoU@0.5 + Pointing Game (endpoint)
  - Epochs-to-threshold (D-ETT): primary = smallest epoch with WGA >= 0.80*final;
    secondary = absolute WGA >= 0.70, censored at budget with counts
  - Wilson 95% CIs per group accuracy
  - plain_mean_acc labeled "overall test accuracy (test-prevalence weighted)"
  - Auto-calibrated language: n < 6 -> direction / sign-consistency only, no
    p-values (min achievable p at small n makes them meaningless); n >= 6 ->
    inferential tests. WARN on incomplete/ragged seed sets.

Pure numpy + stdlib (no scipy dependency added). Wilcoxon, Wilson, bootstrap are
implemented directly so the summarizer introduces no new heavy dependency.

Usage:
    python benchmarks/summarize_spurious.py <results_dir_or_json> [--out report.md]
                                            [--label "conv3/old-pipeline pilot"]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONDITION_ORDER = ["base_frozen", "erm_continue", "dfr", "bnnr_random", "bnnr_xai"]
# Pairwise WGA family: the headline (xai vs random) plus the standard baselines.
PAIRS = [
    ("bnnr_xai", "bnnr_random"),
    ("bnnr_xai", "dfr"),
    ("bnnr_xai", "erm_continue"),
    ("bnnr_random", "erm_continue"),
    ("dfr", "erm_continue"),
]
N_INFERENTIAL_MIN = 6  # below this: direction/sign-consistency only, no p-values


# --------------------------------------------------------------------------- #
# Stats primitives (pure numpy)
# --------------------------------------------------------------------------- #
N_EXACT_MAX = 25  # D-EXACT-P: exact enumeration up to this n (no zeros/ties)


def _wilcoxon_null_counts(n: int) -> np.ndarray:
    """Exact null distribution of the signed-rank statistic W for n pairs.

    Returns an integer array ``c`` of length n(n+1)/2 + 1 where ``c[w]`` is the
    number of the 2**n equally likely sign assignments giving W- == w. Computed
    by subset-sum DP over ranks 1..n (O(n^2 * max_W)), which is exact and fast
    at n <= 25 where naive 2**n enumeration would be 33M assignments.
    """
    max_w = n * (n + 1) // 2
    counts = np.zeros(max_w + 1, dtype=np.int64)
    counts[0] = 1
    for r in range(1, n + 1):
        counts[r:] += counts[:-r].copy()
    return counts


def wilcoxon_exact_p(w_min: float, n: int) -> float:
    """Exact two-sided p for observed min(W+, W-) under the signed-rank null.

    Two-sided = P(W <= w_min) + P(W >= max_W - w_min), which by the symmetry of
    the null equals 2 * P(W <= w_min) (capped at 1). Valid only when there are
    no zero differences and no ties in |d| (the caller enforces this).
    """
    counts = _wilcoxon_null_counts(n)
    total = float(2 ** n)
    lo = float(counts[: int(math.floor(w_min)) + 1].sum())
    return min(1.0, 2.0 * lo / total)


def wilcoxon_signed_rank(diffs: np.ndarray) -> tuple[float, float, float, float]:
    """Two-sided Wilcoxon signed-rank on paired differences.

    Returns (W_plus, W_minus, p_two_sided, rank_biserial_r). Zeros are dropped
    (Wilcoxon convention).

    D-EXACT-P: when n <= N_EXACT_MAX and the retained differences contain no
    ties in |d|, p is EXACT (full enumeration of the null via DP). Otherwise it
    falls back to the tie-corrected normal approximation with continuity
    correction. Use ``wilcoxon_p_method`` to label which was used.
    rank_biserial = (W+ - W-)/(W+ + W-), sign consistent with the median diff.
    """
    d = diffs[diffs != 0]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0, 0.0
    ranks = _rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = w_plus + w_minus
    rbr = (w_plus - w_minus) / total if total > 0 else 0.0
    w = min(w_plus, w_minus)

    if wilcoxon_p_method(diffs) == "exact":
        return w_plus, w_minus, wilcoxon_exact_p(w, n), rbr

    # Normal approximation with tie correction
    mean_w = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = (counts**3 - counts).sum()
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        p = 1.0
    else:
        z = (w - mean_w + 0.5) / math.sqrt(var_w)
        p = 2.0 * _norm_cdf(z)
        p = min(1.0, max(0.0, p))
    return w_plus, w_minus, p, rbr


def wilcoxon_p_method(diffs: np.ndarray) -> str:
    """Which p-value method ``wilcoxon_signed_rank`` uses: 'exact' or 'approx'.

    Exact requires n <= N_EXACT_MAX after dropping zeros AND no ties in |d|.
    Reported in the output so the reader knows which is on the table.
    """
    d = diffs[diffs != 0]
    n = len(d)
    if n == 0:
        return "approx"
    has_ties = len(np.unique(np.abs(d))) < n
    n_dropped = len(diffs) - n
    if n <= N_EXACT_MAX and not has_ties and n_dropped == 0:
        return "exact"
    return "approx"


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties get mean rank), 1-based."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down), monotone."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_median_diff_ci(diffs: np.ndarray, n_boot: int = 10000,
                             seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap 95% CI on the paired median difference. Returns (lo, hi, width)."""
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        sample = diffs[rng.integers(0, n, n)]
        meds[b] = np.median(sample)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi), float(hi - lo)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_runs(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted(path.glob("results_*.json"))
        if not files:
            raise SystemExit(f"No results_*.json in {path}")
        runs: list[dict[str, Any]] = []
        for f in files:
            runs.extend(json.loads(f.read_text()).get("runs", []))
        return runs
    return json.loads(path.read_text()).get("runs", [])


def by_condition_seed(runs: list[dict[str, Any]]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for r in runs:
        out.setdefault(r["condition"], {})[r["seed"]] = r
    return out


def paired_wga(cs: dict[str, dict[int, dict]], a: str, b: str
               ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """WGA arrays for conditions a,b on their COMMON seed set (paired)."""
    sa, sb = cs.get(a, {}), cs.get(b, {})
    seeds = sorted(set(sa) & set(sb))
    xa = np.array([sa[s]["worst_group_acc"] for s in seeds])
    xb = np.array([sb[s]["worst_group_acc"] for s in seeds])
    return xa, xb, seeds


# --------------------------------------------------------------------------- #
# D-ETT epochs-to-threshold
# --------------------------------------------------------------------------- #
def epochs_to_threshold(wga_curve: list[float], budget: int
                        ) -> tuple[Any, Any]:
    """(primary, secondary) per D-ETT.
    primary = smallest epoch index (1-based) with WGA >= 0.80 * final;
    secondary = smallest epoch with WGA >= 0.70 absolute, else 'censored'."""
    if not wga_curve:
        return None, None
    final = wga_curve[-1]
    prim = next((i + 1 for i, w in enumerate(wga_curve) if w >= 0.80 * final), None)
    sec = next((i + 1 for i, w in enumerate(wga_curve) if w >= 0.70), "censored")
    return prim, sec


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}" if not (isinstance(x, float) and math.isnan(x)) else "—"


def build_report(runs: list[dict[str, Any]], label: str | None) -> str:
    cs = by_condition_seed(runs)
    present = [c for c in CONDITION_ORDER if c in cs]
    seed_sets = {c: sorted(cs[c]) for c in present}
    all_seeds = sorted({s for c in present for s in seed_sets[c]})
    n = len(all_seeds)
    ragged = len({tuple(seed_sets[c]) for c in present}) > 1
    inferential = n >= N_INFERENTIAL_MIN

    L: list[str] = []
    L.append("# SpuriousBench — summary report")
    if label:
        L.append(f"\n> **{label}**\n")
    L.append(f"\nConditions present: {', '.join(present)}")
    L.append(f"Seeds: n={n} ({all_seeds})")
    if ragged:
        L.append("\n**WARNING: incomplete/ragged seed sets across conditions** — "
                 "per-condition seed lists differ; paired tests use only common "
                 "seeds per pair. Seed coverage:")
        for c in present:
            L.append(f"  - {c}: {seed_sets[c]}")
    if not inferential:
        L.append(f"\n**Calibrated language: n={n} < {N_INFERENTIAL_MIN} → "
                 "direction and sign-consistency only; no p-values reported "
                 "(minimum achievable Wilcoxon p at this n is too large to be "
                 "meaningful).**")

    # WGA table
    L.append("\n## Worst-group accuracy (WGA)\n")
    header = "| condition | " + " | ".join(f"s{s}" for s in all_seeds) + " | median |"
    sep = "|" + "---|" * (len(all_seeds) + 2)
    L.append(header)
    L.append(sep)
    for c in present:
        cells = []
        for s in all_seeds:
            r = cs[c].get(s)
            cells.append(fmt_pct(r["worst_group_acc"]) if r else "·")
        med = np.median([cs[c][s]["worst_group_acc"] for s in seed_sets[c]])
        L.append(f"| {c} | " + " | ".join(cells) + f" | **{fmt_pct(med)}** |")

    # Pairwise contrasts
    L.append("\n## Pairwise WGA contrasts (paired)\n")
    raw_ps: list[float] = []
    methods: list[str] = []  # D-EXACT-P: 'exact' or 'approx' per contrast
    pair_rows: list[tuple[str, str, np.ndarray, list[int], float, float, float,
                          float, float, float]] = []
    for a, b in PAIRS:
        if a not in cs or b not in cs:
            continue
        xa, xb, seeds = paired_wga(cs, a, b)
        if len(seeds) == 0:
            continue
        diffs = xa - xb
        med_diff = float(np.median(diffs))
        w_plus, w_minus, p, rbr = wilcoxon_signed_rank(diffs)
        lo, hi, width = bootstrap_median_diff_ci(diffs)
        raw_ps.append(p)
        methods.append(wilcoxon_p_method(diffs))
        pair_rows.append((a, b, diffs, seeds, med_diff, p, rbr, lo, hi, width))

    holm = holm_bonferroni([r[5] for r in pair_rows]) if pair_rows else []

    if inferential:
        L.append("| contrast | n | median Δ WGA (pp) | rank-biserial r | "
                 "p (Wilcoxon) | method | p (Holm) | 95% CI median Δ (pp) |")
        L.append("|---|---|---|---|---|---|---|---|")
    else:
        L.append("| contrast | n | median Δ WGA (pp) | sign-consistent | "
                 "rank-biserial r | 95% CI median Δ (pp) |")
        L.append("|---|---|---|---|---|---|")
    for i, (a, b, diffs, seeds, med_diff, p, rbr, lo, hi, width) in enumerate(pair_rows):
        pp = med_diff * 100
        ci = f"[{lo*100:+.2f}, {hi*100:+.2f}]"
        if inferential:
            L.append(f"| {a} vs {b} | {len(seeds)} | {pp:+.2f} | {rbr:+.3f} | "
                     f"{p:.4f} | {methods[i]} | {holm[i]:.4f} | {ci} |")
        else:
            sign_consistent = "yes" if np.all(diffs > 0) or np.all(diffs < 0) else "no"
            L.append(f"| {a} vs {b} | {len(seeds)} | {pp:+.2f} | {sign_consistent} | "
                     f"{rbr:+.3f} | {ci} |")

    if inferential and pair_rows:
        n_exact = sum(1 for m in methods if m == "exact")
        L.append(f"\n_p-value method (D-EXACT-P): `exact` = full enumeration of "
                 f"the signed-rank null over all 2^n sign assignments (used when "
                 f"n ≤ {N_EXACT_MAX} with no zero differences and no ties in |d|); "
                 f"`approx` = normal approximation with continuity and tie "
                 f"correction. {n_exact}/{len(methods)} contrasts exact. Holm-"
                 f"Bonferroni is applied over this pairwise family of "
                 f"{len(methods)}._")

    # CI width prose
    if pair_rows:
        L.append("\n**CI widths (what we can exclude):**")
        for a, b, diffs, seeds, med_diff, p, rbr, lo, hi, width in pair_rows:
            L.append(f"- {a} vs {b}: paired median Δ = {med_diff*100:+.2f} pp, "
                     f"95% CI width = {width*100:.2f} pp "
                     f"(rules out effects outside [{lo*100:+.2f}, {hi*100:+.2f}] pp).")

    # Faithfulness trio (endpoint)
    L.append("\n## Faithfulness (endpoint, vs ground-truth masks)\n")
    L.append("| condition | EBPG | IoU@0.5 | Pointing Game | probe size |")
    L.append("|---|---|---|---|---|")
    for c in present:
        eb = np.median([cs[c][s].get("ebpg_mean", float("nan")) for s in seed_sets[c]])
        iou = np.median([cs[c][s].get("iou_mean", float("nan")) for s in seed_sets[c]])
        pg = np.median([cs[c][s].get("pg_mean", float("nan")) for s in seed_sets[c]])
        ps = cs[c][seed_sets[c][0]].get("faith_probe_size", 0)
        L.append(f"| {c} | {eb:.3f} | {iou:.3f} | {pg:.3f} | {ps} |")

    # Epochs-to-threshold (D-ETT)
    L.append("\n## Epochs-to-threshold (D-ETT)\n")
    L.append("Primary: smallest epoch with WGA ≥ 0.80×final. "
             "Secondary: WGA ≥ 70% absolute (else censored at budget).\n")
    L.append("| condition | median epochs→0.80×final | →70% abs (censored count) |")
    L.append("|---|---|---|")
    for c in present:
        prims, secs, cens = [], [], 0
        for s in seed_sets[c]:
            r = cs[c][s]
            curve = r.get("wga_per_epoch", [])
            budget = r.get("budget", 0)
            if not curve:
                continue
            prim, sec = epochs_to_threshold(curve, budget)
            if prim is not None:
                prims.append(prim)
            if sec == "censored":
                cens += 1
            elif isinstance(sec, int):
                secs.append(sec)
        pm = f"{np.median(prims):.1f}" if prims else "—"
        sm = f"{np.median(secs):.1f}" if secs else "—"
        L.append(f"| {c} | {pm} | {sm} ({cens} censored) |")

    # Per-group Wilson CIs (from group_acc; needs per-group n — use test counts if present)
    L.append("\n## Per-group accuracy with Wilson 95% CIs (median seed)\n")
    L.append("_plain_mean_acc is reported elsewhere as "
             "\"overall test accuracy (test-prevalence weighted)\"._\n")
    L.append("| condition | group | acc | Wilson 95% CI |")
    L.append("|---|---|---|---|")
    any_missing_n = False
    for c in present:
        # pick the median-WGA seed as representative
        s_rep = seed_sets[c][len(seed_sets[c]) // 2]
        rec = cs[c][s_rep]
        ga = rec.get("group_acc", {})
        gn = rec.get("group_n", {}) or {}
        for gname, acc in ga.items():
            if isinstance(acc, float) and math.isnan(acc):
                continue
            # D-GROUP-N: real Wilson CI when per-group counts are present;
            # degrade to 'n/a' for legacy records written before the addition.
            n_g = gn.get(gname)
            if isinstance(n_g, int) and n_g > 0:
                lo, hi = wilson_ci(int(round(acc * n_g)), n_g)
                cell = f"[{lo*100:.2f}, {hi*100:.2f}] (n={n_g})"
            else:
                any_missing_n = True
                cell = "n/a (group_n absent from record)"
            L.append(f"| {c} | {gname} | {acc*100:.2f} | {cell} |")

    if any_missing_n:
        L.append("\n_Note: some records predate the `group_n` schema addition, "
                 "so their Wilson CIs are unavailable. Re-running those "
                 "conditions with the current harness populates the counts._")
    else:
        L.append("\n_Wilson 95% CIs computed from per-group test counts "
                 "(`group_n`). The representative seed is the middle element of "
                 "the sorted seed list, not the best or worst - CIs here are "
                 "descriptive per-group precision, not the paired inference._")

    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="results dir or results_*.json file")
    ap.add_argument("--out", default=None, help="write markdown here")
    ap.add_argument("--label", default=None, help="banner label for the report")
    args = ap.parse_args()

    runs = load_runs(Path(args.path))
    report = build_report(runs, args.label)
    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out} ({len(runs)} runs)")
    else:
        print(report)


if __name__ == "__main__":
    main()
