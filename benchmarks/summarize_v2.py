#!/usr/bin/env python3
"""Summarize benchmark v2 results with paper-quality statistics.

Features:
  - Wilcoxon signed-rank test (paired, per seed) for bnnr_xai vs each baseline
  - Bootstrap 95% CI for the difference in medians (10 000 resamplings, numpy-only)
  - Effect size: rank-biserial correlation r = 1 - 2W / (n*(n+1))
  - Holm-Bonferroni correction over the 4 simultaneous tests
  - Compute-transparency: GPU-epochs per condition
  - Dedicated section: bnnr_xai vs bnnr_random

Examples
--------
    python benchmarks/summarize_v2.py \\
        --results benchmarks/results_imagewoof_v2.json --markdown

    python benchmarks/summarize_v2.py \\
        --results benchmarks/results_imagewoof_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARKS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BENCHMARKS_DIR / "results_imagewoof_v2.json"

# All conditions in display order
ALL_CONDITIONS = [
    "no_aug",
    "randaugment",
    "trivialaugment",
    "churchnoise_only",
    "bnnr_xai",
    "bnnr_random",
]

DISPLAY_NAMES = {
    "no_aug": "No augmentation (crop + flip)",
    "randaugment": "RandAugment (torchvision)",
    "trivialaugment": "TrivialAugmentWide (torchvision)",
    "churchnoise_only": "ChurchNoise only (non-XAI ablation)",
    "bnnr_xai": "BNNR XAI-guided (equal compute)",
    "bnnr_random": "BNNR random selection (XAI ablation)",
}

# Conditions compared against bnnr_xai in the Wilcoxon tests
COMPARISON_CONDITIONS = ["no_aug", "randaugment", "trivialaugment", "bnnr_random"]


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (paired)
# ---------------------------------------------------------------------------


def _wilcoxon_signed_rank(
    x: list[float], y: list[float]
) -> tuple[float, float, float] | None:
    """Paired Wilcoxon signed-rank test (two-sided).

    Returns (W_statistic, p_value, rank_biserial_r) or None if n < 2.

    Algorithm: classic Wilcoxon signed-rank (Pratt, 1959; handling zero differences
    by discarding them). p-value via scipy.stats.wilcoxon if available, otherwise
    via the exact permutation distribution for n <= 10 (manual) or a normal
    approximation for larger n.

    rank_biserial_r = 1 - 2*W / (n*(n+1))
    where W = sum of ranks of negative differences (W+ = sum positive ranks).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length for paired test")
    if len(x) < 2:
        return None

    diffs = [float(xi) - float(yi) for xi, yi in zip(x, y)]

    # Try scipy first (accurate for all n)
    try:
        from scipy.stats import wilcoxon as _scipy_wilcoxon

        nonzero = [d for d in diffs if d != 0.0]
        if len(nonzero) == 0:
            return 0.0, 1.0, 0.0
        result = _scipy_wilcoxon(nonzero, alternative="two-sided")
        W = float(result.statistic)
        p = float(result.pvalue)
        n = len(nonzero)
        r = 1.0 - 2.0 * W / (n * (n + 1)) if n > 0 else 0.0
        return W, p, r
    except ImportError:
        pass

    # Manual fallback
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0.0

    # Rank absolute values
    abs_d = [abs(d) for d in nonzero]
    order = sorted(range(n), key=lambda i: abs_d[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    W_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)
    W_minus = sum(ranks[i] for i in range(n) if nonzero[i] < 0)
    W = min(W_plus, W_minus)

    # Exact distribution for small n (n <= 15)
    if n <= 15:
        p = _wilcoxon_exact_p(n, W)
    else:
        # Normal approximation with continuity correction
        mu = n * (n + 1) / 4.0
        sigma2 = n * (n + 1) * (2 * n + 1) / 24.0
        if sigma2 == 0:
            p = 1.0
        else:
            z = (W - mu + 0.5) / math.sqrt(sigma2)  # continuity correction toward 0
            p = _normal_two_sided_p(abs(z))

    r = 1.0 - 2.0 * W / (n * (n + 1))
    return W, min(1.0, p), r


def _wilcoxon_exact_p(n: int, W: float) -> float:
    """Exact two-sided p-value for Wilcoxon signed-rank statistic W (min of W+, W-).

    Enumerate all 2^n sign assignments of ranks 1..n; count how often
    min(W+, W-) <= observed W.
    """
    total = 1 << n  # 2^n
    count = 0
    ranks = list(range(1, n + 1))
    max_sum = sum(ranks)
    for mask in range(total):
        w_plus = sum(ranks[i] for i in range(n) if mask >> i & 1)
        w_minus = max_sum - w_plus
        w_stat = min(w_plus, w_minus)
        if w_stat <= W:
            count += 1
    return float(count) / float(total)


def _normal_two_sided_p(z: float) -> float:
    """Two-sided p-value from standard normal |z|.

    Uses the complementary error function for accuracy.
    """
    return float(math.erfc(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction.

    Returns adjusted p-values in the original order.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, running_max)
    return adjusted


# ---------------------------------------------------------------------------
# Bootstrap CI for median difference
# ---------------------------------------------------------------------------


def _bootstrap_median_diff_ci(
    x: list[float],
    y: list[float],
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """95% bootstrap CI for median(x) - median(y) (paired, with replacement).

    Returns (lower, upper) in percentage points if values are in [0,1].
    """
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    xa = np.array(x)
    ya = np.array(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.median(xa[idx]) - np.median(ya[idx]))
    alpha = 1.0 - ci
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# Significance stars
# ---------------------------------------------------------------------------


def _sig_stars(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _p_label(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


# ---------------------------------------------------------------------------
# IQR
# ---------------------------------------------------------------------------


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = float(np.quantile(sorted_v, 0.25))
    q3 = float(np.quantile(sorted_v, 0.75))
    return q3 - q1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to results JSON (default: benchmarks/results_imagewoof_v2.json)",
    )
    parser.add_argument(
        "--regime",
        default="scratch",
        help="Filter by regime (default: scratch)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output markdown table",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap CI (faster, useful for quick inspection)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10_000,
        help="Bootstrap resampling iterations (default 10000)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"Missing {args.results}.\n"
            "Run: python benchmarks/run_imagewoof_v2.py --seeds 42,43,44,45,46,47,48,49,50,51 --device cuda"
        )

    data = json.loads(args.results.read_text(encoding="utf-8"))
    all_runs = data.get("runs") or []

    # Filter: valid runs, matching regime
    runs = [
        r for r in all_runs
        if "val_metric" in r
        and "error" not in r
        and r.get("regime", "scratch") == args.regime
    ]
    if not runs:
        raise SystemExit(
            f"No valid runs for regime={args.regime!r} in {args.results}.\n"
            f"Total run records: {len(all_runs)}"
        )

    # Group by condition, keyed by seed for paired tests
    by_cond: dict[str, dict[int, float]] = defaultdict(dict)
    gpu_epochs_by_cond: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        cid = r["condition"]
        seed = int(r["seed"])
        metric = float(r.get("held_out_test_metric") or r.get("val_metric") or 0.0)
        by_cond[cid][seed] = metric
        ep = r.get("total_gpu_epochs")
        if ep is not None:
            gpu_epochs_by_cond[cid].append(int(ep))

    # Print header
    print(f"\nBenchmark: {data.get('benchmark_id', 'unknown')}  |  regime={args.regime}")
    hw = data.get("hardware") or {}
    git = str(data.get("git_head") or "")[:8]
    doc_budget = data.get("budget_epochs_total") or data.get("epochs_per_phase")
    print(f"Git: {git or '—'}  GPU: {hw.get('gpu', '—')}  "
          f"PyTorch: {hw.get('pytorch_version', '—')}  budget={doc_budget}\n")

    # Aggregate per condition
    # We need the bnnr_xai values aligned per seed for paired tests
    bnnr_xai_by_seed = by_cond.get("bnnr_xai", {})
    shared_seeds_per_cmp: dict[str, list[int]] = {}
    for cid in COMPARISON_CONDITIONS:
        other_by_seed = by_cond.get(cid, {})
        shared = sorted(set(bnnr_xai_by_seed.keys()) & set(other_by_seed.keys()))
        shared_seeds_per_cmp[cid] = shared

    # Run Wilcoxon tests for bnnr_xai vs each comparison condition
    wilcoxon_results: dict[str, dict[str, Any]] = {}
    for cid in COMPARISON_CONDITIONS:
        shared = shared_seeds_per_cmp[cid]
        if len(shared) < 2:
            wilcoxon_results[cid] = {"W": None, "p": None, "r": None, "n": len(shared)}
            continue
        bx = [bnnr_xai_by_seed[s] for s in shared]
        ot = [by_cond[cid][s] for s in shared]
        wres = _wilcoxon_signed_rank(bx, ot)
        if wres is None:
            wilcoxon_results[cid] = {"W": None, "p": None, "r": None, "n": len(shared)}
        else:
            W, p, r = wres
            # Bootstrap CI for median diff
            if not args.no_bootstrap:
                ci_lo, ci_hi = _bootstrap_median_diff_ci(
                    bx, ot, n_boot=args.n_boot, rng_seed=42
                )
            else:
                ci_lo, ci_hi = float("nan"), float("nan")
            wilcoxon_results[cid] = {
                "W": W, "p": p, "r": r, "n": len(shared),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "bnnr_xai_vals": bx,
                "other_vals": ot,
            }

    # Holm-Bonferroni correction on the 4 tests
    p_raw = [
        (wilcoxon_results[cid].get("p") or 1.0) for cid in COMPARISON_CONDITIONS
    ]
    p_holm = _holm_bonferroni(p_raw)
    for i, cid in enumerate(COMPARISON_CONDITIONS):
        wilcoxon_results[cid]["p_holm"] = p_holm[i]

    # ---- Main summary table ----
    # Columns: Condition | median | ±IQR | mean±std | n | Δ vs no_aug | p(Holm) | r | GPU-epochs
    no_aug_vals = list(by_cond.get("no_aug", {}).values())
    ref_median = statistics.median(no_aug_vals) if no_aug_vals else None

    rows = []
    for cid in ALL_CONDITIONS:
        if cid not in by_cond:
            continue
        vals = list(by_cond[cid].values())
        if not vals:
            continue
        med = statistics.median(vals)
        iqr_v = _iqr(vals)
        mn = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        n = len(vals)

        delta_s = "—"
        if ref_median is not None and cid != "no_aug":
            delta_s = f"{(med - ref_median) * 100:+.2f}pp"

        # Wilcoxon vs bnnr_xai (only for comparison conditions in the table)
        if cid in wilcoxon_results:
            wr = wilcoxon_results[cid]
            p_adj = wr.get("p_holm")
            r_v = wr.get("r")
            p_adj_s = f"{_p_label(p_adj)} {_sig_stars(p_adj)}" if p_adj is not None else "—"
            r_s = f"{r_v:.2f}" if r_v is not None else "—"
            ci_lo = wr.get("ci_lo", float("nan"))
            ci_hi = wr.get("ci_hi", float("nan"))
            ci_s = (
                f"[{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]pp"
                if not (math.isnan(ci_lo) or math.isnan(ci_hi))
                else "—"
            )
        elif cid == "bnnr_xai":
            p_adj_s = "—"
            r_s = "—"
            ci_s = "—"
        else:
            p_adj_s = "—"
            r_s = "—"
            ci_s = "—"

        gpu_epochs_list = gpu_epochs_by_cond.get(cid, [])
        gpu_ep_s = str(int(statistics.median(gpu_epochs_list))) if gpu_epochs_list else "?"

        rows.append({
            "cid": cid,
            "label": DISPLAY_NAMES.get(cid, cid),
            "median": med,
            "iqr": iqr_v,
            "mean": mn,
            "std": std,
            "n": n,
            "delta": delta_s,
            "p_holm": p_adj_s,
            "r": r_s,
            "ci": ci_s,
            "gpu_epochs": gpu_ep_s,
            "per_seed": ", ".join(f"{v*100:.2f}%" for v in sorted(vals)),
        })

    # Print
    if args.markdown:
        _print_markdown_table(rows)
    else:
        _print_text_table(rows)

    # ---- Key comparison: bnnr_xai vs bnnr_random ----
    _print_key_comparison(wilcoxon_results, by_cond, bnnr_xai_by_seed, args)

    # ---- Footer ----
    print(
        "\nStatistical notes:"
        "\n  Tests: Wilcoxon signed-rank (paired, two-sided) — scipy if available, "
        "else exact enumeration (n<=15) or normal approx."
        "\n  Multiple testing: Holm-Bonferroni correction over 4 simultaneous tests "
        f"(bnnr_xai vs {', '.join(COMPARISON_CONDITIONS)})."
        "\n  Bootstrap CI: 10 000 resamplings of paired (bnnr_xai - baseline) median diff."
        "\n  Effect size r: rank-biserial correlation = 1 - 2W/(n*(n+1))."
        "\n  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05."
        "\n  'p(Holm)' in table: Holm-adjusted p-value for bnnr_xai vs that condition."
        "\n  For bnnr_xai and bnnr_random rows: p(Holm) is shown in the Key Comparison section."
    )


def _print_markdown_table(rows: list[dict[str, Any]]) -> None:
    print(
        "| Condition | Test acc median | ±IQR | mean±std | n | "
        "Δ vs no_aug | p (Holm) vs bnnr_xai | r | Bootstrap 95% CI | GPU-epochs |"
    )
    print(
        "|-----------|----------------|------|----------|---|"
        "------------|---------------------|---|-----------------|-----------|"
    )
    for row in rows:
        med_s = f"{row['median']*100:.2f}%"
        iqr_s = f"±{row['iqr']*100:.2f}pp"
        mn_s = f"{row['mean']*100:.2f}% ±{row['std']*100:.2f}"
        print(
            f"| {row['label']} "
            f"| {med_s} "
            f"| {iqr_s} "
            f"| {mn_s} "
            f"| {row['n']} "
            f"| {row['delta']} "
            f"| {row['p_holm']} "
            f"| {row['r']} "
            f"| {row['ci']} "
            f"| {row['gpu_epochs']} |"
        )


def _print_text_table(rows: list[dict[str, Any]]) -> None:
    w = 40
    header = (
        f"{'Condition':<{w}} {'Median':>8} {'±IQR':>8} {'Mean':>8} {'±Std':>7} "
        f"{'n':>3} {'Δ vs no_aug':>12} {'p(Holm)':>18} {'r':>6} {'GPU-ep':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        med_s = f"{row['median']*100:.2f}%"
        iqr_s = f"±{row['iqr']*100:.2f}"
        mn_s = f"{row['mean']*100:.2f}%"
        std_s = f"±{row['std']*100:.2f}"
        print(
            f"{row['label']:<{w}} {med_s:>8} {iqr_s:>8} {mn_s:>8} {std_s:>7} "
            f"{row['n']:>3} {row['delta']:>12} {row['p_holm']:>18} {row['r']:>6} "
            f"{row['gpu_epochs']:>8}"
        )
        print(f"  per-seed: {row['per_seed']}")
    print()


def _print_key_comparison(
    wilcoxon_results: dict[str, dict[str, Any]],
    by_cond: dict[str, dict[int, float]],
    bnnr_xai_by_seed: dict[int, float],
    args: argparse.Namespace,
) -> None:
    """Print a focused section on the bnnr_xai vs bnnr_random comparison."""
    print("\n" + "=" * 70)
    print("  KEY COMPARISON: bnnr_xai vs bnnr_random")
    print("  (isolates the contribution of XAI-guided selection)")
    print("=" * 70)

    wr = wilcoxon_results.get("bnnr_random")
    if wr is None or wr.get("W") is None:
        print("  Not enough data for bnnr_xai vs bnnr_random comparison.")
        return

    shared = sorted(
        set(bnnr_xai_by_seed.keys()) & set(by_cond.get("bnnr_random", {}).keys())
    )
    bx = [bnnr_xai_by_seed[s] for s in shared]
    br = [by_cond["bnnr_random"][s] for s in shared]

    med_xai = statistics.median(bx)
    med_rand = statistics.median(br)
    delta = (med_xai - med_rand) * 100

    p_raw = wr["p"]
    p_holm = wr["p_holm"]
    r_v = wr["r"]
    W = wr["W"]
    n = wr["n"]

    ci_lo = wr.get("ci_lo", float("nan"))
    ci_hi = wr.get("ci_hi", float("nan"))

    print(f"  n shared seeds: {n}")
    print(f"  bnnr_xai  median held_out_test accuracy: {med_xai*100:.2f}%")
    print(f"  bnnr_random median held_out_test accuracy: {med_rand*100:.2f}%")
    print(f"  Δ (xai - random): {delta:+.2f}pp")
    if not (math.isnan(ci_lo) or math.isnan(ci_hi)):
        print(f"  Bootstrap 95% CI for Δ median: [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]pp")
    print(f"  Wilcoxon W = {W:.1f},  p (raw) = {_p_label(p_raw)},  "
          f"p (Holm-adjusted) = {_p_label(p_holm)}  {_sig_stars(p_holm)}")
    print(f"  Effect size r (rank-biserial) = {r_v:.3f}")
    print(f"  Per-seed xai:  {', '.join(f'{v*100:.2f}%' for v in bx)}")
    print(f"  Per-seed rand: {', '.join(f'{v*100:.2f}%' for v in br)}")

    # Interpret effect size
    if abs(r_v) >= 0.5:
        effect_label = "large"
    elif abs(r_v) >= 0.3:
        effect_label = "medium"
    elif abs(r_v) >= 0.1:
        effect_label = "small"
    else:
        effect_label = "negligible"

    print(f"\n  Interpretation: effect size is {effect_label} (|r|={abs(r_v):.3f}).")
    if p_holm < 0.05:
        print(
            f"  The XAI-guided selection (bnnr_xai) is significantly better than "
            f"random selection (bnnr_random) at alpha=0.05 (Holm-corrected)."
        )
    else:
        print(
            f"  The XAI-guided selection (bnnr_xai) is NOT significantly better than "
            f"random selection (bnnr_random) at alpha=0.05 (Holm-corrected). "
            f"Consider collecting more seeds."
        )
    print()

    # Also show the full bnnr_xai vs baselines comparison in text
    print("=" * 70)
    print("  BNNR XAI vs ALL COMPARISON CONDITIONS (Wilcoxon, Holm-corrected)")
    print("=" * 70)
    for cid in COMPARISON_CONDITIONS:
        wr2 = wilcoxon_results.get(cid, {})
        if wr2.get("W") is None:
            print(f"  vs {DISPLAY_NAMES.get(cid, cid)}: insufficient data (n={wr2.get('n', 0)})")
            continue
        shared2 = wr2.get("n", 0)
        bx2 = wr2.get("bnnr_xai_vals", [])
        ot2 = wr2.get("other_vals", [])
        med_b = statistics.median(bx2) if bx2 else float("nan")
        med_o = statistics.median(ot2) if ot2 else float("nan")
        d2 = (med_b - med_o) * 100
        p2 = wr2.get("p_holm")
        r2 = wr2.get("r")
        ci_lo2 = wr2.get("ci_lo", float("nan"))
        ci_hi2 = wr2.get("ci_hi", float("nan"))
        ci_s2 = (
            f"[{ci_lo2*100:+.2f}, {ci_hi2*100:+.2f}]pp"
            if not (math.isnan(ci_lo2) or math.isnan(ci_hi2))
            else "—"
        )
        print(
            f"  vs {DISPLAY_NAMES.get(cid, cid):<42} "
            f"Δ={d2:+.2f}pp  n={shared2}  "
            f"{_p_label(p2)} {_sig_stars(p2)}  r={r2:.2f}  CI={ci_s2}"
        )
    print()


if __name__ == "__main__":
    main()
