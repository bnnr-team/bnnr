#!/usr/bin/env python3
"""Cross-dataset analysis for the grand BNNR benchmark.

Loads all ``results_*.json`` files from a directory and produces:
  1. Per-dataset table (conditions × metrics, Wilcoxon + Holm-Bonferroni)
  2. Cross-dataset summary table (median accuracy per dataset, mean Δ vs no_aug)
  3. XAI correlation section (Spearman ρ between edge_ratio and accuracy)
  4. Key comparison section (bnnr_xai vs bnnr_random, per dataset, with p-values)
  5. Compute transparency (total and deployed epochs per condition)
  6. Which fill is best (fill-using methods ranked across strategies)

Examples
--------
    python benchmarks/summarize_grand.py --results-dir benchmarks/ --markdown

    python benchmarks/summarize_grand.py \\
        --results-dir benchmarks/ --datasets imagewoof,pets --regime scratch
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

BENCHMARKS_DIR = Path(__file__).resolve().parent

ALL_DATASETS = ["imagewoof", "pets", "flowers102", "dtd", "aircraft", "eurosat"]

ALL_CONDITIONS_ORDERED = [
    "no_aug",
    "randaugment",
    "trivialaugment",
    "autoaugment",
    "churchnoise_only",
    "icd_only",
    "aicd_only",
    "icd_aicd_fixed",
    "bnnr_random",
    "bnnr_xai",
]

DISPLAY_NAMES = {
    "no_aug":          "No augmentation (crop + flip)",
    "randaugment":     "RandAugment (torchvision)",
    "trivialaugment":  "TrivialAugmentWide (torchvision)",
    "autoaugment":     "AutoAugment (ImageNet policy)",
    "churchnoise_only":"ChurchNoise only (non-XAI ablation)",
    "icd_only":        "ICD only",
    "aicd_only":       "AICD only",
    "icd_aicd_fixed":  "ICD+AICD fixed (no search)",
    "bnnr_xai":        "BNNR XAI-guided (equal compute)",
    "bnnr_random":     "BNNR random selection (XAI ablation)",
}

# Conditions compared against bnnr_xai in significance tests
COMPARISON_CONDITIONS = [
    "no_aug", "randaugment", "trivialaugment", "autoaugment",
    "churchnoise_only", "icd_only", "aicd_only", "icd_aicd_fixed", "bnnr_random",
]

FILL_STRATEGY_ORDER = ["gaussian_blur", "local_mean", "global_mean", "noise", "solid"]
FILL_USING_CONDITIONS = [
    "icd_only", "aicd_only", "icd_aicd_fixed", "bnnr_random", "bnnr_xai",
]



# ---------------------------------------------------------------------------
# Statistical helpers (from summarize_v2.py — self-contained copy)
# ---------------------------------------------------------------------------


class WilcoxonResult(NamedTuple):
    W_statistic: float
    p_value: float
    rank_biserial_r: float


def _wilcoxon_signed_rank(
    x: list[float], y: list[float]
) -> WilcoxonResult | None:
    """Paired Wilcoxon signed-rank test (two-sided).

    Returns (W_statistic, p_value, rank_biserial_r) or None if n < 2.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal length for paired test")
    if len(x) < 2:
        return None

    diffs = [float(xi) - float(yi) for xi, yi in zip(x, y)]

    try:
        from scipy.stats import wilcoxon as _scipy_wilcoxon

        nonzero = [d for d in diffs if d != 0.0]
        if len(nonzero) == 0:
            return WilcoxonResult(0.0, 1.0, 0.0)
        result = _scipy_wilcoxon(nonzero, alternative="two-sided")
        W = float(result.statistic)
        p = float(result.pvalue)
        n = len(nonzero)
        r = 1.0 - 2.0 * W / (n * (n + 1)) if n > 0 else 0.0
        return WilcoxonResult(W, p, r)
    except ImportError:
        pass

    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0.0

    abs_d = [abs(d) for d in nonzero]
    order = sorted(range(n), key=lambda i: abs_d[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    W_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)
    W_minus = sum(ranks[i] for i in range(n) if nonzero[i] < 0)
    W = min(W_plus, W_minus)

    if n <= 15:
        p = _wilcoxon_exact_p(n, W)
    else:
        mu = n * (n + 1) / 4.0
        sigma2 = n * (n + 1) * (2 * n + 1) / 24.0
        if sigma2 == 0:
            p = 1.0
        else:
            z = (W - mu + 0.5) / math.sqrt(sigma2)
            p = _normal_two_sided_p(abs(z))

    r = 1.0 - 2.0 * W / (n * (n + 1))
    return W, min(1.0, p), r


def _wilcoxon_exact_p(n: int, W: float) -> float:
    total = 1 << n
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
    return float(math.erfc(z / math.sqrt(2.0)))


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, running_max)
    return adjusted


def _bootstrap_median_diff_ci(
    x: list[float],
    y: list[float],
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Return a bootstrap CI for the median difference (median(x) - median(y)).

    Computational cost scales linearly with ``n_boot`` resamples. The default
    (10,000 iterations) favors stability, but can be expensive when many
    dataset/condition comparisons are evaluated. Callers can lower ``n_boot``
    to speed up large benchmark runs.
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


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.quantile(values, 0.75)) - float(np.quantile(values, 0.25))


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


def _spearman_r(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation.

    We require at least 3 paired samples before reporting rho. With only 1-2
    points, the rank-based correlation is either undefined or trivially
    extreme/unstable, which is not informative for the benchmark summary.
    Returning ``None`` marks the result as insufficient data.
    """
    if len(x) < 3:
        return None
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)

    def _rank(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    rx = _rank(xa)
    ry = _rank(ya)
    rx_m = rx - rx.mean()
    ry_m = ry - ry.mean()
    denom = math.sqrt((rx_m * rx_m).sum() * (ry_m * ry_m).sum())
    if denom < 1e-12:
        return 0.0
    return float((rx_m * ry_m).sum() / denom)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_all_runs(results_dir: Path, regime_filter: str) -> list[dict[str, Any]]:
    """Scan *results_dir* for ``results_*.json`` and return valid run records."""
    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("results_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"  Warning: could not read {path}")
            continue
        for r in data.get("runs") or []:
            if "val_metric" not in r or "error" in r:
                continue
            run_regime = r.get("regime", "scratch")
            if regime_filter != "all" and run_regime != regime_filter:
                continue
            r.setdefault("fill_strategy", "none")
            runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------


def _analyze_dataset(
    dataset: str,
    runs: list[dict[str, Any]],
    *,
    no_bootstrap: bool,
    n_boot: int,
    markdown: bool,
    strategy: str = "none",
) -> None:
    """Print per-dataset condition table with statistics."""
    ds_runs = [r for r in runs if r.get("dataset") == dataset]
    if not ds_runs:
        print(f"\n  [No data for {dataset}]\n")
        return

    # Group by condition → seed → metric
    by_cond: dict[str, dict[int, float]] = defaultdict(dict)
    gpu_epochs_by_cond: dict[str, list[int]] = defaultdict(list)
    deployed_epochs_by_cond: dict[str, list[int]] = defaultdict(list)
    ece_by_cond: dict[str, list[float]] = defaultdict(list)
    for r in ds_runs:
        cid = r["condition"]
        seed = int(r["seed"])
        metric = float(r.get("held_out_test_metric") or r.get("val_metric") or 0.0)
        by_cond[cid][seed] = metric
        ep = r.get("total_gpu_epochs")
        if ep is not None:
            gpu_epochs_by_cond[cid].append(int(ep))
        # Absent in records written before FIX-3-1. Those rows print "?" rather
        # than crashing the summarizer, which is the point of reading with .get.
        dep = r.get("deployed_epochs")
        if dep is not None:
            deployed_epochs_by_cond[cid].append(int(dep))
        # ECE was recorded from the start and never printed. On Imagewoof the
        # most accurate conditions are the worst calibrated by roughly an order
        # of magnitude, which is a headline fact rather than a JSON field.
        ece = r.get("test_ece")
        if ece is not None:
            ece_by_cond[cid].append(float(ece))

    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset.upper()}  |  fill={strategy}")
    print(f"  Conditions found: {', '.join(sorted(by_cond))}")
    print(f"{'='*70}")

    # Wilcoxon tests: bnnr_xai vs each comparison condition
    bnnr_xai_by_seed = by_cond.get("bnnr_xai", {})
    wilcoxon_results: dict[str, dict[str, Any]] = {}
    comp_conds_present = [c for c in COMPARISON_CONDITIONS if c in by_cond]

    for cid in comp_conds_present:
        other_by_seed = by_cond.get(cid, {})
        shared = sorted(set(bnnr_xai_by_seed.keys()) & set(other_by_seed.keys()))
        if len(shared) < 2:
            wilcoxon_results[cid] = {"W": None, "p": None, "r": None, "n": len(shared)}
            continue
        bx = [bnnr_xai_by_seed[s] for s in shared]
        ot = [other_by_seed[s] for s in shared]
        wres = _wilcoxon_signed_rank(bx, ot)
        if wres is None:
            wilcoxon_results[cid] = {"W": None, "p": None, "r": None, "n": len(shared)}
        else:
            W, p, r = wres
            ci_lo, ci_hi = (float("nan"), float("nan"))
            if not no_bootstrap and len(shared) >= 2:
                ci_lo, ci_hi = _bootstrap_median_diff_ci(bx, ot, n_boot=n_boot, rng_seed=42)
            wilcoxon_results[cid] = {
                "W": W, "p": p, "r": r, "n": len(shared),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "bnnr_xai_vals": bx,
                "other_vals": ot,
            }

    # Holm-Bonferroni correction
    if comp_conds_present:
        p_raw = [(wilcoxon_results[c].get("p") or 1.0) for c in comp_conds_present]
        p_holm = _holm_bonferroni(p_raw)
        for i, cid in enumerate(comp_conds_present):
            wilcoxon_results[cid]["p_holm"] = p_holm[i]

    # Reference median for delta
    no_aug_vals = list(by_cond.get("no_aug", {}).values())
    ref_median = statistics.median(no_aug_vals) if no_aug_vals else None

    # Build rows
    rows = []
    conds_to_show = [c for c in ALL_CONDITIONS_ORDERED if c in by_cond]
    for cid in conds_to_show:
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

        wr = wilcoxon_results.get(cid, {})
        p_adj = wr.get("p_holm")
        r_v = wr.get("r")
        p_adj_s = f"{_p_label(p_adj)} {_sig_stars(p_adj)}" if p_adj is not None else "—"
        r_s = f"{r_v:.2f}" if r_v is not None else "—"
        ci_lo_v = wr.get("ci_lo", float("nan"))
        ci_hi_v = wr.get("ci_hi", float("nan"))
        ci_s = (
            f"[{ci_lo_v*100:+.2f}, {ci_hi_v*100:+.2f}]pp"
            if not (math.isnan(ci_lo_v) or math.isnan(ci_hi_v))
            else "—"
        )

        gpu_epochs_list = gpu_epochs_by_cond.get(cid, [])
        gpu_ep_s = str(int(statistics.median(gpu_epochs_list))) if gpu_epochs_list else "?"
        deployed_list = deployed_epochs_by_cond.get(cid, [])
        dep_ep_s = str(int(statistics.median(deployed_list))) if deployed_list else "?"
        ece_list = ece_by_cond.get(cid, [])
        ece_s = f"{statistics.median(ece_list):.3f}" if ece_list else "?"

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
            "deployed_epochs": dep_ep_s,
            "ece": ece_s,
            "per_seed": ", ".join(f"{v*100:.2f}%" for v in sorted(vals)),
        })

    if markdown:
        _print_markdown_table(rows)
    else:
        _print_text_table(rows)

    # Key comparison: bnnr_xai vs bnnr_random
    _print_xai_vs_random(wilcoxon_results, by_cond, bnnr_xai_by_seed)


def _print_text_table(rows: list[dict[str, Any]]) -> None:
    w = 38
    header = (
        f"{'Condition':<{w}} {'Median':>8} {'±IQR':>8} {'Mean':>8} {'±Std':>7} "
        f"{'n':>3} {'Δ vs no_aug':>12} {'p(Holm)':>18} {'r':>6} {'ECE':>7} "
        f"{'GPU-ep':>8} {'Depl-ep':>8}"
    )
    # Say what the ordering means. Accuracy and calibration reverse the ranking
    # on this data, so a table sorted by one and read as "best" is misleading.
    print("  ranked by: median held-out accuracy (higher better). "
          "ECE is calibration error (LOWER better) and does not follow it.")
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
            f"{row['ece']:>7} {row['gpu_epochs']:>8} {row['deployed_epochs']:>8}"
        )
        print(f"  per-seed: {row['per_seed']}")
    print()


def _print_markdown_table(rows: list[dict[str, Any]]) -> None:
    print(
        "Ranked by median held-out accuracy (higher better). "
        "ECE is calibration error, where **lower is better**, and it does not "
        "follow the accuracy ranking: on Imagewoof the most accurate conditions "
        "are the worst calibrated by roughly an order of magnitude.\n"
    )
    print(
        "| Condition | Median | ±IQR | mean±std | n | "
        "Δ vs no_aug | p (Holm) vs bnnr_xai | r | Bootstrap 95% CI | "
        "ECE ↓ | GPU-epochs | Deployed epochs |"
    )
    print(
        "|-----------|--------|------|----------|---|"
        "------------|---------------------|---|-----------------|"
        "-------|-----------|-----------------|"
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
            f"| {row['ece']} | {row['gpu_epochs']} | {row['deployed_epochs']} |"
        )


def _print_xai_vs_random(
    wilcoxon_results: dict[str, dict[str, Any]],
    by_cond: dict[str, dict[int, float]],
    bnnr_xai_by_seed: dict[int, float],
) -> None:
    wr = wilcoxon_results.get("bnnr_random")
    if wr is None or wr.get("W") is None:
        print("  bnnr_xai vs bnnr_random: insufficient data.\n")
        return

    shared = sorted(
        set(bnnr_xai_by_seed.keys()) & set(by_cond.get("bnnr_random", {}).keys())
    )
    bx = [bnnr_xai_by_seed[s] for s in shared]
    br = [by_cond["bnnr_random"][s] for s in shared]
    med_xai = statistics.median(bx)
    med_rand = statistics.median(br)
    delta = (med_xai - med_rand) * 100

    p_holm = wr.get("p_holm", wr.get("p"))
    r_v = wr.get("r")
    ci_lo = wr.get("ci_lo", float("nan"))
    ci_hi = wr.get("ci_hi", float("nan"))

    print(f"  KEY: bnnr_xai ({med_xai*100:.2f}%) vs bnnr_random ({med_rand*100:.2f}%)  "
          f"Δ={delta:+.2f}pp  n={len(shared)}  "
          f"{_p_label(p_holm)} {_sig_stars(p_holm)}  r={r_v:.2f}")
    if not (math.isnan(ci_lo) or math.isnan(ci_hi)):
        print(f"  Bootstrap 95% CI for Δ median: [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]pp")
    print()


# ---------------------------------------------------------------------------
# Cross-dataset summary table
# ---------------------------------------------------------------------------


def _cross_dataset_table(
    all_runs: list[dict[str, Any]],
    datasets: list[str],
    *,
    markdown: bool,
    strategy: str = "none",
) -> None:
    """Print a condition × dataset accuracy matrix with mean Δ."""
    print("\n" + "=" * 80)
    print(f"  CROSS-DATASET SUMMARY  (median held_out_test accuracy)  |  fill={strategy}")
    print("=" * 80)

    # Collect medians: by_cond_ds[cond][dataset] = median accuracy
    by_cond_ds: dict[str, dict[str, float | None]] = defaultdict(lambda: {ds: None for ds in datasets})
    no_aug_medians: dict[str, float] = {}

    for ds in datasets:
        ds_runs = [r for r in all_runs if r.get("dataset") == ds]
        for cid in ALL_CONDITIONS_ORDERED:
            vals = [
                float(r.get("held_out_test_metric") or r.get("val_metric") or 0.0)
                for r in ds_runs if r["condition"] == cid
            ]
            if vals:
                by_cond_ds[cid][ds] = statistics.median(vals)
        no_aug_vals = [
            float(r.get("held_out_test_metric") or r.get("val_metric") or 0.0)
            for r in ds_runs if r["condition"] == "no_aug"
        ]
        if no_aug_vals:
            no_aug_medians[ds] = statistics.median(no_aug_vals)

    conds_with_data = [
        c for c in ALL_CONDITIONS_ORDERED
        if any(by_cond_ds[c].get(ds) is not None for ds in datasets)
    ]

    if markdown:
        header = "| Condition | " + " | ".join(datasets) + " | Mean Δ |"
        sep = "|-----------|" + "|".join(["-------"] * len(datasets)) + "|--------|"
        print(header)
        print(sep)
        for cid in conds_with_data:
            cells: list[str] = []
            deltas: list[float] = []
            for ds in datasets:
                med = by_cond_ds[cid].get(ds)
                if med is None:
                    cells.append("—")
                else:
                    ref = no_aug_medians.get(ds)
                    if ref is not None and cid != "no_aug":
                        d = (med - ref) * 100
                        deltas.append(d)
                        cells.append(f"{med*100:.2f}% ({d:+.1f}pp)")
                    else:
                        cells.append(f"{med*100:.2f}%")
            mean_delta_s = f"{statistics.mean(deltas):+.2f}pp" if deltas else "—"
            print(f"| {DISPLAY_NAMES.get(cid, cid)} | " + " | ".join(cells) + f" | {mean_delta_s} |")
    else:
        col_w = 16
        header_parts = ["Condition".ljust(38)]
        for ds in datasets:
            header_parts.append(ds[:col_w].center(col_w + 2))
        header_parts.append("Mean Δ".rjust(10))
        header_line = " ".join(header_parts)
        print(header_line)
        print("-" * len(header_line))
        for cid in conds_with_data:
            row_parts = [DISPLAY_NAMES.get(cid, cid)[:38].ljust(38)]
            deltas: list[float] = []
            for ds in datasets:
                med = by_cond_ds[cid].get(ds)
                if med is None:
                    row_parts.append("  —".center(col_w + 2))
                else:
                    ref = no_aug_medians.get(ds)
                    if ref is not None and cid != "no_aug":
                        d = (med - ref) * 100
                        deltas.append(d)
                        row_parts.append(f"{med*100:.1f}%({d:+.1f})".center(col_w + 2))
                    else:
                        row_parts.append(f"{med*100:.1f}%".center(col_w + 2))
            mean_delta_s = f"{statistics.mean(deltas):+.2f}pp" if deltas else "—"
            row_parts.append(mean_delta_s.rjust(10))
            print(" ".join(row_parts))
    print()


# ---------------------------------------------------------------------------
# XAI correlation section
# ---------------------------------------------------------------------------


def _xai_correlation_section(
    all_runs: list[dict[str, Any]],
    datasets: list[str],
    *,
    markdown: bool,
    strategy: str = "none",
) -> None:
    """Print Spearman ρ between xai_edge_ratio and test_accuracy per dataset."""
    print("\n" + "=" * 70)
    print(f"  XAI ANALYSIS: edge_ratio vs accuracy correlations  |  fill={strategy}")
    print("=" * 70)

    for ds in datasets:
        ds_runs = [
            r for r in all_runs
            if r.get("dataset") == ds
            and r.get("xai_edge_ratio") is not None
            and (r.get("test_accuracy") or r.get("held_out_test_metric")) is not None
        ]
        if not ds_runs:
            print(f"  {ds}: no XAI data")
            continue

        edge_ratios = [float(r["xai_edge_ratio"]) for r in ds_runs]
        accuracies = [
            float(r.get("test_accuracy") or r.get("held_out_test_metric") or 0.0)
            for r in ds_runs
        ]
        rho = _spearman_r(edge_ratios, accuracies)
        rho_s = f"ρ={rho:.3f}" if rho is not None else "ρ=n/a"

        # Mean edge_ratio per condition
        by_cond: dict[str, list[float]] = defaultdict(list)
        for r in ds_runs:
            by_cond[r["condition"]].append(float(r["xai_edge_ratio"]))

        cond_means = {
            c: float(np.mean(v)) for c, v in by_cond.items() if v
        }
        cond_strs = [
            f"{c}={cond_means[c]:.3f}"
            for c in ALL_CONDITIONS_ORDERED if c in cond_means
        ]
        print(f"  {ds}: {rho_s} (n={len(ds_runs)}) | edge_ratio by condition: {', '.join(cond_strs)}")

    print()


# ---------------------------------------------------------------------------
# Key comparison: bnnr_xai vs bnnr_random per dataset
# ---------------------------------------------------------------------------


def _key_comparison_section(
    all_runs: list[dict[str, Any]],
    datasets: list[str],
    *,
    no_bootstrap: bool,
    n_boot: int,
    markdown: bool,
    strategy: str = "none",
) -> None:
    """bnnr_xai vs bnnr_random, per dataset, with Wilcoxon p-values."""
    print("\n" + "=" * 70)
    print(f"  KEY COMPARISON: bnnr_xai vs bnnr_random (per dataset)  |  fill={strategy}")
    print("  (Isolates the contribution of XAI-guided selection)")
    print("=" * 70)

    rows = []
    for ds in datasets:
        ds_runs = [r for r in all_runs if r.get("dataset") == ds]

        xai_by_seed: dict[int, float] = {}
        rand_by_seed: dict[int, float] = {}
        for r in ds_runs:
            seed = int(r["seed"])
            metric = float(r.get("held_out_test_metric") or r.get("val_metric") or 0.0)
            if r["condition"] == "bnnr_xai":
                xai_by_seed[seed] = metric
            elif r["condition"] == "bnnr_random":
                rand_by_seed[seed] = metric

        shared = sorted(set(xai_by_seed) & set(rand_by_seed))
        if len(shared) < 2:
            rows.append({
                "dataset": ds,
                "n": len(shared),
                "med_xai": None, "med_rand": None,
                "delta": None, "p": None, "r": None, "stars": "—",
                "ci_lo": float("nan"), "ci_hi": float("nan"),
            })
            continue

        bx = [xai_by_seed[s] for s in shared]
        br = [rand_by_seed[s] for s in shared]
        med_xai = statistics.median(bx)
        med_rand = statistics.median(br)
        delta = (med_xai - med_rand) * 100

        wres = _wilcoxon_signed_rank(bx, br)
        ci_lo, ci_hi = float("nan"), float("nan")
        if wres is not None and not no_bootstrap:
            ci_lo, ci_hi = _bootstrap_median_diff_ci(bx, br, n_boot=n_boot, rng_seed=42)
        W, p, r = wres if wres is not None else (None, None, None)
        rows.append({
            "dataset": ds,
            "n": len(shared),
            "med_xai": med_xai, "med_rand": med_rand,
            "delta": delta, "p": p, "r": r,
            "stars": _sig_stars(p),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })

    if markdown:
        print("| Dataset | n | bnnr_xai | bnnr_random | Δ | p | stars | r | Bootstrap 95% CI |")
        print("|---------|---|----------|-------------|---|---|-------|---|-----------------|")
        for row in rows:
            if row["med_xai"] is None:
                print(f"| {row['dataset']} | {row['n']} | — | — | — | — | — | — | — |")
                continue
            ci_s = (
                f"[{row['ci_lo']*100:+.2f}, {row['ci_hi']*100:+.2f}]pp"
                if not (math.isnan(row["ci_lo"]) or math.isnan(row["ci_hi"]))
                else "—"
            )
            print(
                f"| {row['dataset']} "
                f"| {row['n']} "
                f"| {row['med_xai']*100:.2f}% "
                f"| {row['med_rand']*100:.2f}% "
                f"| {row['delta']:+.2f}pp "
                f"| {_p_label(row['p'])} "
                f"| {row['stars']} "
                f"| {row['r']:.2f} "
                f"| {ci_s} |"
            )
    else:
        print(
            f"{'Dataset':<12} {'n':>3} {'bnnr_xai':>10} {'bnnr_random':>12} "
            f"{'Δ':>8} {'p':>10} {'sig':>5} {'r':>6}"
        )
        print("-" * 75)
        for row in rows:
            if row["med_xai"] is None:
                print(f"  {row['dataset']:<12} {row['n']:>3}  insufficient data")
                continue
            ci_s = ""
            if not (math.isnan(row["ci_lo"]) or math.isnan(row["ci_hi"])):
                ci_s = f"  CI=[{row['ci_lo']*100:+.2f},{row['ci_hi']*100:+.2f}]pp"
            print(
                f"  {row['dataset']:<12} {row['n']:>3} "
                f"{row['med_xai']*100:>9.2f}% "
                f"{row['med_rand']*100:>11.2f}% "
                f"{row['delta']:>+7.2f}pp "
                f"{_p_label(row['p']):>10} "
                f"{row['stars']:>5} "
                f"{row['r']:>5.2f}"
                f"{ci_s}"
            )
    print()


# ---------------------------------------------------------------------------
# Compute transparency
# ---------------------------------------------------------------------------


def _compute_transparency_section(
    all_runs: list[dict[str, Any]],
    datasets: list[str],
    *,
    strategy: str = "none",
) -> None:
    """Print median total and deployed epochs per condition.

    Both numbers, because they answer different questions and T20's headline
    deficit turned out to be the difference between them. total_gpu_epochs is
    what an equal-compute protocol matches; deployed_epochs is what the shipped
    model actually received. Printing only one is how the confound survived
    into the results the first time.
    """
    print("\n" + "=" * 70)
    print(f"  COMPUTE TRANSPARENCY: median epochs per condition  |  fill={strategy}")
    print("  total = every epoch trained;  deployed = epochs the shipped model got")
    print("=" * 70)
    for ds in datasets:
        ds_runs = [r for r in all_runs if r.get("dataset") == ds]
        if not ds_runs:
            continue
        total_by_cond: dict[str, list[int]] = defaultdict(list)
        deployed_by_cond: dict[str, list[int]] = defaultdict(list)
        for r in ds_runs:
            ep = r.get("total_gpu_epochs")
            if ep is not None:
                total_by_cond[r["condition"]].append(int(ep))
            dep = r.get("deployed_epochs")
            if dep is not None:
                deployed_by_cond[r["condition"]].append(int(dep))
        parts = []
        for c in ALL_CONDITIONS_ORDERED:
            if not total_by_cond.get(c):
                continue
            total = int(statistics.median(total_by_cond[c]))
            deployed_list = deployed_by_cond.get(c)
            deployed = int(statistics.median(deployed_list)) if deployed_list else None
            parts.append(f"{c}={total}/{deployed if deployed is not None else '?'}")
        if parts:
            print(f"  {ds}: {', '.join(parts)}  (total/deployed)")
    print()

# ---------------------------------------------------------------------------
# Which fill is best
# ---------------------------------------------------------------------------

def _meaningful_gate_reachable(n_present: int, n_shared: int) -> tuple[bool, float]:
    """Can the 'meaningful' verdict fire at all, given #strategies and #seeds?

    (n_present - 1) tests are run against the reference, so Holm multiplies the
    smallest raw p by (n_present - 1). The smallest a two-sided signed-rank p can
    be at n_shared paired seeds (all diffs same sign, no ties) is 2 / 2**n_shared.
    If even that best case times the Holm factor can't clear 0.05, no data can
    make the gate fire. Returns (reachable, holm_floor); floor is a lower bound,
    so an "unreachable" result here is definitive.
    """
    if n_shared < 2:
        return False, 1.0
    m = max(1, n_present - 1)
    min_raw_p = 2.0 ** (1 - n_shared)  # = 2 / 2**n_shared
    floor = min(1.0, min_raw_p * m)
    return floor < 0.05, floor

def _rank_strategies_for_method(
    by_strat: dict[str, dict[int, float]],
    strategies: list[str],
    *,
    no_bootstrap: bool,
    n_boot: int,
    reference: str | None = None,
) -> dict[str, Any] | None:
    """Rank fill strategies for one (dataset, method) cell on shared seeds.

    reference=None  -> empirical-best mode: the top strategy by median is the
                       reference; the verdict asks whether it is separated from
                       the runner-up (original behaviour).
    reference=<s>   -> fixed-reference mode: <s> (e.g. the library default) is the
                       reference; the verdict asks whether any strategy *beats* it
                       ("should we change the default?"). Falls back to empirical
                       best, with a flag, if <s> has no data in this cell.
    """
    present = [s for s in strategies if by_strat.get(s)]
    if len(present) < 2:
        return None

    shared = sorted(set.intersection(*(set(by_strat[s]) for s in present)))
    if len(shared) < 2:
        return {"insufficient": True, "present": present, "shared": shared}

    vecs = {s: [by_strat[s][seed] for seed in shared] for s in present}
    medians = {s: statistics.median(vecs[s]) for s in present}

    order_index = {s: i for i, s in enumerate(FILL_STRATEGY_ORDER)}
    ranked = sorted(present, key=lambda s: (-medians[s], order_index.get(s, len(order_index))))
    top = ranked[0]

    reference_missing = bool(reference) and reference not in present
    if reference and reference in present:
        ref, mode = reference, "reference"
    else:
        ref, mode = top, "best"
    ref_vec = vecs[ref]

    entries = []
    for s in ranked:
        if s == ref:
            entries.append({"strategy": s, "median": medians[s], "is_ref": True,
                            "delta": None, "p": None, "p_holm": None, "r": None,
                            "stars": "—", "ci_lo": float("nan"), "ci_hi": float("nan")})
            continue
        wres = _wilcoxon_signed_rank(vecs[s], ref_vec)
        W, p, r = wres if wres is not None else (None, None, None)
        ci_lo, ci_hi = float("nan"), float("nan")
        if wres is not None and not no_bootstrap:
            # Δ orientation: strategy - ref (positive => strategy is higher).
            ci_lo, ci_hi = _bootstrap_median_diff_ci(vecs[s], ref_vec, n_boot=n_boot, rng_seed=42)
        entries.append({"strategy": s, "median": medians[s], "is_ref": False,
                        "delta": (medians[s] - medians[ref]) * 100,
                        "p": p, "p_holm": None, "r": r, "stars": _sig_stars(p),
                        "ci_lo": ci_lo, "ci_hi": ci_hi})

    testable = [e for e in entries if not e["is_ref"] and e["p"] is not None]
    if testable:
        for e, p_adj in zip(testable, _holm_bonferroni([e["p"] for e in testable])):
            e["p_holm"] = p_adj
            e["stars"] = _sig_stars(p_adj)

    non_ref = [e for e in entries if not e["is_ref"]]
    challenger = non_ref[0] if non_ref else None  # highest-median non-reference strategy

    def _sig_and_ci(e):
        if e is None or e["p_holm"] is None:
            return False
        ci_excl = (not math.isnan(e["ci_lo"]) and not math.isnan(e["ci_hi"])
                   and not (e["ci_lo"] <= 0.0 <= e["ci_hi"]))
        return e["p_holm"] < 0.05 and (no_bootstrap or ci_excl)

    if mode == "reference":
        meaningful = (challenger is not None and challenger["median"] > medians[ref]
                      and _sig_and_ci(challenger))
    else:
        meaningful = _sig_and_ci(challenger)  # challenger == runner-up in best mode

    gate_reachable, holm_floor = _meaningful_gate_reachable(len(present), len(shared))

    return {"insufficient": False, "mode": mode, "reference": ref,
            "reference_requested": reference, "reference_missing": reference_missing,
            "reference_is_top": ref == top, "shared": shared, "best": top,
            "challenger": challenger["strategy"] if challenger else None,
            "runner_up": challenger["strategy"] if challenger else None,
            "entries": entries, "meaningful": meaningful,
            "gate_reachable": gate_reachable, "holm_floor": holm_floor}

def _which_fill_is_best_section(
    all_runs: list[dict[str, Any]],
    datasets: list[str],
    strategies: list[str],
    *,
    no_bootstrap: bool,
    n_boot: int,
    markdown: bool,
    reference: str | None = None,
) -> None:
    """Rank fill strategies head-to-head for each fill-using method (Phase 4).

    Runs once over the whole result set (not per-strategy view). For every
    dataset and fill-using method it compares strategies on shared seeds using
    the held-out split metric, ranks them, tests best-vs-each with Wilcoxon +
    bootstrap CI, and states whether the top gap is meaningful. Fill-independent
    methods are excluded — there is nothing to compare across fills.
    """
    print("\n" + "#" * 80)
    print("#  WHICH FILL IS BEST  (fill-using methods, held-out split)")
    print("#  Strategies compared on shared seeds, ranked by median held-out accuracy.")
    if reference:
        print(f"#  Reference-anchored mode: testing whether any fill beats '{reference}' (the default).")
    print("#" * 80)

    if len(strategies) < 2:
        print(
            "\n  Only one fill strategy present "
            f"({strategies[0] if strategies else 'none'}); nothing to rank.\n"
        )
        return

    any_output = False
    for ds in datasets:
        ds_runs = [r for r in all_runs if r.get("dataset") == ds]
        if not ds_runs:
            continue

        # method -> strategy -> {seed: held_out_metric}
        by_method: dict[str, dict[str, dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for r in ds_runs:
            cid = r["condition"]
            if cid not in FILL_USING_CONDITIONS:
                continue
            strat = r.get("fill_strategy", "none")
            if strat == "none":
                continue  # a fill-using method must carry a real strategy tag
            metric = float(r.get("held_out_test_metric") or r.get("val_metric") or 0.0)
            by_method[cid][strat][int(r["seed"])] = metric

        methods_present = [c for c in FILL_USING_CONDITIONS if c in by_method]
        if not methods_present:
            continue

        print(f"\n{'='*70}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'='*70}")
        any_output = True

        for cid in methods_present:
            result = _rank_strategies_for_method(
                by_method[cid], strategies,
                no_bootstrap=no_bootstrap, n_boot=n_boot,
                reference=reference,
            )
            label = DISPLAY_NAMES.get(cid, cid)
            if result is None:
                print(f"\n  {label} ({cid}): only one strategy has data — nothing to rank.")
                continue
            if result.get("insufficient"):
                n_sh = len(result["shared"])
                print(
                    f"\n  {label} ({cid}): insufficient shared-seed data "
                    f"(n={n_sh}); need >= 2 seeds common to all strategies."
                )
                continue

            if markdown:
                _print_fill_ranking_markdown(cid, label, result)
            else:
                _print_fill_ranking_text(cid, label, result)

    if not any_output:
        print("\n  No fill-using method has multi-strategy data to rank.\n")


def _verdict_line(result: dict[str, Any]) -> str:
    ref = result["reference"]
    challenger = result.get("challenger")
    if result.get("mode") == "reference":
        if result["meaningful"]:
            return f"{challenger} beats the reference ({ref}) significantly — candidate to replace the default."
        if result.get("reference_is_top"):
            return f"reference ({ref}) has the highest median; no strategy beats it — keep the default."
        return f"{challenger} leads the reference ({ref}) but not significantly — keep the default (change unjustified)."
    if result["meaningful"]:
        return f"best: {ref} — gap to runner-up ({challenger}) is significant (meaningful)."
    return (f"best: {ref} — but not clearly separated from runner-up ({challenger}); "
            "treat the ranking as tentative.")


def _print_fill_ranking_text(cid: str, label: str, result: dict[str, Any]) -> None:
    shared = result["shared"]
    seed_s = ",".join(str(s) for s in shared)
    ref = result["reference"]
    tag = "reference" if result.get("mode") == "reference" else "best"
    print(f"\n  {label} ({cid})  — shared seeds n={len(shared)}: {seed_s}")
    if result.get("reference_missing"):
        print(f"    note: requested reference {result.get('reference_requested')!r} "
              f"has no data here; using empirical best ({ref}).")
    for i, e in enumerate(result["entries"], start=1):
        med_s = f"{e['median']*100:.2f}%"
        if e["is_ref"]:
            print(f"    {i}. {e['strategy']:<14} {med_s:>8}   ({tag})")
            continue
        ci_s = ""
        if not (math.isnan(e["ci_lo"]) or math.isnan(e["ci_hi"])):
            ci_s = f"  CI=[{e['ci_lo']*100:+.2f},{e['ci_hi']*100:+.2f}]pp"
        r_s = f"{e['r']:.2f}" if e["r"] is not None else "—"
        print(
            f"    {i}. {e['strategy']:<14} {med_s:>8}   "
            f"Δ={e['delta']:+.2f}pp vs {ref}  "
            f"{_p_label(e['p_holm'])} {e['stars']}  r={r_s}{ci_s}"
        )
    print(f"    -> {_verdict_line(result)}")
    if not result.get("gate_reachable", True):
        n_strat = len(result["entries"])
        print(
            f"    note: underpowered — with {n_strat} strategies and n={len(shared)} shared "
            f"seeds the 'meaningful' gate cannot fire (best-case Holm p>={result['holm_floor']:.4f}); "
            f"any verdict reads 'tentative'. Increase seeds (>=8 for 5 strategies)."
        )


def _print_fill_ranking_markdown(cid: str, label: str, result: dict[str, Any]) -> None:
    shared = result["shared"]
    ref = result["reference"]
    tag = "reference" if result.get("mode") == "reference" else "best"
    print(f"\n**{label} (`{cid}`)** — shared seeds n={len(shared)}")
    if result.get("reference_missing"):
        print(f"\n> requested reference `{result.get('reference_requested')}` has no data here; "
              f"using empirical best (`{ref}`).")
    print("| Rank | Strategy | Median | Δ vs ref | p (Holm) | stars | r | Bootstrap 95% CI |")
    print("|------|----------|--------|---------|---|-------|---|------------------|")
    for i, e in enumerate(result["entries"], start=1):
        med_s = f"{e['median']*100:.2f}%"
        if e["is_ref"]:
            print(f"| {i} | {e['strategy']} | {med_s} | ({tag}) | — | — | — | — |")
            continue
        ci_s = (f"[{e['ci_lo']*100:+.2f}, {e['ci_hi']*100:+.2f}]pp"
                if not (math.isnan(e["ci_lo"]) or math.isnan(e["ci_hi"])) else "—")
        r_s = f"{e['r']:.2f}" if e["r"] is not None else "—"
        print(
            f"| {i} | {e['strategy']} | {med_s} | {e['delta']:+.2f}pp | "
            f"{_p_label(e['p_holm'])} | {e['stars']} | {r_s} | {ci_s} |"
        )
    print(f"\n**{_verdict_line(result).capitalize()}**")
    if not result.get("gate_reachable", True):
        n_strat = len(result["entries"])
        print(
            f"\n> **Underpowered:** {n_strat} strategies, n={len(shared)} shared seeds — "
            f"the 'meaningful' gate cannot fire (best-case Holm p>={result['holm_floor']:.4f}); "
            f"any verdict reads 'tentative'. Increase seeds (>=8 for 5 strategies)."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BENCHMARKS_DIR,
        help="Directory containing results_*.json files (default: benchmarks/)",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets to include (default: all found)",
    )
    parser.add_argument(
        "--regime",
        default="all",
        choices=["scratch", "pretrained", "all"],
        help="Filter by regime (default: all)",
    )
    parser.add_argument("--markdown", action="store_true", help="Output markdown tables")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap CI")
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=10_000,
        help="Bootstrap resampling iterations (default 10000)",
    )
    parser.add_argument(
        "--reference-fill",
        default=None,
        help="Anchor the 'which fill is best' comparison on this fixed strategy "
             "(e.g. the current library default 'gaussian_blur') and ask whether any "
             "fill beats it. Default: rank by empirical best and test best-vs-runner-up.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"Results directory not found: {args.results_dir}")

    all_runs = _load_all_runs(args.results_dir, args.regime)
    if not all_runs:
        raise SystemExit(
            f"No valid runs found in {args.results_dir} "
            f"(regime filter: {args.regime}).\n"
            "Run: python benchmarks/run_grand_benchmark.py --dataset imagewoof --smoke"
        )

    # Determine which datasets to analyse
    if args.datasets is not None:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        found = sorted({r.get("dataset", "unknown") for r in all_runs})
        datasets = [d for d in ALL_DATASETS if d in found]
        if not datasets:
            datasets = found

    real_strategies = {r.get("fill_strategy", "none") for r in all_runs} - {"none"}
    strategies = [s for s in FILL_STRATEGY_ORDER if s in real_strategies]
    strategies += sorted(real_strategies - set(FILL_STRATEGY_ORDER))
    if not strategies:
        strategies = ["none"]  # baseline-only file: a single unlabelled pass

    print(f"\nGrand Benchmark Summary  |  regime={args.regime}")
    print(f"Results dir: {args.results_dir}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Fill strategies: {', '.join(strategies)}")
    print(f"Total valid runs: {len(all_runs)}")

    for strategy in strategies:
        # View = this strategy's fill-using results + the shared "none" baselines.
        view = [
            r for r in all_runs
            if r.get("fill_strategy", "none") in (strategy, "none")
        ]

        print("\n" + "#" * 80)
        print(f"#  FILL STRATEGY: {strategy}   ({len(view)} runs in view)")
        print("#" * 80)

        # 1. Per-dataset tables
        for ds in datasets:
            _analyze_dataset(
                ds, view,
                no_bootstrap=args.no_bootstrap,
                n_boot=args.bootstrap_n,
                markdown=args.markdown,
                strategy=strategy,
            )

        # 2. Cross-dataset summary table
        _cross_dataset_table(view, datasets, markdown=args.markdown, strategy=strategy)

        # 3. XAI correlation section
        _xai_correlation_section(view, datasets, markdown=args.markdown, strategy=strategy)

        # 4. Key comparison: bnnr_xai vs bnnr_random
        _key_comparison_section(
            view, datasets,
            no_bootstrap=args.no_bootstrap,
            n_boot=args.bootstrap_n,
            markdown=args.markdown,
            strategy=strategy,
        )

        # 5. Compute transparency
        _compute_transparency_section(view, datasets, strategy=strategy)

    _which_fill_is_best_section(
        all_runs, datasets, strategies,
        no_bootstrap=args.no_bootstrap,
        n_boot=args.bootstrap_n,
        markdown=args.markdown,
        reference=args.reference_fill,
    )

    # Footer
    print(
        "\nStatistical notes:"
        "\n  Tests: Wilcoxon signed-rank (paired, two-sided) — scipy if available, "
        "else exact enumeration (n<=15) or normal approx."
        "\n  Multiple testing: Holm-Bonferroni correction over all pairwise tests vs bnnr_xai."
        "\n  Bootstrap CI: paired (bnnr_xai - baseline) median difference."
        "\n  Effect size r: rank-biserial correlation = 1 - 2W/(n*(n+1))."
        "\n  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05."
    )


if __name__ == "__main__":
    main()
