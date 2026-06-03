#!/usr/bin/env python3
"""Summarize benchmark results — metrics, std dev, and significance tests.

Works with both benchmarks/results.json (CIFAR-10) and
benchmarks/results_resnet50.json (CIFAR-100 / ResNet50).

Examples
--------
    python benchmarks/summarize.py --results benchmarks/results_resnet50.json --markdown
    python benchmarks/summarize.py --results benchmarks/results_resnet50.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BENCHMARKS_DIR / "results_resnet50.json"

DISPLAY = {
    "no_bnnr": "Without BNNR (crop + flip)",
    "randaugment": "RandAugment (torchvision)",
    "trivialaugment": "TrivialAugmentWide (torchvision)",
    "bnnr_branch_search": "BNNR branch search (ICD + AICD)",
}


def _agg(values: list[float], fn: Callable[[list[float]], float]) -> float | None:
    return fn(values) if values else None


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float] | None:
    """Welch's t-test (two-sided, unequal variance). Returns (t, p) or None."""
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        mean_a, mean_b = statistics.mean(a), statistics.mean(b)
        var_a = statistics.variance(a)
        var_b = statistics.variance(b)
        na, nb = len(a), len(b)
        se = math.sqrt(var_a / na + var_b / nb)
        if se == 0:
            return None
        t = (mean_a - mean_b) / se
        # Welch-Satterthwaite degrees of freedom
        num = (var_a / na + var_b / nb) ** 2
        den = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
        df = num / den if den > 0 else 1.0
        # Approximate p-value via incomplete beta — use simple lookup for small df
        p = _p_from_t(abs(t), df)
        return t, p
    except (statistics.StatisticsError, ZeroDivisionError):
        return None


def _p_from_t(t: float, df: float) -> float:
    """Two-sided p-value approximation for Welch t (accurate enough for n=5)."""
    # Abramowitz & Stegun approximation of the t CDF tail
    x = df / (df + t * t)
    # Regularized incomplete beta via continued fraction (simple approx)
    # For our purposes (n=5 seeds), we only need to know p<0.05/0.01/0.001
    try:
        import math

        # Use scipy if available for accurate p-values
        from scipy.stats import t as t_dist

        return float(2 * t_dist.sf(t, df))
    except ImportError:
        pass
    # Fallback: crude approximation valid for df >= 4
    # Based on Hill (1970) approximation
    a = df / 2.0
    x = df / (df + t * t)
    # Incomplete beta approximation
    lbeta = math.lgamma(a) + math.lgamma(0.5) - math.lgamma(a + 0.5)
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    # Simple Euler continued fraction approximation
    p_half = math.exp(a * math.log(x) + 0.5 * math.log(1 - x) - lbeta) / a
    return min(1.0, 2 * p_half)


def _sig_stars(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 0.001:
        return "p<0.001 ***"
    if p < 0.01:
        return f"p={p:.3f} **"
    if p < 0.05:
        return f"p={p:.3f} *"
    return f"p={p:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to results JSON (default: benchmarks/results_resnet50.json)",
    )
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"Missing {args.results}.\n"
            "Run: python benchmarks/run_resnet50.py --seeds 42,43,44,45,46 --device cuda"
        )

    data = json.loads(args.results.read_text(encoding="utf-8"))
    runs = [r for r in (data.get("runs") or []) if "val_metric" in r and "error" not in r]
    if not runs:
        raise SystemExit("No valid (non-error) runs found in results file.")

    agg_fn = statistics.median if args.aggregate == "median" else statistics.mean

    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_cond[r["condition"]].append(r)

    ref_accs = [float(r["val_metric"]) for r in by_cond.get("no_bnnr", [])]
    ref_acc = _agg(ref_accs, agg_fn)

    print(f"\nBenchmark: {data.get('benchmark_id', 'unknown')}")
    hw = data.get("hardware") or {}
    git = str(data.get("git_head") or data.get("git_commit") or "")[:8]
    print(f"Git: {git or '—'}  GPU: {hw.get('gpu', '—')}  PyTorch: {hw.get('pytorch_version', '—')}\n")

    rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    for cid in DISPLAY:
        if cid not in by_cond:
            continue
        rs = by_cond[cid]
        accs = [float(r["val_metric"]) for r in rs]
        acc = agg_fn(accs) * 100
        std = statistics.stdev(accs) * 100 if len(accs) > 1 else 0.0
        n = len(accs)
        acc_s = f"{acc:.2f}% ±{std:.2f} (n={n})"

        delta = sig = "—"
        if ref_acc is not None and cid != "no_bnnr":
            delta = f"{acc - ref_acc * 100:+.2f}pp"
            result = _welch_t(accs, ref_accs)
            sig = _sig_stars(result[1] if result else None)

        covs = [float(r["attention_coverage"]) for r in rs if r.get("attention_coverage") is not None]
        edges = [float(r["attention_edge_ratio"]) for r in rs if r.get("attention_edge_ratio") is not None]
        cov_s = f"{agg_fn(covs):.1%}" if covs else "—"
        edge_s = f"{agg_fn(edges):.2f}" if edges else "—"

        per_seed = ", ".join(f"{v*100:.2f}%" for v in accs)
        rows.append((DISPLAY[cid], acc_s, delta, sig, cov_s, edge_s, per_seed, cid))

    if args.markdown:
        print("| Condition | Val accuracy (median ±std) | Δ vs no BNNR | Significance | Coverage | Edge ratio | Per-seed |")
        print("|-----------|---------------------------|--------------|--------------|----------|------------|----------|")
        for label, acc, delta, sig, cov, edge, ps, _ in rows:
            print(f"| {label} | {acc} | {delta} | {sig} | {cov} | {edge} | {ps} |")
    else:
        print(f"{'Condition':<38} {'Accuracy (±std)':>22} {'Δ':>10} {'Sig':>14} {'Cov':>7} {'Edge':>6}")
        print("-" * 100)
        for label, acc, delta, sig, cov, edge, ps, _ in rows:
            print(f"{label:<38} {acc:>22} {delta:>10} {sig:>14} {cov:>7} {edge:>6}")
            print(f"  per-seed: {ps}")

    print(
        "\nSignificance: Welch t-test (two-sided, no_bnnr as reference). "
        "*** p<0.001, ** p<0.01, * p<0.05. "
        "n=5 seeds recommended for publishable results."
    )


if __name__ == "__main__":
    main()
