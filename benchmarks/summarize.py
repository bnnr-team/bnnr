#!/usr/bin/env python3
"""Summarize benchmarks/results.json — metrics and attention statistics."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BENCHMARKS_DIR / "results.json"

DISPLAY = {
    "no_bnnr": "Without BNNR (crop + flip)",
    "randaugment": "RandAugment (torchvision)",
    "trivialaugment": "TrivialAugmentWide (torchvision)",
    "bnnr_branch_search": "BNNR branch search (ICD + AICD)",
}


def _agg(values: list[float], fn: Callable[[list[float]], float]) -> float | None:
    return fn(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"Missing {args.results}. Run: python benchmarks/run.py")

    data = json.loads(args.results.read_text(encoding="utf-8"))
    runs = data.get("runs") or []
    if not runs:
        raise SystemExit("No runs in results file.")

    agg_fn = statistics.median if args.aggregate == "median" else statistics.mean

    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_cond[r["condition"]].append(r)

    ref_acc = _agg([float(r["val_metric"]) for r in by_cond.get("no_bnnr", [])], agg_fn)

    print(f"\nBenchmark: {data.get('benchmark_id', 'cifar10')}")
    print(f"Git: {str(data.get('git_commit', ''))[:8]}  Hardware: {data.get('hardware')}\n")

    print("## What we compare\n")
    for comp in data.get("comparisons") or []:
        print(f"- **{comp['label']}**: {', '.join(comp['conditions'])}")

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for cid in DISPLAY:
        if cid not in by_cond:
            continue
        rs = by_cond[cid]
        accs = [float(r["val_metric"]) for r in rs]
        acc = agg_fn(accs) * 100
        acc_s = f"{acc:.1f}%" + (f" (n={len(accs)})" if len(accs) > 1 else "")
        delta = "—"
        if ref_acc is not None and cid != "no_bnnr":
            delta = f"{acc - ref_acc * 100:+.1f}pp vs no BNNR"

        covs = [float(r["attention_coverage"]) for r in rs if r.get("attention_coverage") is not None]
        edges = [float(r["attention_edge_ratio"]) for r in rs if r.get("attention_edge_ratio") is not None]
        cov_s = f"{agg_fn(covs):.1%}" if covs else "—"
        edge_s = f"{agg_fn(edges):.2f}" if edges else "—"

        per_seed = ", ".join(f"{v*100:.1f}%" for v in accs)
        xai = str(rs[-1].get("xai_dir") or "—")
        rows.append((DISPLAY[cid], acc_s, delta, cov_s, edge_s, per_seed, xai))

    if args.markdown:
        print("\n| Condition | Val accuracy | Δ vs no BNNR | Mean attention coverage | Edge ratio | Per-seed | XAI dir |")
        print("|-----------|--------------|--------------|-------------------------|------------|----------|---------|")
        for label, acc, delta, cov, edge, ps, xai in rows:
            print(f"| {label} | {acc} | {delta} | {cov} | {edge} | {ps} | `{xai}` |")
    else:
        print(f"\n{'Condition':<36} {'Accuracy':>12} {'Δ no BNNR':>12} {'Coverage':>10} {'Edge':>8}")
        print("-" * 82)
        for label, acc, delta, cov, edge, _, xai in rows:
            print(f"{label:<36} {acc:>12} {delta:>12} {cov:>10} {edge:>8}")
            print(f"  xai: {xai}")

    print(
        "\nAttention overlays: same validation indices in each run's `xai/` folder. "
        "Lower edge ratio + tighter coverage often means less background focus."
    )


if __name__ == "__main__":
    main()
