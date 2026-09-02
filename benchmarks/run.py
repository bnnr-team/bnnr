#!/usr/bin/env python3
"""Run CIFAR-10 augmentation benchmarks. See benchmarks/README.md."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from lib import (  # noqa: E402
    CONDITIONS,
    DEFAULT_CONFIG,
    benchmark_document,
    force_utf8_stdout,
    git_head,
    load_results,
    run_condition,
    save_results,
    torch_info,
)

DEFAULT_RESULTS = BENCHMARKS_DIR / "results.json"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "runs"


def _estimate_hours(device: str, n_seeds: int, condition_ids: list[str]) -> str:
    minutes_per = {
        "no_bnnr": 5,
        "randaugment": 5,
        "bnnr_branch_search": 45,
    }
    total_min = sum(minutes_per.get(c, 15) for c in condition_ids) * n_seeds
    if device == "cuda":
        total_min = int(total_min * 0.25)
    return f"~{total_min / 60:.1f}h ({len(condition_ids)} conditions × {n_seeds} seed(s), {device})"


def main() -> None:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="CIFAR-10 benchmark: no BNNR vs RandAugment vs BNNR branch search",
        epilog="""
Examples:
  python benchmarks/run.py --seeds 42 --device cpu
  python benchmarks/run.py --seeds 42,43,44 --device cpu
  python benchmarks/summarize.py --markdown
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10",))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cpu", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS.keys()),
        help="Comma-separated condition ids",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    args = parser.parse_args()

    if args.list_conditions:
        for cid, spec in CONDITIONS.items():
            print(f"\n## {cid} — {spec.label}")
            print(f"   augs: {', '.join(spec.augmentation_names) or '(none)'}")
            print(f"   {spec.description}")
        return

    condition_ids = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in condition_ids if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown: {unknown}. Use --list-conditions.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    args.output.mkdir(parents=True, exist_ok=True)

    if args.fresh and args.results.exists():
        bak = args.results.with_suffix(
            f".bak.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        args.results.rename(bak)
        print(f"Backed up {args.results} -> {bak}")

    store = load_results(args.results)
    store.update(benchmark_document())
    store["hardware"] = torch_info()
    store["git_commit"] = git_head()
    store["updated_at"] = datetime.now(timezone.utc).isoformat()

    done = {(r["condition"], r["dataset"], r["seed"]) for r in store.get("runs", [])}
    dev = args.device if args.device != "auto" else "cpu"
    print(
        f"\nCIFAR-10 benchmark | seeds={seeds}\n"
        f"Conditions: {condition_ids}\n"
        f"ETA: {_estimate_hours(dev, len(seeds), condition_ids)}\n"
        f"Results: {args.results}\n"
    )

    for seed in seeds:
        for cid in condition_ids:
            key = (cid, args.dataset, seed)
            if key in done:
                print(f"SKIP {cid} seed={seed}")
                continue
            print(f"\n>>> {cid} | seed={seed}")
            entry = run_condition(
                condition_id=cid,
                dataset=args.dataset,
                seed=seed,
                device=args.device,
                config_path=args.config,
                output_root=args.output,
            )
            store.setdefault("runs", []).append(entry)
            store["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_results(args.results, store)
            cov = entry.get("attention_coverage")
            cov_s = f"  coverage={cov:.3f}" if cov is not None else ""
            print(f"    acc={entry['val_metric']:.4f}{cov_s}  ({entry['wall_clock_s']}s)")

    print(f"\nDone -> {args.results}\nSummarize: python benchmarks/summarize.py --markdown")


if __name__ == "__main__":
    main()
