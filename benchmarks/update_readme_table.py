#!/usr/bin/env python3
"""Update README.md benchmark table from benchmarks/results.json."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
RESULTS = Path(__file__).resolve().parent / "results.json"

DISPLAY = {
    "cifar10": "CIFAR-10",
    "fashion_mnist": "Fashion-MNIST",
    "stl10": "STL-10",
}


def _aggregate(runs: list[dict], use_median: bool) -> list[dict]:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_ds[r["dataset"]].append(r)

    rows: list[dict] = []
    for dataset, items in sorted(by_ds.items()):
        baselines = [x["val_metric"] for x in items if x["mode"] == "baseline"]
        bnnrs = [x["val_metric"] for x in items if x["mode"] == "bnnr"]
        if not baselines or not bnnrs:
            continue
        agg = statistics.median if use_median else statistics.mean
        b = agg(baselines) * 100
        bn = agg(bnnrs) * 100
        rows.append(
            {
                "dataset": DISPLAY.get(dataset, dataset),
                "baseline": f"{b:.1f}%",
                "bnnr": f"{bn:.1f}%",
                "gain": f"{bn - b:+.1f}pp",
            }
        )
    return rows


def _placeholder_rows() -> list[dict]:
    return [
        {
            "dataset": name,
            "baseline": "TBD",
            "bnnr": "TBD",
            "gain": "—",
        }
        for name in ("CIFAR-10", "Fashion-MNIST", "STL-10")
    ]


def _build_table(rows: list[dict], footnote: str) -> str:
    lines = [
        "| Dataset | Baseline | + BNNR | Gain |",
        "|---------|----------|--------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['baseline']} | {r['bnnr']} | {r['gain']} |"
        )
    lines.append("")
    lines.append(footnote)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    use_median = args.aggregate == "median"
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        runs = data.get("runs") or []
        rows = _aggregate(runs, use_median) if runs else _placeholder_rows()
        if runs:
            n = len({(r["dataset"], r["seed"]) for r in runs if r["mode"] == "baseline"})
            footnote = (
                f"*Pilot / early results from [`benchmarks/results.json`](benchmarks/results.json) "
                f"(built-in demo CNN, val accuracy, {n} seed(s)). Reproduce: "
                f"`python benchmarks/run_benchmarks.py --pilot --dataset cifar10 --seed 42`.*"
            )
        else:
            footnote = (
                "*Run [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py) to fill this table. "
                "See [benchmarks/README.md](benchmarks/README.md).*"
            )
    else:
        rows = _placeholder_rows()
        footnote = "*See [benchmarks/README.md](benchmarks/README.md) for reproduction steps.*"

    table = _build_table(rows, footnote)
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(
        r"## Benchmarks\n\n.*?(?=\n---\n\n## Quickstart)",
        re.DOTALL,
    )
    replacement = f"## Benchmarks\n\n{table}\n\n---\n\n## Quickstart"
    if not pattern.search(text):
        raise SystemExit("Could not find ## Benchmarks section in README.md")
    new_text = pattern.sub(replacement, text, count=1)
    if args.dry_run:
        print(new_text[text.find("## Benchmarks") : text.find("## Quickstart") + 20])
        return
    README.write_text(new_text, encoding="utf-8")
    print(f"Updated {README}")


if __name__ == "__main__":
    main()
