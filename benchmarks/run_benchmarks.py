#!/usr/bin/env python3
"""Run baseline vs BNNR benchmarks and append results to benchmarks/results.json."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BENCHMARKS_DIR / "configs"
RESULTS_PATH = BENCHMARKS_DIR / "results.json"
RUNS_DIR = BENCHMARKS_DIR / "runs"

DATASETS = ("cifar10", "fashion_mnist", "stl10")


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _torch_info() -> dict[str, str | None]:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return {
            "pytorch_version": torch.__version__,
            "gpu": gpu,
            "platform": platform.platform(),
        }
    except ImportError:
        return {
            "pytorch_version": None,
            "gpu": None,
            "platform": platform.platform(),
        }


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {
        "protocol_version": 1,
        "notes": "",
        "hardware": {},
        "runs": [],
    }


def _save_results(data: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _extract_baseline_accuracy(report_path: Path, selection_metric: str) -> float | None:
    """Best-effort baseline val metric from report.json."""
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    summaries = raw.get("iteration_summaries") or {}
    if summaries:
        first = summaries.get("1") or summaries.get(1) or {}
        base = first.get("baseline_metrics") or {}
        if selection_metric in base:
            return float(base[selection_metric])
    checkpoints = raw.get("checkpoints") or []
    for ck in checkpoints:
        if ck.get("augmentation") == "baseline":
            metrics = ck.get("metrics") or {}
            if selection_metric in metrics:
                return float(metrics[selection_metric])
    return None


def run_single(
    *,
    mode: str,
    dataset: str,
    seed: int,
    device: str,
    output_root: Path,
    fast: bool,
) -> dict:
    """Run one training job via Python API."""
    sys.path.insert(0, str(REPO_ROOT / "src"))

    from bnnr.config import load_config
    from bnnr.core import BNNRTrainer
    from bnnr.pipelines import build_pipeline

    if mode == "baseline":
        cfg_path = CONFIGS_DIR / ("pilot_fast.yaml" if fast else "baseline_only.yaml")
        preset = "none"
    elif mode == "bnnr":
        cfg_path = CONFIGS_DIR / ("pilot_fast.yaml" if fast else "bnnr_light.yaml")
        if not fast:
            cfg = load_config(cfg_path)
            cfg_path = None  # use loaded then override iterations below
        preset = "light"
    else:
        raise ValueError(mode)

    run_dir = output_root / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{mode}_{dataset}_s{seed}"
    ckpt_dir = run_dir / "checkpoints"
    report_dir = run_dir / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if cfg_path is not None:
        cfg = load_config(cfg_path)
    else:
        cfg = load_config(CONFIGS_DIR / "bnnr_light.yaml")

    if mode == "baseline":
        cfg = cfg.model_copy(update={"max_iterations": 0})
    elif fast:
        cfg = cfg.model_copy(update={"max_iterations": 2, "m_epochs": 3})
    else:
        cfg = cfg.model_copy(update={"max_iterations": 3})

    cfg = cfg.model_copy(
        update={
            "seed": seed,
            "device": device,
            "checkpoint_dir": ckpt_dir,
            "report_dir": report_dir,
            "event_log_enabled": False,
        }
    )

    adapter, train_loader, val_loader, augmentations = build_pipeline(
        dataset_name=dataset,
        config=cfg,
        data_dir=REPO_ROOT / "data",
        batch_size=64,
        augmentation_preset=preset,
    )

    t0 = time.perf_counter()
    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
    result = trainer.run()
    elapsed_s = time.perf_counter() - t0

    sel = cfg.selection_metric
    best = float(result.best_metrics.get(sel, 0.0))
    baseline_acc: float | None = None
    if mode == "bnnr" and result.report_json_path.exists():
        baseline_acc = _extract_baseline_accuracy(result.report_json_path, sel)
    if mode == "baseline":
        baseline_acc = best

    return {
        "mode": mode,
        "dataset": dataset,
        "seed": seed,
        "selection_metric": sel,
        "val_metric": best,
        "baseline_val_metric": baseline_acc,
        "best_path": result.best_path,
        "wall_clock_s": round(elapsed_s, 1),
        "report_json": str(result.report_json_path.relative_to(REPO_ROOT)),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BNNR baseline vs branching benchmarks")
    parser.add_argument("--pilot", action="store_true", help="Single dataset, baseline+bnnr, fast config if --fast")
    parser.add_argument("--fast", action="store_true", help="Use pilot_fast.yaml settings")
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--datasets", default="", help="Comma-separated list (overrides --dataset)")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    if args.pilot:
        datasets = [args.dataset]
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [args.dataset]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    args.output.mkdir(parents=True, exist_ok=True)

    store = _load_results()
    hw = _torch_info()
    store["hardware"] = hw
    store["git_commit"] = _git_head()
    store["updated_at"] = datetime.now(timezone.utc).isoformat()

    for dataset in datasets:
        for seed in seeds:
            for mode in ("baseline", "bnnr"):
                print(f"\n=== {mode} | {dataset} | seed={seed} ===", flush=True)
                entry = run_single(
                    mode=mode,
                    dataset=dataset,
                    seed=seed,
                    device=args.device,
                    output_root=args.output,
                    fast=args.fast or args.pilot,
                )
                store["runs"].append(entry)
                _save_results(store)
                print(
                    f"  {entry['selection_metric']}={entry['val_metric']:.4f} "
                    f"({entry['wall_clock_s']}s) -> {entry['report_json']}",
                    flush=True,
                )

    print(f"\nWrote {RESULTS_PATH}", flush=True)
    print("Next: python benchmarks/update_readme_table.py", flush=True)


if __name__ == "__main__":
    main()
