"""CIFAR-10 benchmark library — metrics + attention-map export."""

from __future__ import annotations

import copy
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BENCHMARKS_DIR / "config.yaml"


def _rel(p: Path) -> str:
    """Return path relative to REPO_ROOT when possible, otherwise absolute."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

Strategy = Literal["plain_training", "randaugment", "trivialaugment", "bnnr_branch_search"]


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    label: str
    strategy: Strategy
    description: str
    augmentation_names: tuple[str, ...]
    max_iterations: int = 0


CONDITIONS: dict[str, ConditionSpec] = {
    "no_bnnr": ConditionSpec(
        id="no_bnnr",
        label="Without BNNR (crop + flip only)",
        strategy="plain_training",
        description=(
            "Standard CIFAR-10 training: RandomCrop + RandomHorizontalFlip. "
            "No BNNR batch augmentations and no branch search."
        ),
        augmentation_names=(),
        max_iterations=0,
    ),
    "randaugment": ConditionSpec(
        id="randaugment",
        label="RandAugment (torchvision)",
        strategy="randaugment",
        description=(
            "External baseline: RandomCrop + RandomHorizontalFlip + "
            "torchvision RandAugment on the dataset transform. "
            "Random policy-based augmentations — no saliency guidance, no BNNR."
        ),
        augmentation_names=("RandAugment",),
        max_iterations=0,
    ),
    "bnnr_branch_search": ConditionSpec(
        id="bnnr_branch_search",
        label="BNNR branch search (ICD + AICD + ChurchNoise)",
        strategy="bnnr_branch_search",
        description=(
            "Full BNNR loop: baseline phase, then branch search over "
            "**ICD** (mask high-saliency regions), **AICD** (mask background), "
            "and **ChurchNoise**. Keeps branches that improve validation accuracy."
        ),
        augmentation_names=("ICD", "AICD", "augmentation_1"),
        max_iterations=3,
    ),
}

COMPARISONS = [
    {
        "id": "bnnr_vs_no_aug",
        "label": "BNNR branch search vs no augmentation",
        "conditions": ["bnnr_branch_search", "no_bnnr"],
        "metrics": ["val_accuracy", "attention_coverage", "attention_edge_ratio"],
    },
    {
        "id": "bnnr_vs_randaugment",
        "label": "BNNR branch search vs RandAugment",
        "conditions": ["bnnr_branch_search", "randaugment"],
        "metrics": ["val_accuracy", "attention_coverage", "attention_edge_ratio"],
    },
    {
        "id": "attention_maps",
        "label": "Where the model looks (OptiCAM on shared val images)",
        "conditions": ["no_bnnr", "randaugment", "bnnr_branch_search"],
        "artifacts": ["xai_overlays"],
    },
]


def build_bnnr_candidate_augmentations(
    model: Any,
    target_layers: list[Any],
    seed: int,
) -> list[Any]:
    """BNNR candidate pool for the benchmark — saliency-guided ICD/AICD plus ChurchNoise."""
    from bnnr.augmentations import ChurchNoise
    from bnnr.icd import AICD, ICD

    return [
        ICD(
            model=model,
            target_layers=target_layers,
            threshold_percentile=75.0,
            probability=0.5,
            random_state=seed,
        ),
        AICD(
            model=model,
            target_layers=target_layers,
            threshold_percentile=75.0,
            probability=0.5,
            random_state=seed + 1,
        ),
        ChurchNoise(
            probability=0.5,
            intensity=0.5,
            noise_strength_range=(3.0, 8.0),
            random_state=seed + 2,
        ),
    ]


def load_training_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    from bnnr.config_model import BNNRConfig

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench = dict(raw.pop("_benchmark", {}) or {})
    cfg = BNNRConfig(**raw)
    return cfg, bench


def benchmark_document() -> dict[str, Any]:
    return {
        "benchmark_id": "cifar10_augmentation_comparison",
        "model": "built-in demo CNN (bnnr.pipelines._CifarCNN)",
        "dataset": "cifar10 (torchvision, full train/val split)",
        "batch_size": 64,
        "optimizer": "Adam(lr=1e-3)",
        "scheduler": "CosineAnnealingLR",
        "primary_metric": "validation accuracy (best epoch)",
        "attention_method": "OptiCAM saliency overlays on fixed validation indices",
        "shared_config": str(DEFAULT_CONFIG.relative_to(REPO_ROOT)),
        "comparisons": COMPARISONS,
        "conditions": {
            cid: {
                "label": spec.label,
                "strategy": spec.strategy,
                "max_iterations": spec.max_iterations,
                "augmentations": list(spec.augmentation_names),
                "description": spec.description,
            }
            for cid, spec in CONDITIONS.items()
        },
    }


def git_head() -> str | None:
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


def torch_info() -> dict[str, str | None]:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return {
            "pytorch_version": torch.__version__,
            "gpu": gpu,
            "platform": platform.platform(),
        }
    except ImportError:
        return {"pytorch_version": None, "gpu": None, "platform": platform.platform()}


def load_results(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {**benchmark_document(), "hardware": {}, "runs": []}


def save_results(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_dataloader(seed: int) -> tuple[Any, Any]:
    import torch

    def worker_init(worker_id: int) -> None:
        w = seed + worker_id
        np.random.seed(w)
        random.seed(w)
        torch.manual_seed(w)

    gen = torch.Generator()
    gen.manual_seed(seed)
    return worker_init, gen


def _build_cifar10_pipeline(*, cfg: Any, batch_size: int = 64):
    from bnnr.pipelines import build_cifar10_pipeline

    return build_cifar10_pipeline(
        config=cfg,
        data_dir=REPO_ROOT / "data",
        batch_size=batch_size,
        max_train_samples=None,
        max_val_samples=None,
        augmentation_preset="none",
    )


def _build_cifar10_randaugment_loaders(
    cfg: Any,
    *,
    num_ops: int,
    magnitude: int,
    batch_size: int = 64,
):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from bnnr.pipelines import _IndexedDataset, _maybe_subset

    adapter, _, val_loader, _ = _build_cifar10_pipeline(cfg=cfg, batch_size=batch_size)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose([transforms.ToTensor()])
    data_root = str(REPO_ROOT / "data")
    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    val_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    train_ds = _maybe_subset(train_ds, None)
    val_ds = _maybe_subset(val_ds, None)

    worker_init, generator = _seed_dataloader(cfg.seed)
    train_loader = DataLoader(
        _IndexedDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init,
        generator=generator,
    )
    val_loader = DataLoader(_IndexedDataset(val_ds), batch_size=batch_size, shuffle=False)
    return adapter, train_loader, val_loader


def _extract_baseline_from_report(report_path: Path, selection_metric: str) -> float | None:
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    summaries = raw.get("iteration_summaries") or {}
    if summaries:
        first = summaries.get("1") or summaries.get(1) or {}
        base = first.get("baseline_metrics") or {}
        if selection_metric in base:
            return float(base[selection_metric])
    for ck in raw.get("checkpoints") or []:
        if ck.get("augmentation") == "baseline":
            metrics = ck.get("metrics") or {}
            if selection_metric in metrics:
                return float(metrics[selection_metric])
    return None


def export_attention_maps(
    adapter: Any,
    val_loader: Any,
    *,
    sample_indices: list[int],
    output_dir: Path,
    xai_method: str,
) -> dict[str, Any]:
    """OptiCAM overlays on the same validation images for every condition."""
    import torch
    from torch import Tensor

    from bnnr.xai import generate_saliency_maps, save_xai_visualization
    from bnnr.xai_analysis import analyze_saliency_map

    dataset = getattr(val_loader, "dataset", None)
    if dataset is None:
        return {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}

    output_dir.mkdir(parents=True, exist_ok=True)
    images_t: list[Tensor] = []
    labels_t: list[Tensor] = []
    used_indices: list[int] = []

    for ds_idx in sample_indices:
        try:
            sample = dataset[ds_idx]
        except (IndexError, KeyError):
            continue
        img = sample[0] if isinstance(sample, (list, tuple)) else sample
        lbl = sample[1] if isinstance(sample, (list, tuple)) and len(sample) > 1 else 0
        if not isinstance(img, Tensor):
            continue
        images_t.append(img.unsqueeze(0))
        lbl_val = int(lbl.item()) if isinstance(lbl, Tensor) else int(lbl)
        labels_t.append(torch.tensor(lbl_val).view(1))
        used_indices.append(ds_idx)

    if not images_t:
        return {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}

    images_batch = torch.cat(images_t, dim=0)
    labels_batch = torch.cat(labels_t, dim=0)
    model = adapter.get_model()
    target_layers = adapter.get_target_layers()
    device = next(model.parameters()).device
    model.eval()

    maps = generate_saliency_maps(
        model,
        images_batch.to(device),
        labels_batch.to(device),
        target_layers,
        method=xai_method,
    )

    imgs_np: list[np.ndarray] = []
    for img in images_batch:
        hwc = img.detach().cpu().float().permute(1, 2, 0).numpy()
        mn, mx = float(hwc.min()), float(hwc.max())
        if mx > mn:
            hwc = (hwc - mn) / (mx - mn)
        imgs_np.append(np.clip(hwc * 255.0, 0.0, 255.0).astype(np.uint8))

    images_arr = np.stack(imgs_np, axis=0)
    paths = save_xai_visualization(images_arr, maps, output_dir, prefix="attention")
    per_sample = [analyze_saliency_map(m) for m in maps]

    def _mean(key: str) -> float:
        vals = [float(s.get(key, 0.0)) for s in per_sample]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    aggregate = {
        "mean_coverage": _mean("coverage"),
        "mean_edge_ratio": _mean("edge_ratio"),
        "mean_entropy": _mean("entropy"),
        "mean_gini": _mean("gini"),
    }

    manifest = {
        "method": xai_method,
        "sample_indices": used_indices,
        "aggregate_stats": aggregate,
        "per_sample_stats": per_sample,
        "overlays": [str(p.relative_to(output_dir)) for p in paths],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "xai_dir": _rel(output_dir),
        "overlay_paths": [_rel(p) for p in paths],
        "sample_indices": used_indices,
        "aggregate_stats": aggregate,
    }


def _run_plain_epochs(
    *,
    condition: ConditionSpec,
    cfg: Any,
    adapter: Any,
    train_loader: Any,
    val_loader: Any,
    augmentations: list[Any],
    run_dir: Path,
    bench: dict[str, Any],
    extra_meta: dict[str, Any] | None = None,
    export_xai: bool = True,
) -> dict[str, Any]:
    from bnnr.core import BNNRTrainer
    from bnnr.training.loop import evaluate, train_epoch
    from bnnr.utils import set_seed

    set_seed(cfg.seed)
    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)

    t0 = time.perf_counter()
    best_val: dict[str, float] = {}
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    sel = cfg.selection_metric

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    header = (
        f"\n{'='*60}\n"
        f"  {condition.id.upper()}\n"
        f"  batch_augs={[a.name for a in augmentations] if augmentations else '(none)'}\n"
        f"  m_epochs={cfg.m_epochs}  seed={cfg.seed}  device={cfg.device}\n"
        f"{'='*60}"
    )
    print(header, flush=True)
    log_path.write_text(header + "\n", encoding="utf-8")

    for epoch in range(1, cfg.m_epochs + 1):
        train_epoch(trainer, train_loader, augmentations=augmentations)
        val_metrics = evaluate(trainer, val_loader)
        epoch_end_fn = getattr(trainer.model, "epoch_end", None)
        if callable(epoch_end_fn):
            epoch_end_fn()
        score = float(val_metrics.get(sel, 0.0))
        line = (
            f"  epoch {epoch}/{cfg.m_epochs} — {sel}={score:.4f}  "
            f"loss={val_metrics.get('loss', 0):.4f}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if not best_val or score >= best_val.get(sel, -1.0):
            best_val = copy.deepcopy(val_metrics)
            best_epoch = epoch
            best_state = copy.deepcopy(trainer.model.state_dict())

    if best_state is not None:
        trainer.model.load_state_dict(best_state)

    elapsed_s = time.perf_counter() - t0
    if export_xai:
        xai_indices = list(bench.get("xai_val_indices") or [0, 127, 255, 512])
        xai_method = str(bench.get("xai_method") or cfg.xai_method or "opticam")
        xai_meta = export_attention_maps(
            adapter,
            val_loader,
            sample_indices=xai_indices,
            output_dir=run_dir / "xai",
            xai_method=xai_method,
        )
    else:
        xai_meta = {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}

    summary_path = run_dir / "run_summary.json"
    payload = {
        "condition": condition.id,
        "best_epoch": best_epoch,
        "best_metrics": best_val,
        "augmentation_names": [a.name for a in augmentations] if augmentations else [],
        "xai": xai_meta,
        **(extra_meta or {}),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    aug_label = "+".join(a.name for a in augmentations) if augmentations else "none"
    return _result_entry(
        condition=condition,
        cfg=cfg,
        best_val=best_val,
        best_epoch=best_epoch,
        elapsed_s=elapsed_s,
        run_dir=run_dir,
        report_path=summary_path,
        best_path=f"{condition.strategy}:{aug_label}",
        xai_meta=xai_meta,
        extra=extra_meta,
    )


def _result_entry(
    *,
    condition: ConditionSpec,
    cfg: Any,
    best_val: dict[str, float],
    best_epoch: int | None,
    elapsed_s: float,
    run_dir: Path,
    report_path: Path,
    best_path: str,
    xai_meta: dict[str, Any],
    baseline_val: float | None = None,
    gain_pp: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sel = cfg.selection_metric
    agg = xai_meta.get("aggregate_stats") or {}
    return {
        "condition": condition.id,
        "strategy": condition.strategy,
        "dataset": "cifar10",
        "seed": cfg.seed,
        "max_iterations": condition.max_iterations,
        "m_epochs": cfg.m_epochs,
        "selection_metric": sel,
        "val_metric": float(best_val.get(sel, 0.0)),
        "f1_macro": float(best_val.get("f1_macro", 0.0)),
        "loss": float(best_val.get("loss", 0.0)),
        "best_epoch": best_epoch,
        "baseline_val_metric": baseline_val,
        "gain_vs_within_run_baseline_pp": gain_pp,
        "best_path": best_path,
        "augmentation_names": list(condition.augmentation_names),
        "selected_augmentations": list(condition.augmentation_names),
        "attention_coverage": agg.get("mean_coverage"),
        "attention_edge_ratio": agg.get("mean_edge_ratio"),
        "xai_dir": xai_meta.get("xai_dir"),
        "xai_overlays": xai_meta.get("overlay_paths") or [],
        "wall_clock_s": round(elapsed_s, 1),
        "report_json": _rel(report_path),
        "run_dir": _rel(run_dir),
        **(extra or {}),
    }


def run_no_bnnr(
    *,
    condition: ConditionSpec,
    seed: int,
    device: str,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = _make_run_dir(output_root, condition.id, seed)
    base_cfg, bench = load_training_config(config_path)
    cfg = _run_config(base_cfg, seed=seed, device=device, run_dir=run_dir, xai=False)

    adapter, train_loader, val_loader, _ = _build_cifar10_pipeline(cfg=cfg)
    return _run_plain_epochs(
        condition=condition,
        cfg=cfg,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=[],
        run_dir=run_dir,
        bench=bench,
    )


def run_randaugment(
    *,
    condition: ConditionSpec,
    seed: int,
    device: str,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = _make_run_dir(output_root, condition.id, seed)
    base_cfg, bench = load_training_config(config_path)
    num_ops = int(bench.get("randaugment_num_ops", 2))
    magnitude = int(bench.get("randaugment_magnitude", 9))

    cfg = _run_config(base_cfg, seed=seed, device=device, run_dir=run_dir, xai=False)
    adapter, train_loader, val_loader = _build_cifar10_randaugment_loaders(
        cfg, num_ops=num_ops, magnitude=magnitude,
    )
    meta = {
        "randaugment_num_ops": num_ops,
        "randaugment_magnitude": magnitude,
        "train_transform": (
            f"RandomCrop(32,4)+RandomHorizontalFlip+RandAugment({num_ops},{magnitude})+ToTensor"
        ),
    }
    return _run_plain_epochs(
        condition=condition,
        cfg=cfg,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=[],
        run_dir=run_dir,
        bench=bench,
        extra_meta=meta,
    )


def run_bnnr_branch_search(
    *,
    condition: ConditionSpec,
    seed: int,
    device: str,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    from bnnr.core import BNNRTrainer

    run_dir = _make_run_dir(output_root, condition.id, seed)
    base_cfg, bench = load_training_config(config_path)
    cfg = _run_config(
        base_cfg,
        seed=seed,
        device=device,
        run_dir=run_dir,
        xai=True,
        max_iterations=condition.max_iterations,
    )

    adapter, train_loader, val_loader, _ = _build_cifar10_pipeline(cfg=cfg)
    target_layers = adapter.get_target_layers()
    augmentations = build_bnnr_candidate_augmentations(
        adapter.get_model(), target_layers, seed,
    )

    print(
        f"\n{'='*60}\n"
        f"  BNNR BRANCH SEARCH\n"
        f"  candidates={[a.name for a in augmentations]}\n"
        f"  max_iterations={condition.max_iterations}  m_epochs={cfg.m_epochs}\n"
        f"  seed={seed}  device={device}\n"
        f"{'='*60}",
        flush=True,
    )

    t0 = time.perf_counter()
    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
    result = trainer.run()
    elapsed_s = time.perf_counter() - t0

    sel = cfg.selection_metric
    best = float(result.best_metrics.get(sel, 0.0))
    baseline_val = _extract_baseline_from_report(result.report_json_path, sel)
    gain: float | None = None
    if baseline_val is not None:
        gain = round((best - baseline_val) * 100, 2)

    selected = [part for part in (result.best_path or "").split("+") if part and part != "baseline"]
    if not selected and result.best_path and result.best_path != "baseline":
        selected = [result.best_path]

    xai_indices = list(bench.get("xai_val_indices") or [0, 127, 255, 512])
    xai_method = str(bench.get("xai_method") or cfg.xai_method or "opticam")
    xai_meta = export_attention_maps(
        adapter,
        val_loader,
        sample_indices=xai_indices,
        output_dir=run_dir / "xai",
        xai_method=xai_method,
    )

    return _result_entry(
        condition=condition,
        cfg=cfg,
        best_val=result.best_metrics,
        best_epoch=None,
        elapsed_s=elapsed_s,
        run_dir=run_dir,
        report_path=result.report_json_path,
        best_path=result.best_path,
        xai_meta=xai_meta,
        baseline_val=baseline_val,
        gain_pp=gain,
        extra={
            "selected_augmentations": selected,
            "augmentation_names": [a.name for a in augmentations],
        },
    )


def _make_run_dir(output_root: Path, condition_id: str, seed: int) -> Path:
    run_dir = output_root / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{condition_id}_s{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_config(
    base_cfg: Any,
    *,
    seed: int,
    device: str,
    run_dir: Path,
    xai: bool,
    max_iterations: int = 0,
) -> Any:
    return base_cfg.model_copy(
        update={
            "max_iterations": max_iterations,
            "seed": seed,
            "device": _resolve_device(device),
            "checkpoint_dir": run_dir / "checkpoints",
            "report_dir": run_dir / "reports",
            "event_log_enabled": False,
            "save_checkpoints": False,
            "xai_enabled": xai,
        }
    )


def run_condition(
    *,
    condition_id: str,
    dataset: str,
    seed: int,
    device: str,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if dataset != "cifar10":
        raise ValueError(f"Only cifar10 is supported, got {dataset!r}")

    spec = CONDITIONS[condition_id]
    if spec.strategy == "plain_training":
        return run_no_bnnr(
            condition=spec, seed=seed, device=device,
            config_path=config_path, output_root=output_root,
        )
    if spec.strategy == "randaugment":
        return run_randaugment(
            condition=spec, seed=seed, device=device,
            config_path=config_path, output_root=output_root,
        )
    return run_bnnr_branch_search(
        condition=spec, seed=seed, device=device,
        config_path=config_path, output_root=output_root,
    )
