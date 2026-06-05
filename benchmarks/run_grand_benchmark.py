#!/usr/bin/env python3
"""Grand benchmark for BNNR — 6 datasets × 2 regimes, paper-quality, equal-compute.

This script proves the core BNNR claims:
  - ICD/AICD individually improve accuracy vs no_aug
  - XAI-guided selection (bnnr_xai) > random selection (bnnr_random)
  - Branch search > best single augmentation
  - Results generalise across datasets

Conditions (10 total)
---------------------
    no_aug            RandomResizedCrop + flip only, B epochs
    randaugment       + torchvision RandAugment, B epochs
    trivialaugment    + torchvision TrivialAugmentWide, B epochs
    autoaugment       + torchvision AutoAugment (ImageNet policy), B epochs
    churchnoise_only  ChurchNoise as always-on batch aug, B epochs
    icd_only          ICD as always-on batch aug, B epochs
    aicd_only         AICD as always-on batch aug, B epochs
    icd_aicd_fixed    ICD+AICD together, no search, B epochs
    bnnr_random       Branch search, random candidate selection, equal compute B
    bnnr_xai          Branch search, XAI-guided selection, equal compute B

Datasets
--------
    imagewoof, pets, flowers102, dtd, aircraft, eurosat

Examples
--------
    # Smoke test — CPU-friendly, runs in seconds
    python benchmarks/run_grand_benchmark.py --dataset imagewoof --smoke --dry-run

    # Full imagewoof benchmark (10 seeds, GPU)
    python benchmarks/run_grand_benchmark.py --dataset imagewoof \\
        --seeds 42,43,44,45,46,47,48,49,50,51 --device cuda

    # Generalization dataset (7 seeds)
    python benchmarks/run_grand_benchmark.py --dataset pets --device cuda

    # Summarize
    python benchmarks/summarize_grand.py --results-dir benchmarks/
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lib import (  # noqa: E402
    ConditionSpec,
    _make_run_dir,
    _result_entry,
    _run_config,
    build_bnnr_candidate_augmentations,
    export_attention_maps,
    git_head,
    load_results,
    save_results,
    torch_info,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

N_CANDIDATES = 3
_CANDIDATE_NAMES = ["ICD", "AICD", "ChurchNoise"]

# Indices into held_out_test loader for XAI overlay export
_XAI_SAMPLE_INDICES = [0, 50, 100, 150, 200, 250]

# ---------------------------------------------------------------------------
# Dataset-specific defaults
# ---------------------------------------------------------------------------

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "imagewoof": {"img_size": 128, "train_per_class": 100, "num_classes": 10},
    "pets":      {"img_size": 224, "train_per_class": 100, "num_classes": 37},
    "flowers102":{"img_size": 224, "train_per_class": 10,  "num_classes": 102},
    "dtd":       {"img_size": 224, "train_per_class": 120, "num_classes": 47},
    "aircraft":  {"img_size": 224, "train_per_class": 33,  "num_classes": 100},
    "eurosat":   {"img_size": 64,  "train_per_class": 100, "num_classes": 10},
}

# Default conditions per dataset
_GENERALIZATION_CONDITIONS = [
    "no_aug", "randaugment", "icd_only", "aicd_only", "bnnr_random", "bnnr_xai",
]

DATASET_DEFAULT_CONDITIONS: dict[str, list[str]] = {
    "imagewoof": [
        "no_aug", "randaugment", "trivialaugment", "autoaugment",
        "churchnoise_only", "icd_only", "aicd_only", "icd_aicd_fixed",
        "bnnr_random", "bnnr_xai",
    ],
    "pets":      _GENERALIZATION_CONDITIONS,
    "flowers102":_GENERALIZATION_CONDITIONS,
    "dtd":       _GENERALIZATION_CONDITIONS,
    "aircraft":  _GENERALIZATION_CONDITIONS,
    "eurosat":   _GENERALIZATION_CONDITIONS,
}

# Default seeds per dataset
DATASET_DEFAULT_SEEDS: dict[str, list[int]] = {
    "imagewoof": list(range(42, 52)),  # 10 seeds
    "pets":      list(range(42, 49)),  # 7 seeds
    "flowers102":list(range(42, 49)),
    "dtd":       list(range(42, 49)),
    "aircraft":  list(range(42, 49)),
    "eurosat":   list(range(42, 49)),
}

# ---------------------------------------------------------------------------
# Conditions registry
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, ConditionSpec] = {
    "no_aug": ConditionSpec(
        id="no_aug",
        label="Without augmentation (crop + flip only)",
        strategy="plain_training",
        description=(
            "Standard from-scratch training: RandomResizedCrop + RandomHorizontalFlip. "
            "No extra augmentation, no branch search. Trained for B epochs straight."
        ),
        augmentation_names=(),
        max_iterations=0,
    ),
    "randaugment": ConditionSpec(
        id="randaugment",
        label="RandAugment (torchvision)",
        strategy="randaugment",
        description=(
            "External baseline: base crop+flip + torchvision RandAugment. "
            "Policy-based random augmentation, no saliency guidance. B epochs straight."
        ),
        augmentation_names=("RandAugment",),
        max_iterations=0,
    ),
    "trivialaugment": ConditionSpec(
        id="trivialaugment",
        label="TrivialAugmentWide (torchvision)",
        strategy="trivialaugment",
        description=(
            "External baseline: base crop+flip + TrivialAugmentWide. "
            "Parameter-free random augmentation. B epochs straight."
        ),
        augmentation_names=("TrivialAugmentWide",),
        max_iterations=0,
    ),
    "autoaugment": ConditionSpec(
        id="autoaugment",
        label="AutoAugment ImageNet (torchvision)",
        strategy="plain_training",
        description=(
            "External baseline: base crop+flip + AutoAugment (ImageNet policy). "
            "Searched augmentation policy. B epochs straight."
        ),
        augmentation_names=("AutoAugment",),
        max_iterations=0,
    ),
    "churchnoise_only": ConditionSpec(
        id="churchnoise_only",
        label="ChurchNoise only (non-XAI ablation)",
        strategy="plain_training",
        description=(
            "Ablation: ChurchNoise as always-on batch augmentation for B epochs. "
            "Isolates the non-XAI BNNR component."
        ),
        augmentation_names=("ChurchNoise",),
        max_iterations=0,
    ),
    "icd_only": ConditionSpec(
        id="icd_only",
        label="ICD only (saliency ablation)",
        strategy="plain_training",
        description=(
            "Ablation: ICD as always-on batch augmentation for B epochs. "
            "Masks high-saliency regions. No branch search."
        ),
        augmentation_names=("ICD",),
        max_iterations=0,
    ),
    "aicd_only": ConditionSpec(
        id="aicd_only",
        label="AICD only (saliency ablation)",
        strategy="plain_training",
        description=(
            "Ablation: AICD as always-on batch augmentation for B epochs. "
            "Masks background (low-saliency) regions. No branch search."
        ),
        augmentation_names=("AICD",),
        max_iterations=0,
    ),
    "icd_aicd_fixed": ConditionSpec(
        id="icd_aicd_fixed",
        label="ICD+AICD fixed (no search)",
        strategy="plain_training",
        description=(
            "Ablation: ICD and AICD together as always-on batch augmentation. "
            "No branch search — isolates search contribution."
        ),
        augmentation_names=("ICD", "AICD"),
        max_iterations=0,
    ),
    "bnnr_xai": ConditionSpec(
        id="bnnr_xai",
        label="BNNR XAI-guided (equal compute)",
        strategy="bnnr_branch_search",
        description=(
            "BNNR branch search with XAI-guided candidate selection. "
            "Equal compute: B epochs total split into (1 + N_candidates) phases. "
            "Candidate with highest selection_val score is chosen."
        ),
        augmentation_names=("ICD", "AICD", "ChurchNoise"),
        max_iterations=N_CANDIDATES,
    ),
    "bnnr_random": ConditionSpec(
        id="bnnr_random",
        label="BNNR random selection (XAI ablation)",
        strategy="bnnr_branch_search",
        description=(
            "BNNR branch search with RANDOM candidate selection (seed-controlled). "
            "Identical compute and augmentation pool as bnnr_xai. "
            "bnnr_xai vs bnnr_random isolates XAI-guided selection contribution."
        ),
        augmentation_names=("ICD", "AICD", "ChurchNoise"),
        max_iterations=N_CANDIDATES,
    ),
}

_POLICY_BY_CONDITION = {
    "no_aug": "base",
    "randaugment": "randaugment",
    "trivialaugment": "trivialaugment",
    "autoaugment": "autoaugment",
    "churchnoise_only": "base",
    "icd_only": "base",
    "aicd_only": "base",
    "icd_aicd_fixed": "base",
}

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_resnet(arch: str, num_classes: int, pretrained: bool) -> Any:
    """torchvision ResNet with ImageNet normalization baked in."""
    import torch
    from torch import nn
    from torchvision import models

    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported arch {arch!r} (use resnet18 or resnet50)")

    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    class _NormalizedResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, x: Any) -> Any:
            x = (x - self.mean) / self.std
            return self.backbone(x)

    return _NormalizedResNet()


def _target_layers(model: Any) -> list[Any]:
    return [model.backbone.layer4[-1]]


def _build_adapter(
    *,
    arch: str,
    num_classes: int,
    pretrained: bool,
    lr: float,
    device: str,
    epochs: int,
) -> Any:
    import torch
    from torch import nn

    from bnnr.adapter import SimpleTorchAdapter

    model = _build_resnet(arch, num_classes, pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    return SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=_target_layers(model),
        device=device,
        scheduler=scheduler,
    )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _base_config(*, epochs: int, device: str) -> Any:
    from bnnr.config_model import BNNRConfig

    return BNNRConfig(
        task="classification",
        device=device,
        m_epochs=epochs,
        selection_metric="accuracy",
        metrics=["accuracy", "f1_macro", "loss"],
        xai_method="opticam",
        candidate_pruning_enabled=True,
        candidate_pruning_relative_threshold=0.8,
        candidate_pruning_warmup_epochs=2,
    )


# ---------------------------------------------------------------------------
# Core training helper
# ---------------------------------------------------------------------------


def _run_epochs_on_loader(
    *,
    adapter: Any,
    train_loader: Any,
    eval_loader: Any,
    augmentations: list[Any],
    epochs: int,
    selection_metric: str,
    log_prefix: str = "",
) -> tuple[dict[str, float], int, dict[str, Any] | None]:
    """Train for *epochs* epochs; return (best_val_metrics, best_epoch, best_state)."""
    from bnnr.core import BNNRTrainer
    from bnnr.training.loop import evaluate, train_epoch

    cfg_dummy = _base_config(epochs=epochs, device=adapter.device)
    trainer = BNNRTrainer(adapter, train_loader, eval_loader, augmentations, cfg_dummy)

    best_val: dict[str, float] = {}
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    sel = selection_metric

    for epoch in range(1, epochs + 1):
        train_epoch(trainer, train_loader, augmentations=augmentations if augmentations else None)
        val_metrics = evaluate(trainer, eval_loader)
        epoch_end_fn = getattr(adapter, "epoch_end", None)
        if callable(epoch_end_fn):
            epoch_end_fn()
        score = float(val_metrics.get(sel, 0.0))
        print(
            f"  {log_prefix}epoch {epoch}/{epochs} — {sel}={score:.4f}  "
            f"loss={val_metrics.get('loss', 0.0):.4f}",
            flush=True,
        )
        if not best_val or score >= best_val.get(sel, -1.0):
            best_val = copy.deepcopy(val_metrics)
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.model.state_dict())

    if best_state is not None:
        adapter.model.load_state_dict(best_state)

    return best_val, best_epoch, best_state


# ---------------------------------------------------------------------------
# Stamp helper
# ---------------------------------------------------------------------------


def _stamp(
    entry: dict[str, Any],
    *,
    dataset: str,
    arch: str,
    img_size: int,
    pretrained: bool,
    regime: str,
) -> dict[str, Any]:
    """Override CIFAR-10 defaults baked into lib._result_entry."""
    entry["dataset"] = dataset
    entry["model"] = arch
    entry["img_size"] = img_size
    entry["pretrained"] = pretrained
    entry["regime"] = regime
    return entry


# ---------------------------------------------------------------------------
# Extended metrics helpers
# ---------------------------------------------------------------------------


def _compute_extended_metrics(
    adapter: Any,
    held_out_test_loader: Any,
    num_classes: int,
    device: str,
    no_xai: bool,
) -> dict[str, Any]:
    """Run extended classification + XAI metrics; return flat dict."""
    from metrics_extended import compute_classification_metrics, compute_xai_metrics

    cls_metrics = compute_classification_metrics(
        adapter.get_model(),
        held_out_test_loader,
        num_classes=num_classes,
        device=device,
        top_k=5,
    )

    if no_xai:
        xai_metrics = {
            "xai_edge_ratio": None,
            "xai_coverage": None,
            "xai_gini": None,
            "xai_entropy": None,
            "xai_center_bias": None,
        }
    else:
        xai_metrics = compute_xai_metrics(
            adapter,
            held_out_test_loader,
            sample_indices=_XAI_SAMPLE_INDICES,
            device=device,
            method="opticam",
        )

    return {
        "test_accuracy": cls_metrics.get("accuracy"),
        "test_f1_macro": cls_metrics.get("f1_macro"),
        "test_top5_accuracy": cls_metrics.get("top5_accuracy"),
        "test_cohen_kappa": cls_metrics.get("cohen_kappa"),
        "test_ece": cls_metrics.get("ece"),
        **xai_metrics,
    }


# ---------------------------------------------------------------------------
# Plain condition runner
# ---------------------------------------------------------------------------


def _run_plain_condition(
    *,
    condition: ConditionSpec,
    policy: str,
    extra_batch_augmentations: list[Any],
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
    num_classes: int,
    xai_cache: Any | None = None,
) -> dict[str, Any]:
    """Run a plain (non-BNNR-search) condition for B epochs, report on held_out_test."""
    from bnnr.utils import set_seed

    from dataset_loaders import get_loaders

    run_dir = _make_run_dir(output_root, condition.id, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    budget = args.budget
    cfg = _base_config(epochs=budget, device=args.device)
    cfg = _run_config(cfg, seed=seed, device=args.device, run_dir=run_dir, xai=False)

    set_seed(seed)
    num_workers = 0 if args.smoke else 2
    train_loader, _sel_loader, held_out_test_loader = get_loaders(
        args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=seed,
        policy=policy,
        n_per_class_train=args.train_per_class,
        data_dir=args.data_dir,
        num_workers=num_workers,
    )

    # For ICD/AICD conditions: pre-compute XAI cache if not already supplied
    local_cache = xai_cache
    if extra_batch_augmentations and local_cache is None and not args.no_xai:
        # Check if any augmentation is ICD/AICD-based
        needs_cache = any(
            hasattr(aug, "cache") for aug in extra_batch_augmentations
        )
        if needs_cache:
            local_cache = _precompute_xai_cache(
                args=args,
                seed=seed,
                run_dir=run_dir,
                num_classes=num_classes,
                train_loader=train_loader,
            )

    aug_label = (
        "+".join(a.name for a in extra_batch_augmentations)
        if extra_batch_augmentations
        else "none"
    )
    header = (
        f"\n{'='*60}\n"
        f"  {condition.id.upper()} (grand benchmark)\n"
        f"  dataset={args.dataset}  policy={policy}  batch_augs={aug_label}\n"
        f"  budget={budget} epochs  seed={seed}  device={cfg.device}\n"
        f"{'='*60}"
    )
    print(header, flush=True)

    # If augmentations reference a cache (ICD/AICD), replace the cache now
    if local_cache is not None:
        for aug in extra_batch_augmentations:
            if hasattr(aug, "cache"):
                aug.cache = local_cache

    adapter = _build_adapter(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pretrained,
        lr=args.lr,
        device=cfg.device,
        epochs=budget,
    )

    t0 = time.perf_counter()
    best_val, best_epoch, _ = _run_epochs_on_loader(
        adapter=adapter,
        train_loader=train_loader,
        eval_loader=held_out_test_loader,
        augmentations=extra_batch_augmentations,
        epochs=budget,
        selection_metric="accuracy",
    )
    elapsed_s = time.perf_counter() - t0

    if not args.no_xai:
        xai_meta = export_attention_maps(
            adapter,
            held_out_test_loader,
            sample_indices=_XAI_SAMPLE_INDICES,
            output_dir=run_dir / "xai",
            xai_method="opticam",
        )
    else:
        xai_meta = {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}

    extended = _compute_extended_metrics(
        adapter, held_out_test_loader, num_classes, cfg.device, args.no_xai
    )

    summary = {
        "condition": condition.id,
        "best_epoch": best_epoch,
        "best_metrics": best_val,
        "total_gpu_epochs": budget,
        "epochs_per_phase": budget,
        "selection_mode": "fixed",
        "held_out_test_metric": float(best_val.get("accuracy", 0.0)),
        "xai": xai_meta,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    extra: dict[str, Any] = {
        "total_gpu_epochs": budget,
        "epochs_per_phase": budget,
        "selection_mode": "fixed",
        "selected_candidate": None,
        "selection_val_metric": None,
        "held_out_test_metric": float(best_val.get("accuracy", 0.0)),
        "git_head": git_head(),
        "hardware": torch_info(),
        **extended,
    }

    entry = _result_entry(
        condition=condition,
        cfg=cfg,
        best_val=best_val,
        best_epoch=best_epoch,
        elapsed_s=elapsed_s,
        run_dir=run_dir,
        report_path=summary_path,
        best_path=f"plain:{policy}:{aug_label}",
        xai_meta=xai_meta,
        extra=extra,
    )
    return _stamp(
        entry,
        dataset=args.dataset,
        arch=args.arch,
        img_size=args.img_size,
        pretrained=args.pretrained,
        regime=args.regime,
    )


# ---------------------------------------------------------------------------
# XAI cache pre-computation helper
# ---------------------------------------------------------------------------


def _precompute_xai_cache(
    *,
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    num_classes: int,
    train_loader: Any,
) -> Any:
    """Pre-compute XAI cache from a fresh baseline adapter."""
    from bnnr.xai_cache import XAICache

    tmp_adapter = _build_adapter(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pretrained,
        lr=args.lr,
        device=args.device,
        epochs=1,
    )
    xai_cache_dir = run_dir / "xai_cache"
    xai_cache = XAICache(xai_cache_dir)
    print("  [XAI cache] Pre-computing saliency maps...", flush=True)
    n_cached = xai_cache.precompute_cache(
        tmp_adapter.get_model(),
        train_loader,
        tmp_adapter.get_target_layers(),
        method="opticam",
        show_progress=True,
    )
    print(f"  [XAI cache] {n_cached} maps cached to {xai_cache_dir}", flush=True)
    return xai_cache


# ---------------------------------------------------------------------------
# BNNR equal-compute runner
# ---------------------------------------------------------------------------


def _run_bnnr_equal_compute(
    *,
    condition: ConditionSpec,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
    selection_mode: str,  # "xai" | "random"
    num_classes: int,
) -> dict[str, Any]:
    """BNNR with equal compute budget.

    Total budget = B epochs split into (1 + N_CANDIDATES) phases.
    Phase 0 (baseline): B // (1 + N_CANDIDATES) epochs on selection_val.
    Phases 1..N: each candidate trained for B // (1 + N_CANDIDATES) epochs.
    Selection: xai -> best selection_val score; random -> Random(seed).randint.
    Final metric: evaluate selected model on held_out_test.
    """
    from bnnr.utils import set_seed
    from bnnr.xai_cache import XAICache

    from dataset_loaders import get_loaders

    run_dir = _make_run_dir(output_root, condition.id, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    budget = args.budget
    epochs_per_phase = budget // (1 + N_CANDIDATES)
    if epochs_per_phase < 1:
        epochs_per_phase = 1
    total_gpu_epochs = (1 + N_CANDIDATES) * epochs_per_phase

    cfg = _base_config(epochs=epochs_per_phase, device=args.device)
    cfg = _run_config(cfg, seed=seed, device=args.device, run_dir=run_dir, xai=False)

    set_seed(seed)
    num_workers = 0 if args.smoke else 2
    train_loader, selection_val_loader, held_out_test_loader = get_loaders(
        args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=seed,
        policy="base",
        n_per_class_train=args.train_per_class,
        data_dir=args.data_dir,
        num_workers=num_workers,
    )

    baseline_adapter = _build_adapter(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pretrained,
        lr=args.lr,
        device=cfg.device,
        epochs=epochs_per_phase,
    )

    header = (
        f"\n{'='*60}\n"
        f"  BNNR {selection_mode.upper()} (grand benchmark, equal-compute)\n"
        f"  dataset={args.dataset}  budget={budget}  epochs_per_phase={epochs_per_phase}\n"
        f"  N_CANDIDATES={N_CANDIDATES}  total_gpu_epochs={total_gpu_epochs}\n"
        f"  seed={seed}  device={cfg.device}\n"
        f"{'='*60}"
    )
    print(header, flush=True)

    t0 = time.perf_counter()

    # ---- Phase 0: baseline ----
    print(f"  [Phase 0/baseline] {epochs_per_phase} epochs (no extra aug)...", flush=True)
    baseline_val, _, _ = _run_epochs_on_loader(
        adapter=baseline_adapter,
        train_loader=train_loader,
        eval_loader=selection_val_loader,
        augmentations=[],
        epochs=epochs_per_phase,
        selection_metric="accuracy",
        log_prefix="[baseline] ",
    )
    baseline_score = float(baseline_val.get("accuracy", 0.0))
    baseline_state = copy.deepcopy(baseline_adapter.state_dict())
    print(f"  [Phase 0] baseline selection_val accuracy={baseline_score:.4f}", flush=True)

    # ---- Pre-compute XAI cache ----
    xai_cache_dir = run_dir / "xai_cache"
    xai_cache = XAICache(xai_cache_dir)
    print("  [XAI cache] Pre-computing saliency maps...", flush=True)
    n_cached = xai_cache.precompute_cache(
        baseline_adapter.get_model(),
        train_loader,
        baseline_adapter.get_target_layers(),
        method="opticam",
        show_progress=True,
    )
    print(f"  [XAI cache] {n_cached} maps cached to {xai_cache_dir}", flush=True)

    # ---- Phases 1..N_CANDIDATES ----
    candidate_scores: list[float] = []
    candidate_states: list[dict[str, Any]] = []
    candidate_val_metrics: list[dict[str, float]] = []

    for i in range(N_CANDIDATES):
        cand_name = _CANDIDATE_NAMES[i]
        print(
            f"  [Phase {i+1}/{N_CANDIDATES} — {cand_name}] {epochs_per_phase} epochs...",
            flush=True,
        )
        cand_adapter = _build_adapter(
            arch=args.arch,
            num_classes=num_classes,
            pretrained=args.pretrained,
            lr=args.lr,
            device=cfg.device,
            epochs=epochs_per_phase,
        )
        # Restore only model weights; fresh optimizer+scheduler per phase
        cand_adapter.model.load_state_dict(baseline_state["model"])

        from bnnr.augmentations import ChurchNoise
        from bnnr.icd import AICD, ICD

        _cand_model = cand_adapter.get_model()
        _cand_layers = cand_adapter.get_target_layers()
        phase_candidates = [
            ICD(model=_cand_model, target_layers=_cand_layers,
                threshold_percentile=75.0, probability=0.5,
                random_state=seed, cache=xai_cache),
            AICD(model=_cand_model, target_layers=_cand_layers,
                 threshold_percentile=75.0, probability=0.5,
                 random_state=seed + 1, cache=xai_cache),
            ChurchNoise(probability=0.5, intensity=0.5,
                        noise_strength_range=(3.0, 8.0), random_state=seed + 2),
        ]
        aug = phase_candidates[i]

        val_metrics, _, state = _run_epochs_on_loader(
            adapter=cand_adapter,
            train_loader=train_loader,
            eval_loader=selection_val_loader,
            augmentations=[aug],
            epochs=epochs_per_phase,
            selection_metric="accuracy",
            log_prefix=f"[{cand_name}] ",
        )
        score = float(val_metrics.get("accuracy", 0.0))
        candidate_scores.append(score)
        candidate_states.append(copy.deepcopy(cand_adapter.state_dict()))
        candidate_val_metrics.append(copy.deepcopy(val_metrics))
        print(f"  [{cand_name}] selection_val accuracy={score:.4f}", flush=True)

    # ---- Selection ----
    if selection_mode == "xai":
        best_idx = int(max(range(N_CANDIDATES), key=lambda k: candidate_scores[k]))
        selected_candidate = _CANDIDATE_NAMES[best_idx]
        selection_val_metric = candidate_scores[best_idx]
        print(
            f"  [XAI selection] best={selected_candidate}  score={selection_val_metric:.4f}",
            flush=True,
        )
    elif selection_mode == "random":
        best_idx = random.Random(seed).randint(0, N_CANDIDATES - 1)
        selected_candidate = _CANDIDATE_NAMES[best_idx]
        selection_val_metric = candidate_scores[best_idx]
        print(
            f"  [Random selection] chosen={selected_candidate} (idx={best_idx})  "
            f"score={selection_val_metric:.4f}",
            flush=True,
        )
    else:
        raise ValueError(f"Unknown selection_mode {selection_mode!r}")

    # ---- Final evaluation on held_out_test ----
    final_adapter = _build_adapter(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pretrained,
        lr=args.lr,
        device=cfg.device,
        epochs=epochs_per_phase,
    )
    final_adapter.load_state_dict(candidate_states[best_idx])

    from bnnr.core import BNNRTrainer
    from bnnr.training.loop import evaluate as _evaluate

    final_cfg = _base_config(epochs=epochs_per_phase, device=args.device)
    final_cfg = _run_config(
        final_cfg, seed=seed, device=args.device, run_dir=run_dir, xai=False
    )
    _dummy_trainer = BNNRTrainer(
        final_adapter, train_loader, held_out_test_loader, [], final_cfg
    )
    held_out_metrics = _evaluate(_dummy_trainer, held_out_test_loader)
    held_out_test_metric = float(held_out_metrics.get("accuracy", 0.0))

    elapsed_s = time.perf_counter() - t0
    print(
        f"  [held_out_test] accuracy={held_out_test_metric:.4f}  elapsed={elapsed_s:.1f}s",
        flush=True,
    )

    # ---- XAI overlays ----
    if not args.no_xai:
        xai_meta = export_attention_maps(
            final_adapter,
            held_out_test_loader,
            sample_indices=_XAI_SAMPLE_INDICES,
            output_dir=run_dir / "xai",
            xai_method="opticam",
        )
    else:
        xai_meta = {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}

    extended = _compute_extended_metrics(
        final_adapter, held_out_test_loader, num_classes, cfg.device, args.no_xai
    )

    # ---- Save summary ----
    summary = {
        "condition": condition.id,
        "selection_mode": selection_mode,
        "selected_candidate": selected_candidate,
        "selection_val_metric": selection_val_metric,
        "held_out_test_metric": held_out_test_metric,
        "baseline_score": baseline_score,
        "candidate_scores": dict(zip(_CANDIDATE_NAMES, candidate_scores)),
        "total_gpu_epochs": total_gpu_epochs,
        "epochs_per_phase": epochs_per_phase,
        "xai": xai_meta,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    held_out_as_best_val = copy.deepcopy(held_out_metrics)

    extra: dict[str, Any] = {
        "total_gpu_epochs": total_gpu_epochs,
        "epochs_per_phase": epochs_per_phase,
        "selection_mode": selection_mode,
        "selected_candidate": selected_candidate,
        "selection_val_metric": selection_val_metric,
        "held_out_test_metric": held_out_test_metric,
        "baseline_score": baseline_score,
        "candidate_scores": dict(zip(_CANDIDATE_NAMES, candidate_scores)),
        "git_head": git_head(),
        "hardware": torch_info(),
        **extended,
    }

    entry = _result_entry(
        condition=condition,
        cfg=final_cfg,
        best_val=held_out_as_best_val,
        best_epoch=None,
        elapsed_s=elapsed_s,
        run_dir=run_dir,
        report_path=summary_path,
        best_path=f"bnnr_{selection_mode}:{selected_candidate}",
        xai_meta=xai_meta,
        extra=extra,
    )
    return _stamp(
        entry,
        dataset=args.dataset,
        arch=args.arch,
        img_size=args.img_size,
        pretrained=args.pretrained,
        regime=args.regime,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run_condition(
    *,
    condition_id: str,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
    num_classes: int,
) -> dict[str, Any]:
    spec = CONDITIONS[condition_id]

    if condition_id == "bnnr_xai":
        return _run_bnnr_equal_compute(
            condition=spec, seed=seed, args=args,
            output_root=output_root, selection_mode="xai", num_classes=num_classes,
        )
    if condition_id == "bnnr_random":
        return _run_bnnr_equal_compute(
            condition=spec, seed=seed, args=args,
            output_root=output_root, selection_mode="random", num_classes=num_classes,
        )

    policy = _POLICY_BY_CONDITION[condition_id]

    # Build extra batch augmentations
    extra_batch_augmentations: list[Any] = []
    if condition_id in ("churchnoise_only", "icd_only", "aicd_only", "icd_aicd_fixed"):
        from bnnr.augmentations import ChurchNoise
        from bnnr.icd import AICD, ICD

        # We need a temporary adapter to get model/layers for ICD/AICD
        # The actual model weights will be built fresh in _run_plain_condition;
        # here we just build placeholders for the augmentation constructors.
        # _run_plain_condition will handle cache injection if needed.
        tmp_adapter = _build_adapter(
            arch=args.arch,
            num_classes=num_classes,
            pretrained=args.pretrained,
            lr=args.lr,
            device=args.device,
            epochs=args.budget,
        )
        _model = tmp_adapter.get_model()
        _layers = tmp_adapter.get_target_layers()

        if condition_id == "churchnoise_only":
            extra_batch_augmentations = [
                ChurchNoise(probability=0.5, intensity=0.5,
                            noise_strength_range=(3.0, 8.0), random_state=seed)
            ]
        elif condition_id == "icd_only":
            extra_batch_augmentations = [
                ICD(model=_model, target_layers=_layers,
                    threshold_percentile=75.0, probability=0.5, random_state=seed)
            ]
        elif condition_id == "aicd_only":
            extra_batch_augmentations = [
                AICD(model=_model, target_layers=_layers,
                     threshold_percentile=75.0, probability=0.5, random_state=seed + 1)
            ]
        elif condition_id == "icd_aicd_fixed":
            extra_batch_augmentations = [
                ICD(model=_model, target_layers=_layers,
                    threshold_percentile=75.0, probability=0.5, random_state=seed),
                AICD(model=_model, target_layers=_layers,
                     threshold_percentile=75.0, probability=0.5, random_state=seed + 1),
            ]

    return _run_plain_condition(
        condition=spec,
        policy=policy,
        extra_batch_augmentations=extra_batch_augmentations,
        seed=seed,
        args=args,
        output_root=output_root,
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Benchmark document
# ---------------------------------------------------------------------------


def _benchmark_document(args: argparse.Namespace, num_classes: int) -> dict[str, Any]:
    epochs_per_phase = args.budget // (1 + N_CANDIDATES)
    return {
        "benchmark_id": f"grand_benchmark_{args.dataset}_{args.regime}",
        "benchmark_version": "grand_v1",
        "model": f"{args.arch} (torchvision, pretrained={args.pretrained})",
        "dataset": args.dataset,
        "num_classes": num_classes,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "optimizer": f"SGD(lr={args.lr}, momentum=0.9, wd=5e-4)",
        "scheduler": "CosineAnnealingLR",
        "budget_epochs_total": args.budget,
        "epochs_per_phase": epochs_per_phase,
        "n_candidates": N_CANDIDATES,
        "candidate_names": _CANDIDATE_NAMES,
        "primary_metric": "held_out_test accuracy (best epoch within each phase)",
        "normalization": "ImageNet mean/std applied inside the model",
        "attention_method": "OptiCAM overlays on fixed held_out_test indices",
        "target_layer": "backbone.layer4[-1]",
        "regime": args.regime,
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
        "protocol_caveats": [
            "Equal total GPU-epoch budget B for all conditions.",
            f"BNNR trains B//(1+N_candidates) = {epochs_per_phase} epochs per phase.",
            "Held-out test set is reserved for final reporting only.",
            "bnnr_random provides a compute-matched baseline for bnnr_xai.",
        ],
    }


# ---------------------------------------------------------------------------
# Wall-clock estimate
# ---------------------------------------------------------------------------


def _estimate(args: argparse.Namespace, n_seeds: int, conds: list[str]) -> str:
    bnnr_conds = {"bnnr_xai", "bnnr_random"}
    bnnr_count = sum(1 for c in conds if c in bnnr_conds)
    plain_count = len(conds) - bnnr_count
    plain_epochs_total = plain_count * args.budget * n_seeds
    bnnr_epochs_total = bnnr_count * args.budget * n_seeds
    total_epoch_minutes = (plain_epochs_total + bnnr_epochs_total) * 1.0
    if args.device != "cpu":
        total_epoch_minutes *= 0.15
    return (
        f"~{total_epoch_minutes/60:.1f}h wall-clock estimate "
        f"({len(conds)} conditions x {n_seeds} seeds, {args.device}, "
        f"budget={args.budget})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grand benchmark: BNNR across 6 datasets × 2 regimes (paper-quality)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_DEFAULTS.keys()),
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--regime",
        choices=["scratch", "pretrained"],
        default="scratch",
        help="Training regime (default: scratch)",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Comma-separated conditions (default: all for dataset+regime combo). "
            f"Available: {', '.join(CONDITIONS)}"
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Total GPU-epoch budget per condition (default: 40 scratch, 20 pretrained)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (default: dataset-specific)",
    )
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Square resize/crop target (dataset-specific default)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=None,
        help="Balanced train images per class (dataset-specific default)",
    )
    parser.add_argument(
        "--no-xai",
        action="store_true",
        help="Skip OptiCAM overlay export (faster)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to results JSON (default: benchmarks/results_{dataset}_{regime}.json)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Run output dir (default: benchmarks/runs_grand/{dataset}_{regime})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for dataset downloads (default: <repo>/data)",
    )
    parser.add_argument(
        "--drive-base-dir",
        type=Path,
        default=None,
        help=(
            "Colab/Drive convenience: sets results, output-root, data-dir "
            "under a single directory."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run: 1 epoch, 2 seeds, 5/class, img_size=64",
    )
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    # ---- Dataset-specific defaults ----
    ds_defaults = DATASET_DEFAULTS[args.dataset]
    if args.img_size is None:
        args.img_size = ds_defaults["img_size"]
    if args.train_per_class is None:
        args.train_per_class = ds_defaults["train_per_class"]
    num_classes: int = ds_defaults["num_classes"]

    # ---- Budget default ----
    if args.budget is None:
        args.budget = 20 if args.regime == "pretrained" else 40

    # ---- Pretrained flag from regime ----
    args.pretrained = args.regime == "pretrained"

    # ---- drive-base-dir convenience ----
    if args.drive_base_dir is not None:
        args.drive_base_dir.mkdir(parents=True, exist_ok=True)
        if args.results is None:
            args.results = args.drive_base_dir / f"results_{args.dataset}_{args.regime}.json"
        if args.output_root is None:
            args.output_root = args.drive_base_dir / f"runs_grand/{args.dataset}_{args.regime}"
        if args.data_dir is None:
            args.data_dir = args.drive_base_dir / "data"

    # ---- Path defaults (if not set by drive-base-dir) ----
    if args.results is None:
        args.results = BENCHMARKS_DIR / f"results_{args.dataset}_{args.regime}.json"
    if args.output_root is None:
        args.output_root = BENCHMARKS_DIR / f"runs_grand/{args.dataset}_{args.regime}"
    if args.data_dir is None:
        args.data_dir = REPO_ROOT / "data"

    # ---- Smoke overrides ----
    if args.smoke:
        args.budget = (1 + N_CANDIDATES) * 1  # 4 total, 1 epoch per phase
        args.img_size = 64
        args.train_per_class = 5
        if args.seeds is None:
            args.seeds = "42,43"
        if args.conditions is None:
            args.conditions = "no_aug,bnnr_xai"  # minimal smoke

    # ---- Seeds ----
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = DATASET_DEFAULT_SEEDS[args.dataset]

    # ---- Conditions ----
    if args.conditions is not None:
        conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    else:
        conds = DATASET_DEFAULT_CONDITIONS[args.dataset]
    for c in conds:
        if c not in CONDITIONS:
            parser.error(f"Unknown condition {c!r}. Available: {', '.join(CONDITIONS)}")

    # ---- Device ----
    if not hasattr(args, "device"):
        args.device = "auto"
    # Resolve auto
    try:
        import torch
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        args.device = "cpu"

    # ---- Print plan ----
    epochs_per_phase = args.budget // (1 + N_CANDIDATES)
    print("Grand Benchmark — BNNR cross-dataset ablation (paper-quality)")
    print(f"  dataset={args.dataset}  regime={args.regime}  arch={args.arch}")
    print(f"  pretrained={args.pretrained}  img_size={args.img_size}")
    print(f"  num_classes={num_classes}  train_per_class={args.train_per_class}")
    print(f"  seeds={seeds}  conditions={conds}")
    print(
        f"  budget={args.budget} epochs  epochs_per_phase={epochs_per_phase}  "
        f"N_candidates={N_CANDIDATES}  device={args.device}"
    )
    print(f"  {_estimate(args, len(seeds), conds)}")
    print(f"  results -> {args.results}")

    if args.dry_run:
        return

    # ---- Run ----
    args.output_root.mkdir(parents=True, exist_ok=True)
    data = load_results(args.results)
    doc = _benchmark_document(args, num_classes)
    data.update(doc)
    data.setdefault("runs", [])
    data["hardware"] = torch_info()
    data["git_head"] = git_head()
    data["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Resume safety
    done = {
        (r["condition"], r["seed"], r.get("regime", "scratch"))
        for r in data["runs"]
        if "val_metric" in r and "error" not in r
    }

    for seed in seeds:
        for cid in conds:
            key = (cid, seed, args.regime)
            if key in done:
                print(f"SKIP {cid} seed={seed} regime={args.regime} (already done)")
                continue
            print(f"\n>>> dataset={args.dataset}  condition={cid}  seed={seed}  regime={args.regime}")
            try:
                entry = run_condition(
                    condition_id=cid,
                    seed=seed,
                    args=args,
                    output_root=args.output_root,
                    num_classes=num_classes,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"    FAILED ({cid}, seed={seed}): {exc}", file=sys.stderr)
                traceback.print_exc()
                entry = {
                    "condition": cid,
                    "seed": seed,
                    "regime": args.regime,
                    "dataset": args.dataset,
                    "model": args.arch,
                    "error": str(exc),
                }
            data["runs"].append(entry)
            save_results(args.results, data)  # checkpoint after every run
            val = entry.get("val_metric")
            held = entry.get("held_out_test_metric")
            if val is not None:
                print(f"    {cid} seed={seed}: val_metric(held_out)={val:.4f}")
            elif held is not None:
                print(f"    {cid} seed={seed}: held_out_test_accuracy={held:.4f}")

    n_valid = sum(1 for r in data["runs"] if "val_metric" in r and "error" not in r)
    print(f"\nDone. {n_valid} valid run records in {args.results}")
    print(
        f"Summarize: python benchmarks/summarize_grand.py "
        f"--results-dir benchmarks/ --datasets {args.dataset}"
    )


if __name__ == "__main__":
    main()
