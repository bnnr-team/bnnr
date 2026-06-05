#!/usr/bin/env python3
"""ResNet18 / Imagewoof augmentation benchmark v2 — paper-quality, equal-compute.

Key improvements over v1 (run_imagewoof.py):
  1. **Equal-compute**: All conditions share the same total GPU-epoch budget B.
     Baselines train B epochs straight. BNNR (xai/random) splits B into
     (1 + N_candidates) equal phases of B//(1+N_candidates) epochs each.
  2. **XAI ablation**: ``bnnr_random`` uses the same augmentation pool and
     compute as ``bnnr_xai`` but selects a candidate uniformly at random
     (seed-controlled). bnnr_xai vs bnnr_random isolates the contribution
     of XAI-guided selection.
  3. **Held-out test split**: The Imagewoof val set is split 50/50 per class
     (deterministic, sorted-index split). BNNR candidate selection uses only
     ``selection_val``; all final accuracy numbers are reported on ``held_out_test``.
  4. **ChurchNoise-only ablation**: trains B epochs with only ChurchNoise as
     an always-on batch augmentation, isolating the non-XAI component.
  5. **10 seeds** + proper statistics (see summarize_v2.py).

Conditions (6):
    no_aug            RandomResizedCrop + flip only, B epochs
    randaugment       + torchvision RandAugment, B epochs
    trivialaugment    + torchvision TrivialAugmentWide, B epochs
    churchnoise_only  ChurchNoise as always-on batch aug, B epochs
    bnnr_xai          branch search, XAI-guided selection, equal compute B total
    bnnr_random       branch search, random selection, equal compute B total

Examples
--------
    # Smoke test (CPU-friendly)
    python benchmarks/run_imagewoof_v2.py --smoke

    # Full benchmark, 10 seeds, GPU
    python benchmarks/run_imagewoof_v2.py \
        --seeds 42,43,44,45,46,47,48,49,50,51 --device cuda

    # Single condition / single seed
    python benchmarks/run_imagewoof_v2.py --conditions bnnr_xai --seeds 42 --device cuda

    # Summarize
    python benchmarks/summarize_v2.py --results benchmarks/results_imagewoof_v2.json --markdown
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

DEFAULT_RESULTS = BENCHMARKS_DIR / "results_imagewoof_v2.json"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "runs_imagewoof_v2"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 10

_IMAGEWOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
_IMAGEWOOF_DIRNAME = "imagewoof2-160"

# Indices into held_out_test (a ~50% split of Imagewoof val, ~1964 items).
# Must be < min held_out size. Imagewoof val has ~3929 images; 50% = ~1964.
_XAI_VAL_INDICES = [0, 300, 600, 900, 1200, 1500]

# Number of BNNR candidate augmentations (ICD, AICD, ChurchNoise)
N_CANDIDATES = 3

CONDITIONS: dict[str, ConditionSpec] = {
    "no_aug": ConditionSpec(
        id="no_aug",
        label="Without augmentation (crop + flip only)",
        strategy="plain_training",
        description=(
            "Standard from-scratch training: RandomResizedCrop + "
            "RandomHorizontalFlip. No extra augmentation, no branch search. "
            "Trained for B epochs straight."
        ),
        augmentation_names=(),
        max_iterations=0,
    ),
    "randaugment": ConditionSpec(
        id="randaugment",
        label="RandAugment (torchvision)",
        strategy="randaugment",
        description=(
            "External baseline: RandomResizedCrop + flip + torchvision "
            "RandAugment. Policy-based random augmentation, no saliency guidance. "
            "Trained for B epochs straight."
        ),
        augmentation_names=("RandAugment",),
        max_iterations=0,
    ),
    "trivialaugment": ConditionSpec(
        id="trivialaugment",
        label="TrivialAugmentWide (torchvision)",
        strategy="trivialaugment",
        description=(
            "External baseline: RandomResizedCrop + flip + torchvision "
            "TrivialAugmentWide. Parameter-free random augmentation, no saliency "
            "guidance. Trained for B epochs straight."
        ),
        augmentation_names=("TrivialAugmentWide",),
        max_iterations=0,
    ),
    "churchnoise_only": ConditionSpec(
        id="churchnoise_only",
        label="ChurchNoise only (non-XAI ablation)",
        strategy="plain_training",
        description=(
            "Ablation: ChurchNoise applied as an always-on batch augmentation "
            "for B epochs straight. Isolates the non-XAI BNNR component. "
            "No saliency guidance, no branch search."
        ),
        augmentation_names=("ChurchNoise",),
        max_iterations=0,
    ),
    "bnnr_xai": ConditionSpec(
        id="bnnr_xai",
        label="BNNR XAI-guided (equal compute)",
        strategy="bnnr_branch_search",
        description=(
            "BNNR branch search with XAI-guided candidate selection. "
            "Equal compute: B epochs total split into (1 + N_candidates) phases. "
            "Candidate with highest selection_val score is chosen. "
            "Final metric on held_out_test only."
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
            "bnnr_xai vs bnnr_random isolates the contribution of XAI-guided selection. "
            "Final metric on held_out_test only."
        ),
        augmentation_names=("ICD", "AICD", "ChurchNoise"),
        max_iterations=N_CANDIDATES,
    ),
}

_POLICY_BY_CONDITION = {
    "no_aug": "base",
    "randaugment": "randaugment",
    "trivialaugment": "trivialaugment",
}

_CANDIDATE_NAMES = ["ICD", "AICD", "ChurchNoise"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_resnet(arch: str, num_classes: int, pretrained: bool) -> Any:
    """torchvision ResNet with ImageNet normalization baked in.

    Inputs in [0, 1] (plain ToTensor); model normalizes internally so BNNR
    uint8-range augmentations stay compatible upstream.
    """
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


# ---------------------------------------------------------------------------
# Data: download + split
# ---------------------------------------------------------------------------


def _download_imagewoof(data_root: Path) -> Path:
    """Download + extract Imagewoof2 (160px) once; return dataset root."""
    from torchvision.datasets.utils import download_and_extract_archive

    target = data_root / _IMAGEWOOF_DIRNAME
    if (target / "train").is_dir() and (target / "val").is_dir():
        return target
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Imagewoof2 (160px) -> {data_root} ...", flush=True)
    download_and_extract_archive(_IMAGEWOOF_URL, download_root=str(data_root))
    if not (target / "train").is_dir():
        raise FileNotFoundError(
            f"Expected {target}/train after extraction; got {list(data_root.iterdir())}"
        )
    return target


def _balanced_subset(dataset: Any, n_per_class: int | None, seed: int) -> Any:
    """Return a class-balanced Subset with n_per_class samples each."""
    if n_per_class is None:
        return dataset

    from torch.utils.data import Subset

    by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(dataset.targets):
        by_class.setdefault(int(target), []).append(idx)

    rng = random.Random(seed)
    chosen: list[int] = []
    for _cls, idxs in sorted(by_class.items()):
        pool = idxs[:]
        rng.shuffle(pool)
        chosen.extend(pool[:n_per_class])
    chosen.sort()
    return Subset(dataset, chosen)


def _split_val(val_ds: Any) -> tuple[Any, Any]:
    """Split val dataset 50/50 per class into (selection_val, held_out_test).

    Deterministic: indices are sorted by class, then first 50% of each class
    goes to selection_val, last 50% to held_out_test. No randomness — always
    the same split for reproducibility.

    Returns
    -------
    (selection_val_subset, held_out_test_subset)
    """
    from torch.utils.data import Subset

    # Gather indices grouped by class, using sorted order (deterministic).
    by_class: dict[int, list[int]] = {}
    targets = getattr(val_ds, "targets", None)
    if targets is None:
        # Wrapped dataset (e.g. Subset): iterate through to collect targets
        targets = [val_ds[i][1] for i in range(len(val_ds))]  # type: ignore[arg-type]
    for idx, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(idx)

    selection_indices: list[int] = []
    held_out_indices: list[int] = []
    for _cls, idxs in sorted(by_class.items()):
        sorted_idxs = sorted(idxs)  # deterministic sort by dataset index
        split_point = len(sorted_idxs) // 2
        selection_indices.extend(sorted_idxs[:split_point])
        held_out_indices.extend(sorted_idxs[split_point:])

    selection_indices.sort()
    held_out_indices.sort()
    return Subset(val_ds, selection_indices), Subset(val_ds, held_out_indices)


def _imagewoof_loaders_v2(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path | None = None,
) -> tuple[Any, Any, Any]:
    """Return ``(train_loader, selection_val_loader, held_out_test_loader)``.

    The val set is split 50/50 per class (deterministic). Baselines only use
    held_out_test_loader for their final metric. BNNR conditions use
    selection_val_loader for candidate selection and held_out_test_loader for
    the final reported metric.

    ``policy`` is one of ``base`` | ``randaugment`` | ``trivialaugment``.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from bnnr.pipelines import _IndexedDataset

    root = _download_imagewoof(data_dir if data_dir is not None else REPO_ROOT / "data")

    base_train = [
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if policy == "randaugment":
        base_train.append(transforms.RandAugment())
    elif policy == "trivialaugment":
        base_train.append(transforms.TrivialAugmentWide())
    elif policy != "base":
        raise ValueError(f"Unknown policy {policy!r}")
    base_train.append(transforms.ToTensor())

    train_tf = transforms.Compose(base_train)
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )

    train_ds = ImageFolder(str(root / "train"), transform=train_tf)
    val_ds_full = ImageFolder(str(root / "val"), transform=val_tf)

    train_ds = _balanced_subset(train_ds, n_per_class_train, seed)
    selection_val_ds, held_out_test_ds = _split_val(val_ds_full)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _IndexedDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=generator,
    )
    selection_val_loader = DataLoader(
        _IndexedDataset(selection_val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    held_out_test_loader = DataLoader(
        _IndexedDataset(held_out_test_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, selection_val_loader, held_out_test_loader


# ---------------------------------------------------------------------------
# Adapter builder
# ---------------------------------------------------------------------------


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
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
# Core training helpers
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
    """Train for ``epochs`` epochs; return (best_val_metrics, best_epoch, best_state).

    Tracks the best epoch by ``selection_metric``. Restores best weights at end.
    """
    from bnnr.core import BNNRTrainer
    from bnnr.training.loop import evaluate, train_epoch
    from bnnr.utils import set_seed

    # We do NOT call set_seed here — callers manage global seed before this.
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
        line = (
            f"  {log_prefix}epoch {epoch}/{epochs} — {sel}={score:.4f}  "
            f"loss={val_metrics.get('loss', 0.0):.4f}"
        )
        print(line, flush=True)
        if not best_val or score >= best_val.get(sel, -1.0):
            best_val = copy.deepcopy(val_metrics)
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.model.state_dict())

    if best_state is not None:
        adapter.model.load_state_dict(best_state)

    return best_val, best_epoch, best_state


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------


def _stamp(
    entry: dict[str, Any],
    *,
    arch: str,
    img_size: int,
    pretrained: bool,
    regime: str,
) -> dict[str, Any]:
    """Override cifar10 defaults baked into lib._result_entry."""
    entry["dataset"] = "imagewoof"
    entry["model"] = arch
    entry["img_size"] = img_size
    entry["pretrained"] = pretrained
    entry["regime"] = regime
    return entry


def _run_plain_condition(
    *,
    condition: ConditionSpec,
    policy: str,
    extra_batch_augmentations: list[Any],
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    """Run a plain (non-BNNR) condition: train B epochs, report on held_out_test.

    ``extra_batch_augmentations`` are always-on BNNR-style batch augmentations
    (used for churchnoise_only); for all other plain conditions this is empty.
    """
    from bnnr.utils import set_seed

    run_dir = _make_run_dir(output_root, condition.id, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    budget = args.budget
    cfg = _base_config(epochs=budget, device=args.device)
    cfg = _run_config(cfg, seed=seed, device=args.device, run_dir=run_dir, xai=False)

    set_seed(seed)
    adapter = _build_adapter(
        arch=args.arch,
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        lr=args.lr,
        device=cfg.device,
        epochs=budget,
    )
    train_loader, _sel_loader, held_out_test_loader = _imagewoof_loaders_v2(
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=seed,
        policy=policy,
        n_per_class_train=args.train_per_class,
        data_dir=getattr(args, "data_dir", None),
    )

    aug_label = (
        "+".join(a.name for a in extra_batch_augmentations)
        if extra_batch_augmentations
        else "none"
    )
    header = (
        f"\n{'='*60}\n"
        f"  {condition.id.upper()} (v2 equal-compute)\n"
        f"  policy={policy}  batch_augs={aug_label}\n"
        f"  budget={budget} epochs  seed={seed}  device={cfg.device}\n"
        f"{'='*60}"
    )
    print(header, flush=True)

    t0 = time.perf_counter()
    best_val, best_epoch, _ = _run_epochs_on_loader(
        adapter=adapter,
        train_loader=train_loader,
        eval_loader=held_out_test_loader,
        augmentations=extra_batch_augmentations,
        epochs=budget,
        selection_metric="accuracy",
        log_prefix="",
    )
    elapsed_s = time.perf_counter() - t0

    if not args.no_xai:
        xai_meta = export_attention_maps(
            adapter,
            held_out_test_loader,
            sample_indices=_XAI_VAL_INDICES,
            output_dir=run_dir / "xai",
            xai_method="opticam",
        )
    else:
        xai_meta = {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}

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
    import json
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
    }
    if policy != "base":
        extra["augmentation_policy"] = policy

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
        arch=args.arch,
        img_size=args.img_size,
        pretrained=args.pretrained,
        regime=args.regime,
    )


def _run_bnnr_equal_compute(
    *,
    condition: ConditionSpec,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
    selection_mode: str,  # "xai" | "random"
) -> dict[str, Any]:
    """BNNR with equal compute budget.

    Total budget = B epochs split into (1 + N_CANDIDATES) phases.
    Phase 0 (baseline): B // (1 + N_CANDIDATES) epochs on selection_val.
    Phases 1..N: each candidate trained for B // (1 + N_CANDIDATES) epochs.
    Selection: xai -> best selection_val score; random -> random.Random(seed).choice.
    Final metric: evaluate selected model on held_out_test.
    """
    import json

    from bnnr.utils import set_seed

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
    train_loader, selection_val_loader, held_out_test_loader = _imagewoof_loaders_v2(
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=seed,
        policy="base",
        n_per_class_train=args.train_per_class,
        data_dir=getattr(args, "data_dir", None),
    )

    # Build baseline adapter once; save its initial weights to reset per candidate
    baseline_adapter = _build_adapter(
        arch=args.arch,
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        lr=args.lr,
        device=cfg.device,
        epochs=epochs_per_phase,
    )

    header = (
        f"\n{'='*60}\n"
        f"  BNNR {selection_mode.upper()} (v2 equal-compute)\n"
        f"  budget={budget}  epochs_per_phase={epochs_per_phase}\n"
        f"  N_CANDIDATES={N_CANDIDATES}  total_gpu_epochs={total_gpu_epochs}\n"
        f"  seed={seed}  device={cfg.device}\n"
        f"{'='*60}"
    )
    print(header, flush=True)

    t0 = time.perf_counter()

    # ---- Phase 0: baseline (no extra aug) on selection_val ----
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
    # Save baseline weights so every candidate starts from the same checkpoint
    baseline_state = copy.deepcopy(baseline_adapter.state_dict())
    print(f"  [Phase 0] baseline selection_val accuracy={baseline_score:.4f}", flush=True)

    # ---- Pre-compute XAI cache (one-shot, shared across all candidate phases) ----
    # Without this, ICD/AICD recompute saliency maps online for every training
    # batch — ~100x slower than necessary. Cache once from the baseline model.
    from bnnr.xai_cache import XAICache

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

    # ---- Phases 1..N_CANDIDATES: evaluate each candidate ----
    candidate_scores: list[float] = []
    candidate_states: list[dict[str, Any]] = []
    candidate_val_metrics: list[dict[str, float]] = []

    for i in range(N_CANDIDATES):
        cand_name = _CANDIDATE_NAMES[i]
        print(f"  [Phase {i+1}/{N_CANDIDATES} — {cand_name}] {epochs_per_phase} epochs...", flush=True)

        # Fresh adapter from baseline weights
        cand_adapter = _build_adapter(
            arch=args.arch,
            num_classes=NUM_CLASSES,
            pretrained=args.pretrained,
            lr=args.lr,
            device=cfg.device,
            epochs=epochs_per_phase,
        )
        # Restore only model weights from baseline — not optimizer or scheduler.
        # Restoring the full state would leave CosineAnnealingLR at
        # last_epoch = epochs_per_phase (LR ≈ eta_min), making all candidate
        # phases train at near-zero LR and score identically (meaningless
        # selection). Each candidate gets a fresh optimizer + fresh scheduler
        # so it starts at LR = lr_max, comparable to baseline Phase 0.
        cand_adapter.model.load_state_dict(baseline_state["model"])

        # Build augmentation with the fresh model + shared XAI cache.
        # Passing xai_cache to ICD/AICD avoids recomputing saliency maps
        # online for every training batch (would be ~100x slower otherwise).
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
            f"  [XAI selection] best={selected_candidate}  "
            f"score={selection_val_metric:.4f}",
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

    # ---- Final evaluation: selected model on held_out_test ----
    # The selected model was already trained; just evaluate it.
    # (No additional training — the epoch budget is fully spent.)
    final_adapter = _build_adapter(
        arch=args.arch,
        num_classes=NUM_CLASSES,
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
        f"  [held_out_test] accuracy={held_out_test_metric:.4f}  "
        f"elapsed={elapsed_s:.1f}s",
        flush=True,
    )

    # ---- XAI overlays ----
    if not args.no_xai:
        xai_meta = export_attention_maps(
            final_adapter,
            held_out_test_loader,
            sample_indices=_XAI_VAL_INDICES,
            output_dir=run_dir / "xai",
            xai_method="opticam",
        )
    else:
        xai_meta = {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}

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

    # Override val_metric to be the held_out_test metric for consistent reporting
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
) -> dict[str, Any]:
    spec = CONDITIONS[condition_id]

    if condition_id == "bnnr_xai":
        return _run_bnnr_equal_compute(
            condition=spec,
            seed=seed,
            args=args,
            output_root=output_root,
            selection_mode="xai",
        )
    if condition_id == "bnnr_random":
        return _run_bnnr_equal_compute(
            condition=spec,
            seed=seed,
            args=args,
            output_root=output_root,
            selection_mode="random",
        )

    if condition_id == "churchnoise_only":
        from bnnr.augmentations import ChurchNoise

        batch_augs: list[Any] = [
            ChurchNoise(
                probability=0.5,
                intensity=0.5,
                noise_strength_range=(3.0, 8.0),
                random_state=seed,
            )
        ]
        return _run_plain_condition(
            condition=spec,
            policy="base",
            extra_batch_augmentations=batch_augs,
            seed=seed,
            args=args,
            output_root=output_root,
        )

    # Plain conditions: no_aug, randaugment, trivialaugment
    policy = _POLICY_BY_CONDITION[condition_id]
    return _run_plain_condition(
        condition=spec,
        policy=policy,
        extra_batch_augmentations=[],
        seed=seed,
        args=args,
        output_root=output_root,
    )


# ---------------------------------------------------------------------------
# Results document
# ---------------------------------------------------------------------------


def _benchmark_document_v2(args: argparse.Namespace) -> dict[str, Any]:
    epochs_per_phase = args.budget // (1 + N_CANDIDATES)
    return {
        "benchmark_id": "imagewoof_resnet18_augmentation_comparison_v2",
        "benchmark_version": "v2",
        "model": f"{args.arch} (torchvision, pretrained={args.pretrained})",
        "dataset": (
            f"imagewoof2-160 (fast.ai); balanced {args.train_per_class}/class train, "
            "imagewoof val split 50/50 per class (deterministic) into "
            "selection_val + held_out_test"
        ),
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
        "attention_method": "OptiCAM overlays on fixed Imagewoof val indices",
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
            "Equal total GPU-epoch budget B for all conditions: baselines train B epochs "
            f"straight; BNNR trains B//(1+N_candidates) = {epochs_per_phase} epochs per phase.",
            "Held-out test set: imagewoof val split 50/50 per class (deterministic); "
            "BNNR selection uses selection_val only; all final metrics on held_out_test.",
            "bnnr_random uses identical compute and augmentation pool as bnnr_xai "
            "but selects a candidate uniformly at random (seed-controlled). "
            "bnnr_xai vs bnnr_random isolates the contribution of XAI-guided selection.",
            "Not a claim of ImageNet-SOTA. Low-data augmentation comparison benchmark.",
        ],
    }


# ---------------------------------------------------------------------------
# Estimate wall-clock
# ---------------------------------------------------------------------------


def _estimate_v2(args: argparse.Namespace, n_seeds: int, conds: list[str]) -> str:
    bnnr_conds = {"bnnr_xai", "bnnr_random"}
    bnnr_count = sum(1 for c in conds if c in bnnr_conds)
    plain_count = len(conds) - bnnr_count
    # Rough: each epoch ~1 min on CPU, ~0.15 on GPU
    plain_epochs_total = plain_count * args.budget * n_seeds
    bnnr_epochs_total = bnnr_count * args.budget * n_seeds  # same total by design
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
        description="ResNet18 / Imagewoof augmentation benchmark v2 (paper-quality, equal-compute)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=40,
        help=(
            "Total GPU-epoch budget per condition (default 40). "
            "Baselines train budget epochs straight. "
            f"BNNR splits into (1 + {N_CANDIDATES}) phases of budget//(1+{N_CANDIDATES}) epochs each."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46,47,48,49,50,51",
        help="Comma-separated seeds (default 10 seeds for paper-quality statistics)",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help=f"Comma-separated conditions ({', '.join(CONDITIONS)})",
    )
    parser.add_argument(
        "--regime",
        choices=["scratch", "pretrained"],
        default="scratch",
        help="Training regime label recorded in results (default: scratch)",
    )
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--img-size", type=int, default=128, help="Square resize/crop target")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use ImageNet-pretrained backbone (default: from scratch). "
            "Imagewoof classes overlap ImageNet, so from-scratch is the honest default."
        ),
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=100,
        help="Balanced train images per class (low-data regime). None = full train.",
    )
    parser.add_argument(
        "--no-xai",
        action="store_true",
        help="Skip OptiCAM overlay export (faster; no cv2 dependency)",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory where Imagewoof2 is downloaded/cached (default: <repo>/data). "
            "Set to a Drive path to survive Colab session restarts."
        ),
    )
    parser.add_argument(
        "--drive-base-dir",
        type=Path,
        default=None,
        help=(
            "Convenience for Colab/Drive: a single directory under which "
            "results_imagewoof_v2.json, runs_imagewoof_v2/, and the cached dataset are "
            "written. Overrides --results, --output-root, and --data-dir."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run: 1 epoch/phase (budget=4), 2 seeds, 5/class, img-size 64",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    # --drive-base-dir convenience
    if args.drive_base_dir is not None:
        args.drive_base_dir.mkdir(parents=True, exist_ok=True)
        args.results = args.drive_base_dir / "results_imagewoof_v2.json"
        args.output_root = args.drive_base_dir / "runs_imagewoof_v2"
        if args.data_dir is None:
            args.data_dir = args.drive_base_dir / "data"

    # --smoke overrides
    if args.smoke:
        # budget=4 means epochs_per_phase=1 for BNNR (4//(1+3)=1)
        args.budget = 4
        args.img_size = 64
        args.train_per_class = 5
        if args.seeds == "42,43,44,45,46,47,48,49,50,51":
            args.seeds = "42,43"

    # Resolve device
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    # Also set pretrained based on regime arg
    if args.regime == "pretrained":
        args.pretrained = True

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in CONDITIONS:
            parser.error(f"Unknown condition {c!r}. Choose from: {', '.join(CONDITIONS)}")

    epochs_per_phase = args.budget // (1 + N_CANDIDATES)
    print("ResNet18 / Imagewoof augmentation benchmark v2 (paper-quality, equal-compute)")
    print(f"  arch={args.arch}  pretrained={args.pretrained}  img_size={args.img_size}")
    print(f"  train_per_class={args.train_per_class}  seeds={seeds}  conditions={conds}")
    print(
        f"  budget={args.budget} epochs  epochs_per_phase={epochs_per_phase}  "
        f"N_candidates={N_CANDIDATES}  device={args.device}"
    )
    print(f"  {_estimate_v2(args, len(seeds), conds)}")
    print(f"  results -> {args.results}")
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    data = load_results(args.results)
    doc = _benchmark_document_v2(args)
    data.update(doc)
    data.setdefault("runs", [])
    data["hardware"] = torch_info()
    data["git_head"] = git_head()
    data["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Resume safety: skip (condition, seed, regime) pairs already present
    done = {
        (r["condition"], r["seed"], r.get("regime", "scratch"))
        for r in data["runs"]
        if "val_metric" in r and "error" not in r
    }

    for seed in seeds:
        for cid in conds:
            key = (cid, seed, args.regime)
            if key in done:
                print(f"SKIP {cid} seed={seed} regime={args.regime} (already in {args.results})")
                continue
            print(f"\n>>> condition={cid}  seed={seed}  regime={args.regime}")
            try:
                entry = run_condition(
                    condition_id=cid,
                    seed=seed,
                    args=args,
                    output_root=args.output_root,
                )
            except Exception as exc:  # noqa: BLE001 — record and continue the matrix
                import traceback
                print(f"    FAILED ({cid}, seed={seed}): {exc}", file=sys.stderr)
                traceback.print_exc()
                entry = {
                    "condition": cid,
                    "seed": seed,
                    "regime": args.regime,
                    "dataset": "imagewoof",
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
        f"Summarize: python benchmarks/summarize_v2.py "
        f"--results {args.results} --markdown"
    )


if __name__ == "__main__":
    main()
