"""
Full BNNR pipeline demo — all augmentation types, XAI, and library wrappers.

Demonstrates:
  - All 8 built-in BNNR augmentations: 7 unique (ChurchNoise, DifPresets,
    Drust, LuxferGlass, ProCAM, Smugs, TeaStains) + BasicAugmentation (standard transforms)
  - TorchvisionAugmentation wrapper (ColorJitter, RandomHorizontalFlip)
  - AlbumentationsAugmentation wrapper (HorizontalFlip, RandomBrightnessContrast,
    GaussNoise — if albumentations is installed)
  - KorniaAugmentation wrapper (RandomHorizontalFlip, ColorJitter — if kornia installed)
  - ICD / AICD (XAI-driven saliency-based augmentations)
  - XAI cache precomputation
  - OptiCAM saliency map generation
  - Iterative augmentation selection with pruning
  - JSONL event logging (for dashboard)
  - JSON report generation

Uses a small synthetic RGB dataset for fast execution.

Run:
    PYTHONPATH=src python examples/classification/demo_full_pipeline.py

Expected runtime: ~30-90 seconds on CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from bnnr import (
    AlbumentationsAugmentation,
    AugmentationRegistry,
    BNNRConfig,
    BNNRTrainer,
    BasicAugmentation,
    ChurchNoise,
    DifPresets,
    Drust,
    KorniaAugmentation,
    LuxferGlass,
    ProCAM,
    SimpleTorchAdapter,
    Smugs,
    TeaStains,
    TorchvisionAugmentation,
    albumentations_available,
    kornia_available,
    load_report,
)
from bnnr.events import load_events
from bnnr.augmentations import BaseAugmentation
from bnnr.icd import AICD, ICD
from bnnr.xai_cache import XAICache

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
N_CLASSES = 5
N_TRAIN = 100
N_VAL = 40
IMG_SIZE = 32
BATCH_SIZE = 16
M_EPOCHS = 2
MAX_ITERATIONS = 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset — synthetic colour images with class-dependent patterns
# ═══════════════════════════════════════════════════════════════════════════════

class _IndexedDataset(Dataset):
    """Wraps (image, label) dataset → (image, label, index) for ICD cache."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, lbl = self.dataset[idx]
        return img, lbl, idx


def _make_synthetic_dataset(n: int, n_classes: int) -> TensorDataset:
    """Create a synthetic RGB dataset where each class has a distinct mean colour.

    This makes the classification problem non-trivial but solvable, so BNNR
    augmentation selection has something meaningful to work with.
    """
    torch.manual_seed(SEED)
    images = []
    labels = []
    class_colors = torch.rand(n_classes, 3)  # base colour per class

    for i in range(n):
        cls = i % n_classes
        base = class_colors[cls].view(3, 1, 1).expand(3, IMG_SIZE, IMG_SIZE)
        noise = torch.rand(3, IMG_SIZE, IMG_SIZE) * 0.3
        img = (base * 0.7 + noise).clamp(0, 1)
        images.append(img)
        labels.append(cls)

    return TensorDataset(torch.stack(images), torch.tensor(labels))


# ═══════════════════════════════════════════════════════════════════════════════
#  Model — tiny CNN (works fast on CPU)
# ═══════════════════════════════════════════════════════════════════════════════

class DemoCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Augmentation builder — all available types
# ═══════════════════════════════════════════════════════════════════════════════

def _build_all_augmentations(
    model: nn.Module,
    target_layers: list[nn.Module],
    xai_cache: XAICache | None,
) -> list[BaseAugmentation]:
    """Build augmentation candidates from all available sources."""
    augs: list[BaseAugmentation] = []

    # ── 1. All 8 built-in BNNR augmentations (7 unique + BasicAug) ──────
    augs.extend([
        ChurchNoise(probability=0.5, intensity=0.5, random_state=SEED),
        BasicAugmentation(probability=0.5, intensity=0.5, random_state=SEED + 1),
        DifPresets(probability=0.5, intensity=0.7, random_state=SEED + 2),
        Drust(probability=0.5, intensity=0.5, random_state=SEED + 3),
        LuxferGlass(probability=0.5, intensity=0.5, random_state=SEED + 4),
        ProCAM(probability=0.5, random_state=SEED + 5),
        Smugs(probability=0.5, intensity=1.5, random_state=SEED + 6),
        TeaStains(probability=0.5, intensity=0.5, random_state=SEED + 7),
    ])
    print(f"  ✓ 8 built-in BNNR augmentations (7 unique + BasicAug)")

    # ── 2. Torchvision wrappers ─────────────────────────────────────────
    from torchvision import transforms

    augs.append(TorchvisionAugmentation(
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        name_override="tv_color_jitter",
        probability=0.5,
        random_state=SEED + 20,
    ))
    augs.append(TorchvisionAugmentation(
        transforms.RandomHorizontalFlip(p=1.0),
        name_override="tv_hflip",
        probability=0.5,
        random_state=SEED + 21,
    ))
    print(f"  ✓ 2 torchvision wrappers (ColorJitter, HorizontalFlip)")

    # ── 3. Albumentations wrappers (optional) ───────────────────────────
    if albumentations_available():
        import albumentations as A

        augs.append(AlbumentationsAugmentation(
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]),
            name_override="albu_standard",
            probability=0.5,
            random_state=SEED + 30,
        ))
        augs.append(AlbumentationsAugmentation(
            A.GaussNoise(p=1.0),
            name_override="albu_gauss_noise",
            probability=0.5,
            random_state=SEED + 31,
        ))
        print(f"  ✓ 2 albumentations wrappers (Compose, GaussNoise)")
    else:
        print(f"  ⓘ albumentations not installed — skipping (pip install bnnr[albumentations])")

    # ── 4. Kornia wrappers (optional) ────────────────────────────────────
    if kornia_available():
        import kornia.augmentation as K

        augs.append(KorniaAugmentation(
            K.RandomHorizontalFlip(p=1.0),
            name_override="kornia_hflip",
            probability=0.5,
            random_state=SEED + 40,
        ))
        augs.append(KorniaAugmentation(
            K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
            name_override="kornia_color_jitter",
            probability=0.5,
            random_state=SEED + 41,
        ))
        print(f"  ✓ 2 kornia wrappers (HFlip, ColorJitter)")
    else:
        print(f"  ⓘ kornia not installed — skipping (pip install bnnr[gpu])")

    # ── 5. ICD & AICD (XAI-driven) ──────────────────────────────────────
    icd = ICD(
        model=model, target_layers=target_layers,
        cache=xai_cache, probability=0.5, random_state=SEED + 50,
    )
    icd.name = "icd"
    augs.append(icd)

    aicd = AICD(
        model=model, target_layers=target_layers,
        cache=xai_cache, probability=0.5, random_state=SEED + 51,
    )
    aicd.name = "aicd"
    augs.append(aicd)
    print(f"  ✓ 2 XAI-driven augmentations (ICD, AICD)")

    return augs


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.perf_counter()
    output_dir = Path("demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 64)
    print("  BNNR Full Pipeline Demo")
    print("  All augmentation types • XAI • Iterative selection")
    print("=" * 64)
    print()

    # ── Data ─────────────────────────────────────────────────────────────
    print("[1/6] Creating synthetic dataset …")
    train_ds = _make_synthetic_dataset(N_TRAIN, N_CLASSES)
    val_ds = _make_synthetic_dataset(N_VAL, N_CLASSES)
    indexed_train = _IndexedDataset(train_ds)
    indexed_val = _IndexedDataset(val_ds)
    train_loader = DataLoader(indexed_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(indexed_val, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train: {N_TRAIN}  Val: {N_VAL}  Classes: {N_CLASSES}  "
          f"Image: 3×{IMG_SIZE}×{IMG_SIZE}")
    print()

    # ── Model ────────────────────────────────────────────────────────────
    print("[2/6] Initializing model …")
    model = DemoCNN(n_classes=N_CLASSES)
    target_layers = [model.conv2]  # Last conv for OptiCAM
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DemoCNN: {n_params:,} params")
    print()

    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=target_layers,
        device="cpu",
        eval_metrics=["accuracy", "f1_macro", "fbeta_0.5"],
    )

    # ── XAI Cache ────────────────────────────────────────────────────────
    print("[3/6] Precomputing XAI cache for ICD/AICD …")
    cache_dir = output_dir / "xai_cache"
    xai_cache = XAICache(cache_dir)
    written = xai_cache.precompute_cache(
        model=model, train_loader=train_loader,
        target_layers=target_layers, n_samples=N_TRAIN,
        method="opticam",
    )
    print(f"  Cached {written} saliency maps")
    print()

    # ── Augmentations ────────────────────────────────────────────────────
    print("[4/6] Building augmentation candidates …")
    augmentations = _build_all_augmentations(model, target_layers, xai_cache)
    print(f"\n  Total: {len(augmentations)} augmentation candidates:")
    for i, aug in enumerate(augmentations, 1):
        print(f"    {i:2d}. {aug.name:<24s}  (p={aug.probability:.2f})")
    print()

    # ── Config ───────────────────────────────────────────────────────────
    # Demonstrates custom metric selection: use fbeta_0.5 (precision-heavy)
    # as the metric that drives augmentation branch selection instead of
    # the default "accuracy".  Any sklearn-compatible metric can be used;
    # see docs/configuration.md for the full list.
    print("[5/6] Configuring BNNR …")
    cfg = BNNRConfig(
        m_epochs=M_EPOCHS,
        max_iterations=MAX_ITERATIONS,
        metrics=["accuracy", "f1_macro", "fbeta_0.5", "loss"],
        selection_metric="fbeta_0.5",
        selection_mode="max",
        device="cpu",
        xai_enabled=True,
        xai_method="opticam",
        xai_samples=2,
        xai_cache_dir=cache_dir,
        xai_cache_samples=N_TRAIN,
        save_checkpoints=True,
        verbose=True,
        checkpoint_dir=output_dir / "checkpoints",
        report_dir=output_dir / "reports",
        early_stopping_patience=3,
        candidate_pruning_enabled=True,
        candidate_pruning_warmup_epochs=1,
        candidate_pruning_relative_threshold=0.5,
        event_log_enabled=True,
        report_preview_size=128,
        report_xai_size=128,
        report_probe_images_per_class=1,
        report_probe_max_classes=N_CLASSES,
        seed=SEED,
    )
    print(f"  Epochs per branch      : {M_EPOCHS}")
    print(f"  Decision rounds        : {MAX_ITERATIONS}")
    print(f"  Selection metric       : {cfg.selection_metric}")
    print(f"  XAI method             : OptiCAM")
    print(f"  Candidate pruning      : enabled (threshold=0.5)")
    print(f"  Event logging          : enabled")
    print()

    # ── Train ────────────────────────────────────────────────────────────
    print("[6/6] Running BNNR iterative training …")
    print()
    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
    result = trainer.run()

    elapsed = time.perf_counter() - t0

    # ── Results ──────────────────────────────────────────────────────────
    events_path = result.report_json_path.parent / "events.jsonl"
    events = load_events(events_path) if events_path.exists() else []

    print()
    print("=" * 64)
    print("  DEMO COMPLETE")
    print("-" * 64)
    print(f"  Time elapsed       : {elapsed:.1f}s")
    print(f"  Best path          : {result.best_path}")
    print(f"  Best Fβ=0.5       : {result.best_metrics.get('fbeta_0.5', 0):.4f}")
    print(f"  Best accuracy      : {result.best_metrics.get('accuracy', 0):.4f}")
    print(f"  Best loss          : {result.best_metrics.get('loss', 0):.4f}")
    print(f"  Selected augs      : {result.selected_augmentations}")
    print(f"  Total events       : {len(events)}")
    print(f"  Report JSON        : {result.report_json_path}")
    print(f"  Events (JSONL)     : {events_path}")
    print(f"  Checkpoints        : {cfg.checkpoint_dir}")
    print()

    # Quick verification
    loaded = load_report(result.report_json_path)
    assert loaded.best_path == result.best_path
    print("  ✓ Report load/verify passed")

    ckpts = sorted(cfg.checkpoint_dir.glob("*.pt"))
    print(f"  ✓ {len(ckpts)} checkpoint(s) saved")

    if events:
        event_types = {e["type"] for e in events}
        print(f"  ✓ Event types: {sorted(event_types)}")

    print()
    print("=" * 64)
    print("  All library wrappers and BNNR features exercised successfully!")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
