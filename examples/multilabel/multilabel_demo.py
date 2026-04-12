"""
Multi-label classification demo — BNNR with multi-hot labels.

Demonstrates:
  - Multi-label classification (each image can have MULTIPLE labels)
  - SimpleTorchAdapter with ``multilabel=True`` (auto-selects BCEWithLogitsLoss)
  - BNNRConfig with ``task="multilabel"`` (auto-selects f1_samples metric)
  - All 8 built-in BNNR augmentations (7 unique + BasicAug) + ICD/AICD (XAI-driven)
  - XAI cache precomputation
  - OptiCAM saliency maps
  - Iterative augmentation selection with pruning
  - JSONL event logging (for dashboard)
  - JSON report generation

Uses a synthetic RGB dataset where each image can exhibit multiple "attributes"
(colour channels, textures), making it a natural multi-label problem.

Run with dashboard (recommended):
    PYTHONPATH=src python examples/multilabel/multilabel_demo.py --with-dashboard

Quick test (CPU, ~1-2 min):
    PYTHONPATH=src python examples/multilabel/multilabel_demo.py --quick

Without dashboard:
    PYTHONPATH=src python examples/multilabel/multilabel_demo.py --without-dashboard

Expected runtime: ~2-5 min on CPU (quick), ~10-20 min full.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from bnnr import (
    AugmentationRegistry,
    BNNRConfig,
    BNNRTrainer,
    SimpleTorchAdapter,
    start_dashboard,
)
from bnnr.augmentations import BaseAugmentation
from bnnr.icd import AICD, ICD
from bnnr.xai_cache import XAICache


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
N_LABELS = 6  # number of possible labels per image
IMG_SIZE = 48
LABEL_NAMES = [
    "red_tint",       # 0 — image has warm red component
    "blue_tint",      # 1 — image has cool blue component
    "high_contrast",  # 2 — image has strong edges/contrast
    "striped",        # 3 — image has horizontal stripe pattern
    "spotted",        # 4 — image has round spot pattern
    "bright",         # 5 — image is overall bright
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset — synthetic multi-label RGB images
# ═══════════════════════════════════════════════════════════════════════════════

class _IndexedDataset(Dataset):
    """Wraps (image, label) dataset → (image, label, index) for ICD cache."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):  # noqa: ANN204
        img, lbl = self.dataset[idx]
        return img, lbl, idx


def _make_multilabel_dataset(n: int, seed: int = SEED) -> TensorDataset:
    """Create a synthetic multi-label RGB dataset.

    Each image is generated with a random combination of visual attributes:
      - red_tint:       adds warm red to the image
      - blue_tint:      adds cool blue to the image
      - high_contrast:  adds sharp horizontal edge
      - striped:        adds horizontal stripe pattern
      - spotted:        adds circular spots
      - bright:         increases overall brightness

    Labels are multi-hot vectors of shape (N_LABELS,) with 1-4 active labels.
    The visual signal is imperfect (noisy), making the problem non-trivial.
    """
    rng = np.random.RandomState(seed)
    images = []
    labels = []

    for _ in range(n):
        # Base image — low-intensity noise
        img = rng.rand(3, IMG_SIZE, IMG_SIZE).astype(np.float32) * 0.15

        # Pick 1-4 active attributes
        n_active = rng.randint(1, 5)
        active = rng.choice(N_LABELS, size=n_active, replace=False)
        label = np.zeros(N_LABELS, dtype=np.float32)
        label[active] = 1.0

        # Apply visual features for each active label
        for a in active:
            if a == 0:  # red_tint
                img[0] += rng.uniform(0.3, 0.6)
            elif a == 1:  # blue_tint
                img[2] += rng.uniform(0.3, 0.6)
            elif a == 2:  # high_contrast — horizontal edge at random row
                row = rng.randint(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
                img[:, :row, :] += 0.3
                img[:, row:, :] -= 0.1
            elif a == 3:  # striped — horizontal stripes
                for r in range(0, IMG_SIZE, 4):
                    img[:, r : r + 2, :] += 0.25
            elif a == 4:  # spotted — add 3-5 bright circular spots
                n_spots = rng.randint(3, 6)
                for _ in range(n_spots):
                    cy, cx = rng.randint(4, IMG_SIZE - 4, size=2)
                    radius = rng.randint(2, 5)
                    yy, xx = np.ogrid[-cy : IMG_SIZE - cy, -cx : IMG_SIZE - cx]
                    mask = (xx**2 + yy**2) <= radius**2
                    img[:, mask] += 0.35
            elif a == 5:  # bright — overall brightness boost
                img += rng.uniform(0.2, 0.4)

        # Add noise and clamp
        img += rng.randn(3, IMG_SIZE, IMG_SIZE).astype(np.float32) * 0.05
        img = np.clip(img, 0.0, 1.0)

        images.append(torch.from_numpy(img))
        labels.append(torch.from_numpy(label))

    return TensorDataset(torch.stack(images), torch.stack(labels))


# ═══════════════════════════════════════════════════════════════════════════════
#  Model — small CNN for multi-label classification
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLabelCNN(nn.Module):
    """
    Small CNN for multi-label classification.

    Architecture:
      Block 1: 48→24  (32 ch)
      Block 2: 24→12  (64 ch)
      Block 3: 12→GAP (128 ch)  ← target layer for OptiCAM
      Classifier: 128 → N_LABELS (raw logits, no sigmoid — BCEWithLogitsLoss)

    ~90K parameters — fast on CPU, enough capacity for the synthetic dataset.
    """

    def __init__(self, n_labels: int = N_LABELS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 48 → 24
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 24 → 12
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 12 → GAP
            nn.Conv2d(64, 128, 3, padding=1),   # ← target layer for OptiCAM
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_labels),
            # NOTE: no sigmoid here — BCEWithLogitsLoss includes it internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def target_layer(self) -> nn.Module:
        """Last conv layer for OptiCAM / saliency maps."""
        return self.features[8]  # Conv2d(64, 128, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  Augmentations — all 8 built-in BNNR types (7 unique + BasicAug) + ICD/AICD
# ═══════════════════════════════════════════════════════════════════════════════

def _named(registry_name: str, display_name: str, **kwargs: object) -> BaseAugmentation:
    """Create augmentation from registry with custom display name."""
    aug = AugmentationRegistry.create(registry_name, **kwargs)
    aug.name = display_name
    return aug


def _build_augmentations(
    model: nn.Module,
    target_layers: list[nn.Module],
    xai_cache: XAICache | None,
    *,
    quick: bool = False,
) -> list[BaseAugmentation]:
    """Build augmentation candidates.

    Quick mode  (4):  2 standard + ICD + AICD  — fast CPU testing.
    Full mode  (10):  8 standard + ICD + AICD  — all built-in BNNR types.
    """
    seed = SEED
    augs: list[BaseAugmentation] = []

    if quick:
        # Just 2 representative augmentations for speed
        augs.extend([
            _named("basic_augmentation", "basic_aug", probability=0.5, random_state=seed),
            _named("church_noise", "church_noise", probability=0.5, random_state=seed + 1),
        ])
    else:
        # All 8 built-in BNNR augmentations (7 unique + BasicAug)
        augs.extend([
            _named("basic_augmentation", "basic_augmentation", probability=0.5, random_state=seed),
            _named("church_noise", "church_noise", probability=0.5, random_state=seed + 1),
            _named("dif_presets", "dif_presets", probability=0.5, random_state=seed + 2),
            _named("drust", "drust", probability=0.5, random_state=seed + 3),
            _named("luxfer_glass", "luxfer_glass", probability=0.5, random_state=seed + 4),
            _named("procam", "procam", probability=0.5, random_state=seed + 5),
            _named("smugs", "smugs", probability=0.5, random_state=seed + 6),
            _named("tea_stains", "tea_stains", probability=0.5, random_state=seed + 7),
        ])

    # ICD & AICD — XAI-driven augmentations (work with multi-hot labels!)
    use_cuda = torch.cuda.is_available()
    icd = ICD(
        model=model, target_layers=target_layers,
        cache=xai_cache, probability=0.5, random_state=seed + 10,
        use_cuda=use_cuda,
    )
    icd.name = "icd"
    augs.append(icd)

    aicd = AICD(
        model=model, target_layers=target_layers,
        cache=xai_cache, probability=0.5, random_state=seed + 11,
        use_cuda=use_cuda,
    )
    aicd.name = "aicd"
    augs.append(aicd)

    return augs


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BNNR multi-label classification demo with multi-hot labels",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick test: fewer samples, fewer augmentations, fewer epochs (~1-2 min CPU)",
    )
    p.add_argument("--n-train", type=int, default=None, help="Training samples (default: 400)")
    p.add_argument("--n-val", type=int, default=None, help="Validation samples (default: 200)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--m-epochs", type=int, default=None, help="Epochs per branch (default: 5)")
    p.add_argument("--decisions", type=int, default=None, help="Decision rounds (default: 4)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    dash = p.add_mutually_exclusive_group()
    dash.add_argument("--with-dashboard", dest="with_dashboard", action="store_true")
    dash.add_argument("--without-dashboard", dest="with_dashboard", action="store_false")
    p.set_defaults(with_dashboard=True)
    p.add_argument("--dashboard-port", type=int, default=8080)

    auto = p.add_mutually_exclusive_group()
    auto.add_argument("--dashboard-auto-open", dest="dashboard_auto_open", action="store_true")
    auto.add_argument("--no-dashboard-auto-open", dest="dashboard_auto_open", action="store_false")
    p.set_defaults(dashboard_auto_open=True)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    # ── Defaults / quick overrides ────────────────────────────────────────
    if args.quick:
        n_train = args.n_train or 200
        n_val = args.n_val or 100
        m_epochs = args.m_epochs or 3
        max_decisions = args.decisions or 2
    else:
        n_train = args.n_train or 400
        n_val = args.n_val or 200
        m_epochs = args.m_epochs or 5
        max_decisions = args.decisions or 4

    total_epochs = m_epochs * (max_decisions + 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("=" * 64)
    print("  BNNR  ·  Multi-Label Classification Demo")
    print("-" * 64)
    print(f"  Labels          : {N_LABELS} ({', '.join(LABEL_NAMES)})")
    print(f"  Training set    : {n_train} images")
    print(f"  Validation set  : {n_val} images")
    print(f"  Max epochs      : ~{total_epochs}  ({m_epochs} epochs × {max_decisions + 1} branches)")
    print(f"  Decision rounds : {max_decisions}")
    print(f"  XAI method      : OptiCAM")
    print(f"  Device          : {device}")
    print(f"  Mode            : {'QUICK' if args.quick else 'FULL'}")
    print("=" * 64)
    print()

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path("multilabel_demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dashboard ─────────────────────────────────────────────────────────
    report_dir = output_dir / "reports"
    dashboard_url: str | None = None
    if args.with_dashboard:
        dashboard_url = start_dashboard(
            report_dir,
            port=args.dashboard_port,
            auto_open=args.dashboard_auto_open,
        )

    # ── Data ──────────────────────────────────────────────────────────────
    print("[1/6] Creating synthetic multi-label dataset …")
    train_ds = _make_multilabel_dataset(n_train, seed=SEED)
    val_ds = _make_multilabel_dataset(n_val, seed=SEED + 1000)

    # Quick peek at label distribution
    all_labels = train_ds.tensors[1].numpy()
    counts = all_labels.sum(axis=0).astype(int)
    avg_labels = all_labels.sum(axis=1).mean()
    print(f"  Samples: {n_train} train / {n_val} val")
    print(f"  Avg labels/image: {avg_labels:.1f}")
    print(f"  Per-label counts: {dict(zip(LABEL_NAMES, counts))}")
    print()

    train_loader = DataLoader(
        _IndexedDataset(train_ds), batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        _IndexedDataset(val_ds), batch_size=args.batch_size, shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("[2/6] Initializing model …")
    model = MultiLabelCNN(n_labels=N_LABELS)
    target_layers = [model.target_layer]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MultiLabelCNN: {n_params:,} params  ({N_LABELS} output logits)")
    print()

    # ── Adapter — the key: multilabel=True ────────────────────────────────
    # SimpleTorchAdapter with multilabel=True automatically:
    #   • Uses BCEWithLogitsLoss as criterion
    #   • Applies sigmoid + threshold for predictions
    #   • Computes f1_samples, f1_macro, accuracy metrics
    print("[3/6] Creating adapter (multilabel=True) …")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        target_layers=target_layers,
        device=device,
        multilabel=True,           # ← enables multi-label mode
        multilabel_threshold=0.5,  # ← confidence threshold for predictions
        eval_metrics=["fbeta_2", "f1_samples", "f1_macro", "accuracy"],
    )
    print(f"  Loss function   : BCEWithLogitsLoss")
    print(f"  Eval metrics    : {adapter.eval_metrics}")
    print(f"  Threshold       : {adapter.multilabel_threshold}")
    print()

    # ── XAI Cache ─────────────────────────────────────────────────────────
    print("[4/6] Precomputing XAI cache for ICD/AICD …")
    cache_dir = output_dir / "xai_cache"
    xai_cache = XAICache(cache_dir)
    written = xai_cache.precompute_cache(
        model=model, train_loader=train_loader,
        target_layers=target_layers, n_samples=n_train,
        method="opticam",
    )
    print(f"  Cached {written} saliency maps")
    print()

    # ── Augmentations ─────────────────────────────────────────────────────
    print("[5/6] Building augmentation candidates …")
    augmentations = _build_augmentations(
        model, target_layers, xai_cache, quick=args.quick,
    )
    print(f"  Total: {len(augmentations)} candidates:")
    for i, aug in enumerate(augmentations, 1):
        print(f"    {i:2d}. {aug.name:<24s}  (p={aug.probability:.2f})")
    print()

    # ── BNNRConfig — the key: task="multilabel" ──────────────────────────
    # When task="multilabel", BNNRConfig automatically sets f1_samples as
    # the selection metric.  Here we override it with fbeta_2 (recall-heavy)
    # which is often more useful for multi-label tasks where missing a
    # label is more costly than a false positive.
    #
    # Available multi-label metrics:
    #   fbeta_<β> (e.g. fbeta_0.5, fbeta_2), f1_samples, f1_macro, f1_micro,
    #   f1_weighted, precision*, recall*, jaccard_*, hamming, zero_one_loss
    # See docs/configuration.md for the full list.
    print("[6/6] Configuring BNNR …")
    cfg = BNNRConfig(
        task="multilabel",         # ← enables multi-label mode in the trainer
        m_epochs=m_epochs,
        max_iterations=max_decisions,
        metrics=["fbeta_2", "f1_samples", "f1_macro", "accuracy", "loss"],
        selection_metric="fbeta_2",
        selection_mode="max",
        device=device,
        xai_enabled=True,
        xai_method="opticam",
        xai_samples=2,
        xai_cache_dir=cache_dir,
        xai_cache_samples=n_train,
        save_checkpoints=True,
        verbose=True,
        checkpoint_dir=output_dir / "checkpoints",
        report_dir=report_dir,
        early_stopping_patience=3,
        candidate_pruning_enabled=True,
        candidate_pruning_warmup_epochs=1,
        candidate_pruning_relative_threshold=0.5,
        event_log_enabled=args.with_dashboard,
        report_preview_size=128,
        report_xai_size=128,
        report_probe_images_per_class=2,
        report_probe_max_classes=N_LABELS,
        seed=SEED,
    )
    print(f"  task             : {cfg.task}")
    print(f"  selection_metric : {cfg.selection_metric}")
    print(f"  metrics          : {cfg.metrics}")
    print(f"  epochs/branch    : {m_epochs}")
    print(f"  decision rounds  : {max_decisions}")
    print()

    # ── Train ─────────────────────────────────────────────────────────────
    print("Starting BNNR iterative training …")
    print()
    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
    result = trainer.run()

    elapsed = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  MULTI-LABEL DEMO COMPLETE")
    print("-" * 64)
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print(f"  Best path         : {result.best_path}")
    print(f"  Best Fβ=2         : {result.best_metrics.get('fbeta_2', 0):.4f}")
    print(f"  Best F1 (samples) : {result.best_metrics.get('f1_samples', 0):.4f}")
    print(f"  Best F1 (macro)   : {result.best_metrics.get('f1_macro', 0):.4f}")
    print(f"  Best accuracy     : {result.best_metrics.get('accuracy', 0):.4f}")
    print(f"  Best loss         : {result.best_metrics.get('loss', 0):.4f}")
    print(f"  Selected augs     : {result.selected_augmentations}")
    print(f"  Report JSON       : {result.report_json_path}")
    if args.with_dashboard:
        if dashboard_url:
            print(f"  Dashboard         : {dashboard_url}")
        else:
            print(
                "  Dashboard         : not started (install optional deps: pip install \"bnnr[dashboard]\"). "
                "Events were still logged."
            )
    print("=" * 64)
    print()

    if args.with_dashboard and dashboard_url:
        print("Dashboard is still running — press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
