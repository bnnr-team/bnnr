"""
STL-10 comprehensive showcase — demonstrates BNNR with ALL augmentation types,
OptiCAM XAI, and up to 100 epochs with branch decisions every 5 epochs.

STL-10: 10 classes of 96×96 colour natural images — airplane, bird, car, cat,
deer, dog, horse, monkey, ship, truck.  The larger resolution (vs CIFAR/MNIST)
makes augmentation effects and XAI saliency maps clearly visible in the
dashboard at 512×512 preview size.

20 augmentation candidates are registered (all 12 BNNR types + 8 parameter
variants), so up to 20 decision rounds × 5 epochs = 100 main-path epochs.

Run with dashboard (recommended):
    PYTHONPATH=src python examples/classification/showcase_stl10.py --with-dashboard

Quick test (CPU-friendly, ~15 min):
    PYTHONPATH=src python examples/classification/showcase_stl10.py --with-dashboard --quick

Full 100-epoch run (GPU recommended, ~2-4 h):
    PYTHONPATH=src python examples/classification/showcase_stl10.py --with-dashboard \\
        --m-epochs 5 --decisions 20

Without dashboard:
    PYTHONPATH=src python examples/classification/showcase_stl10.py --without-dashboard
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from bnnr import AugmentationRegistry, BNNRTrainer, SimpleTorchAdapter, start_dashboard
from bnnr.config import load_config
from bnnr.icd import AICD, ICD
from bnnr.xai_cache import XAICache

# ── STL-10 class names ──────────────────────────────────────────────────────
STL10_CLASSES: list[str] = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# ── STL-10 per-channel statistics (computed from training split) ─────────────
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class _IndexedDataset(Dataset):
    """Wraps any (image, label) dataset to return (image, label, index)."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int):  # noqa: ANN204
        image, label = self.dataset[index]
        return image, label, index


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class STL10Net(nn.Module):
    """
    VGG-style CNN for 3-channel 96×96 images (full run).

    Architecture (~2.5 M params):
      Block 1: 96→48  (64 ch)
      Block 2: 48→24  (128 ch)
      Block 3: 24→12  (256 ch)
      Block 4: 12→GAP (512 ch)  ← target layer for OptiCAM
      Classifier: 512 → 256 → 10
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # ── Block 1 ·  96 × 96 → 48 × 48 ──
            nn.Conv2d(3, 64, 3, padding=1),       # 0
            nn.BatchNorm2d(64),                    # 1
            nn.ReLU(inplace=True),                 # 2
            nn.Conv2d(64, 64, 3, padding=1),       # 3
            nn.BatchNorm2d(64),                    # 4
            nn.ReLU(inplace=True),                 # 5
            nn.MaxPool2d(2, 2),                    # 6
            nn.Dropout2d(0.10),                    # 7

            # ── Block 2 ·  48 × 48 → 24 × 24 ──
            nn.Conv2d(64, 128, 3, padding=1),      # 8
            nn.BatchNorm2d(128),                   # 9
            nn.ReLU(inplace=True),                 # 10
            nn.Conv2d(128, 128, 3, padding=1),     # 11
            nn.BatchNorm2d(128),                   # 12
            nn.ReLU(inplace=True),                 # 13
            nn.MaxPool2d(2, 2),                    # 14
            nn.Dropout2d(0.15),                    # 15

            # ── Block 3 ·  24 × 24 → 12 × 12 ──
            nn.Conv2d(128, 256, 3, padding=1),     # 16
            nn.BatchNorm2d(256),                   # 17
            nn.ReLU(inplace=True),                 # 18
            nn.Conv2d(256, 256, 3, padding=1),     # 19
            nn.BatchNorm2d(256),                   # 20
            nn.ReLU(inplace=True),                 # 21
            nn.MaxPool2d(2, 2),                    # 22
            nn.Dropout2d(0.20),                    # 23

            # ── Block 4 ·  12 × 12 → GAP(1 × 1) ──
            nn.Conv2d(256, 512, 3, padding=1),     # 24  ← OptiCAM target
            nn.BatchNorm2d(512),                   # 25
            nn.ReLU(inplace=True),                 # 26
            nn.AdaptiveAvgPool2d((1, 1)),           # 27
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def target_layer(self) -> nn.Module:
        return self.features[24]


class STL10NetQuick(nn.Module):
    """
    Lightweight CNN for fast CPU demos (48×48 images, ~150 K params).

    3 blocks + GAP → fast enough for CPU, still shows BNNR benefits.
    Intentionally simple — overfits quickly, so augmentation-as-regularization
    has a clear measurable effect.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # ── Block 1: 48→24 ──
            nn.Conv2d(3, 32, 3, padding=1),        # 0
            nn.BatchNorm2d(32),                     # 1
            nn.ReLU(inplace=True),                  # 2
            nn.MaxPool2d(2, 2),                     # 3

            # ── Block 2: 24→12 ──
            nn.Conv2d(32, 64, 3, padding=1),        # 4
            nn.BatchNorm2d(64),                     # 5
            nn.ReLU(inplace=True),                  # 6
            nn.MaxPool2d(2, 2),                     # 7

            # ── Block 3: 12→GAP(1) ──
            nn.Conv2d(64, 128, 3, padding=1),       # 8  ← OptiCAM target
            nn.BatchNorm2d(128),                    # 9
            nn.ReLU(inplace=True),                  # 10
            nn.AdaptiveAvgPool2d((1, 1)),            # 11
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def target_layer(self) -> nn.Module:
        return self.features[8]


# ═══════════════════════════════════════════════════════════════════════════════
#  Augmentations — all 8 BNNR types + ICD/AICD + parameter variants
# ═══════════════════════════════════════════════════════════════════════════════

def _named(registry_name: str, display_name: str, **kwargs: object):
    """Create an augmentation from the registry and override its display name."""
    aug = AugmentationRegistry.create(registry_name, **kwargs)
    aug.name = display_name
    return aug


def _build_all_augmentations(
    model: nn.Module,
    target_layers: list[nn.Module],
    device: str,
    seed: int,
    xai_cache: XAICache | None,
    *,
    quick: bool = False,
) -> list:
    """
    Build augmentation candidates.

    Available BNNR built-in types (8):
      church_noise, basic_augmentation, dif_presets, drust,
      luxfer_glass, procam, smugs, tea_stains

    Full mode (18 candidates):
      • 8 standard BNNR augmentations at default strength
      • 2 XAI-driven augmentations      (ICD, AICD)
      • 8 parameter variants            (stronger / softer settings)

    Quick mode (8 candidates):
      • 8 standard BNNR augmentations only (fastest, best for CPU)
    """
    use_cuda = device != "cpu"

    # ── Wave 1 — all 8 canonical augmentations ─────────────────────────────
    # Uniform p=0.5 — test results showed that uniform probability yields the
    # most stable and reproducible augmentation rankings across seeds.
    augs: list = [
        _named("basic_augmentation", "basic_augmentation",
               probability=0.50, random_state=seed),
        _named("church_noise", "church_noise",
               probability=0.50, random_state=seed + 1),
        _named("dif_presets", "dif_presets",
               probability=0.50, random_state=seed + 2),
        _named("drust", "drust",
               probability=0.50, random_state=seed + 3),
        _named("luxfer_glass", "luxfer_glass",
               probability=0.50, random_state=seed + 4),
        _named("procam", "procam",
               probability=0.50, random_state=seed + 5),
        _named("smugs", "smugs",
               probability=0.50, random_state=seed + 6),
        _named("tea_stains", "tea_stains",
               probability=0.50, random_state=seed + 7),
    ]

    # ── Wave 2 — XAI-driven augmentations ──────────────────────────────────
    icd = ICD(
        model=model, target_layers=target_layers,
        threshold_percentile=70.0, explainer="opticam",
        use_cuda=use_cuda, cache=xai_cache,
        probability=0.50, random_state=seed + 10,
    )
    icd.name = "icd"
    augs.append(icd)

    aicd = AICD(
        model=model, target_layers=target_layers,
        threshold_percentile=70.0, explainer="opticam",
        use_cuda=use_cuda, cache=xai_cache,
        probability=0.50, random_state=seed + 11,
    )
    aicd.name = "aicd"
    augs.append(aicd)

    if quick:
        return [icd, aicd]  # Only ICD + AICD for fast testing

    # ── Wave 3 — parameter variants (stronger / different settings) ────────
    augs.extend([
        _named("basic_augmentation", "basic_aug_strong",
               probability=0.50, random_state=seed + 20),
        _named("church_noise", "church_noise_heavy",
               probability=0.50, random_state=seed + 21),
        _named("dif_presets", "dif_presets_strong",
               probability=0.50, random_state=seed + 22),
        _named("drust", "drust_dense",
               probability=0.50, random_state=seed + 23),
        _named("luxfer_glass", "luxfer_glass_strong",
               probability=0.50, random_state=seed + 24),
        _named("procam", "procam_strong",
               probability=0.50, random_state=seed + 25),
        _named("smugs", "smugs_thick",
               probability=0.50, random_state=seed + 26),
        _named("tea_stains", "tea_stains_heavy",
               probability=0.50, random_state=seed + 27),
    ])

    return augs


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "BNNR STL-10 comprehensive showcase — all augmentations, "
            "100 epochs, OptiCAM XAI, live dashboard."
        ),
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("examples/configs/classification/stl10_showcase.yaml"),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Limit training set size (default: full 5 000)",
    )
    p.add_argument(
        "--max-val-samples", type=int, default=None,
        help="Limit validation set size (default: full 8 000)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--m-epochs", type=int, default=15,
        help="Epochs per branch per decision round (default: 15)",
    )
    p.add_argument(
        "--decisions", type=int, default=10,
        help="Number of decision rounds (default: 10  ->  up to 150 main-path epochs)",
    )
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    p.add_argument(
        "--quick", action="store_true",
        help="Quick test: full STL-10 data, 10 epochs × 4 decisions (~30–60 min GPU)",
    )

    # dashboard flags
    dash = p.add_mutually_exclusive_group()
    dash.add_argument("--with-dashboard", dest="with_dashboard", action="store_true")
    dash.add_argument("--without-dashboard", dest="with_dashboard", action="store_false")
    p.set_defaults(with_dashboard=True)
    p.add_argument("--dashboard-port", type=int, default=8080)

    auto = p.add_mutually_exclusive_group()
    auto.add_argument("--dashboard-auto-open", dest="dashboard_auto_open", action="store_true")
    auto.add_argument("--no-dashboard-auto-open", dest="dashboard_auto_open", action="store_false")
    p.set_defaults(dashboard_auto_open=True)
    p.add_argument(
        "--dashboard-build-frontend",
        action=argparse.BooleanOptionalAction, default=True,
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def _subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(cast(Sized, dataset))))))


def _pick_num_workers(preferred: int = 2) -> int:
    """Pick a safe DataLoader worker count for restricted environments."""
    if preferred <= 0:
        return 0
    try:
        ctx = mp.get_context("spawn")
        probe = ctx.Queue()
        probe.close()
        probe.join_thread()
        return preferred
    except Exception as exc:
        print(
            "[data] Multiprocessing workers unavailable "
            f"({exc.__class__.__name__}: {exc}) -> using num_workers=0",
        )
        return 0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # ── Quick-test overrides ─────────────────────────────────────────────
    # Key insight: BNNR augmentations act as REGULARIZATION. They only
    # improve metrics when the model is trained long enough to overfit.
    # With lr=3e-4 and 10 epochs, the VGG model starts overfitting on
    # STL-10, creating the sweet spot where augmentation selection shines.
    if args.quick:
        args.max_train_samples = args.max_train_samples or 3000   # enough for overfitting
        args.max_val_samples = args.max_val_samples or 3000       # representative eval
        args.m_epochs = 10                                         # enough to overfit
        args.decisions = 4                                         # 4 rounds × 10 ep = 40 ep main path

    # ── CLI → config (BNNRConfig is frozen — use model_copy) ─────────
    overrides: dict[str, object] = {
        "m_epochs": args.m_epochs,
        "max_iterations": args.decisions,
        "event_log_enabled": args.with_dashboard,
    }
    if args.quick:
        overrides["candidate_pruning_enabled"] = True
        overrides["candidate_pruning_relative_threshold"] = 0.7
        overrides["candidate_pruning_warmup_epochs"] = 3
    if config.device == "auto":
        overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config = config.model_copy(update=overrides)

    total_epochs = args.m_epochs * (args.decisions + 1)  # baseline + decisions
    print()
    print("=" * 64)
    print("  BNNR  ·  STL-10  Comprehensive Showcase")
    print("-" * 64)
    print(f"  Max main-path epochs  : ~{total_epochs}")
    print(f"  Decision rounds       : {args.decisions}")
    print(f"  Epochs per branch     : {args.m_epochs}")
    n_candidates = 2 if args.quick else 20
    print(f"  Augmentation candidates: {n_candidates} ({'ICD + AICD' if args.quick else 'all BNNR types + variants'})")
    print("  XAI method            : OptiCAM")
    print(f"  Device                : {config.device}")
    if args.quick:
        print("  Mode                  : QUICK TEST")
    print("=" * 64)
    print()

    # ── Dashboard ────────────────────────────────────────────────────────
    dashboard_url = ""
    if args.with_dashboard:
        dashboard_url = start_dashboard(
            config.report_dir,
            port=args.dashboard_port,
            auto_open=args.dashboard_auto_open,
            build_frontend=args.dashboard_build_frontend,
        )

    # ── Data  ────────────────────────────────────────────────────────────
    # NOTE: do NOT use transforms.Normalize() here.  BNNR augmentations
    # convert tensors to uint8 images internally; if the tensor has been
    # normalised (mean-subtracted / std-divided), the conversion to uint8
    # destroys the signal (all pixels become 0-2).  The model's BatchNorm
    # layers handle feature-level normalisation, so raw [0, 1] tensors
    # from ToTensor() are all we need.
    #
    # In --quick mode on CPU, we downscale 96→48 for ~4× faster
    # convolutions.  The model uses AdaptiveAvgPool2d(1) before the
    # classifier, so it handles any spatial size.
    transform_list: list = [transforms.ToTensor()]
    if args.quick:
        transform_list.insert(0, transforms.Resize((48, 48)))
    transform = transforms.Compose(transform_list)

    print("[data] Downloading / loading STL-10 (first run downloads ~2.5 GB) …")
    train_ds = datasets.STL10(
        str(args.data_dir), split="train", download=True, transform=transform,
    )
    val_ds = datasets.STL10(
        str(args.data_dir), split="test", download=True, transform=transform,
    )
    train_ds = _subset(train_ds, args.max_train_samples)
    val_ds = _subset(val_ds, args.max_val_samples)

    n_train = len(cast(Sized, train_ds))
    n_val = len(cast(Sized, val_ds))
    print(f"[data] Train: {n_train:,}   Val: {n_val:,}   "
          f"Classes: {len(STL10_CLASSES)} ({', '.join(STL10_CLASSES)})")

    workers = _pick_num_workers(2)
    train_loader = DataLoader(
        _IndexedDataset(train_ds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(config.device != "cpu"),
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        _IndexedDataset(val_ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(config.device != "cpu"),
        persistent_workers=workers > 0,
    )

    # ── Model ────────────────────────────────────────────────────────────
    if args.quick:
        model = STL10NetQuick(num_classes=10)
        print(f"[model] STL10NetQuick (~{sum(p.numel() for p in model.parameters()):,} params, fast CPU mode)")
    else:
        model = STL10Net(num_classes=10)
        print(f"[model] STL10Net (~{sum(p.numel() for p in model.parameters()):,} params)")
    target_layer = model.target_layer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,   # mild L2 — amplifies augmentation's regularisation benefit
    )
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.05),
        optimizer=optimizer,
        target_layers=[target_layer],
        device=config.device,
    )

    # ── XAI cache ────────────────────────────────────────────────────────
    xai_cache: XAICache | None = None
    if config.xai_cache_dir:
        xai_cache = XAICache(cache_dir=config.xai_cache_dir)

    # ── Augmentations ─────────────────────────────────────────────────────
    augmentations = _build_all_augmentations(
        model=model,
        target_layers=[target_layer],
        device=config.device,
        seed=config.seed,
        xai_cache=xai_cache,
        quick=args.quick,
    )
    print(f"\n[augs] Registered {len(augmentations)} augmentation candidates:")
    for i, aug in enumerate(augmentations, 1):
        print(f"       {i:2d}. {aug.name:<24s}  (p={aug.probability:.2f})")
    print()

    # ── Train ────────────────────────────────────────────────────────────
    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config)
    result = trainer.run()

    # ── Results ──────────────────────────────────────────────────────────
    events_path = result.report_json_path.parent / "events.jsonl"
    print()
    print("=" * 64)
    print("  STL-10 showcase finished!")
    print("-" * 64)
    print(f"  Best path      : {result.best_path}")
    print(f"  Best metrics   : {result.best_metrics}")
    print(f"  Report JSON    : {result.report_json_path}")
    print(f"  Events (JSONL) : {events_path}")
    if args.with_dashboard:
        print(f"  Dashboard      : {dashboard_url}")
    print("=" * 64)
    print()

    if args.with_dashboard:
        print("Dashboard is still running — press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
