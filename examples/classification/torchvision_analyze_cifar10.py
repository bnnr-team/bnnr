"""Torchvision ResNet-18 on CIFAR-10 → BNNR analyze report (Python API).

Trains (or loads) a torchvision ResNet-18 classifier, then runs ``analyze_model``
for a portable HTML/JSON failure + XAI report — without BNNRTrainer.

Install:
    pip install bnnr

Run (clone repo for PYTHONPATH=src):
    PYTHONPATH=src python examples/classification/torchvision_analyze_cifar10.py --quick

CI smoke (no download):
    PYTHONPATH=src python examples/classification/torchvision_analyze_cifar10.py \\
        --output-dir /tmp/out --synthetic --no-xai --device cpu

Citation: docs/citation.md in the bnnr repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from bnnr.adapter import SimpleTorchAdapter
from bnnr.analyze import analyze_model
from bnnr.core import BNNRConfig

SEED = 42
NUM_CLASSES = 10


class IndexedDataset(Dataset):
    """Wrap (image, label) → (image, label, index) for analyze."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, label, idx


class _SyntheticCifarSubset(Dataset):
    """In-memory CIFAR-shaped batch for CI (no download)."""

    def __init__(self, n: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, 3, 32, 32, generator=gen)
        self.labels = torch.arange(n, dtype=torch.long) % NUM_CLASSES

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx].item())


def _build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    return resnet18(weights=None, num_classes=num_classes)


def _cifar_loaders(
    data_root: Path,
    *,
    synthetic: bool,
    quick: bool,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    if synthetic:
        n_train, n_val = 32, 24
        train_ds: Dataset = _SyntheticCifarSubset(n_train, seed=seed)
        val_ds: Dataset = _SyntheticCifarSubset(n_val, seed=seed + 1)
    else:
        train_full = datasets.CIFAR10(
            str(data_root),
            train=True,
            download=True,
            transform=transform,
        )
        val_full = datasets.CIFAR10(
            str(data_root),
            train=False,
            download=True,
            transform=transform,
        )
        if quick:
            gen = torch.Generator().manual_seed(seed)
            train_idx = torch.randperm(len(train_full), generator=gen)[:256].tolist()
            val_idx = torch.randperm(len(val_full), generator=gen)[:128].tolist()
            train_ds = Subset(train_full, train_idx)
            val_ds = Subset(val_full, val_idx)
        else:
            train_ds = train_full
            val_ds = val_full

    train_loader = DataLoader(
        IndexedDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        IndexedDataset(val_ds),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def _train_one_epoch(
    adapter: SimpleTorchAdapter,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    adapter.model.train()
    for batch in train_loader:
        images, labels, _indices = batch
        images = images.to(device)
        labels = labels.to(device)
        adapter.optimizer.zero_grad()
        logits = adapter.model(images)
        loss = adapter.criterion(logits, labels)
        loss.backward()
        adapter.optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torchvision ResNet-18 on CIFAR-10 → BNNR analyze HTML report",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("torchvision_analyze_out"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small CIFAR-10 subset + 1 training epoch before analyze",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="In-memory data only (no CIFAR-10 download; for CI)",
    )
    parser.add_argument("--no-xai", action="store_true", help="Skip XAI in analyze")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Load ResNet-18 state_dict from .pt (skip training)",
    )
    parser.add_argument(
        "--xai-samples",
        type=int,
        default=32,
        help="Probe set size for XAI (lower = faster)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.output_dir / "cifar_data"

    train_loader, val_loader = _cifar_loaders(
        data_root,
        synthetic=args.synthetic,
        quick=args.quick or args.synthetic,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = _build_resnet18().to(device)
    target_layers = [model.layer4[-1]]
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=target_layers,
        device=str(device),
    )

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint from {args.checkpoint}")
    elif not args.synthetic:
        epochs = 1 if args.quick else 3
        print(f"Training ResNet-18 for {epochs} epoch(s) on CIFAR-10 subset …")
        for ep in range(epochs):
            _train_one_epoch(adapter, train_loader, device)
            print(f"  epoch {ep + 1}/{epochs} done")
    else:
        print("Synthetic mode: skipping training (random weights)")

    config = BNNRConfig(device=str(device), task="classification")
    print("Running analyze_model …")
    report = analyze_model(
        adapter,
        val_loader,
        config=config,
        output_dir=args.output_dir,
        run_data_quality=not args.synthetic,
        xai_enabled=not args.no_xai,
        xai_samples=min(args.xai_samples, len(val_loader.dataset)),  # type: ignore[arg-type]
    )
    html_path = args.output_dir / "report.html"
    report.to_html(html_path, artifact_root=args.output_dir, embed_images=True)
    print(f"Saved {args.output_dir / 'analysis_report.json'}")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()
