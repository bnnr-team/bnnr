"""Torchvision **pretrained ResNet50** on CIFAR-100 → BNNR analyze report.

This is the "I have a torchvision model, give me a failure report" path from the
2026 growth audit: take a standard ``torchvision.models.resnet50`` (ImageNet
weights), point ``bnnr analyze`` at it, and get a portable HTML + JSON report
with confusion matrix, top confused pairs, and OptiCAM saliency overlays — no
BNNRTrainer, no retraining of the backbone.

The backbone normalizes inputs *inside* the model (ImageNet mean/std as
registered buffers), so the val loader can feed plain ``ToTensor()`` tensors in
``[0, 1]`` — the same convention the rest of BNNR uses.

Install:
    pip install bnnr

Meaningful report (load your fine-tuned 100-class checkpoint):
    PYTHONPATH=src python examples/analyze_resnet50.py \\
        --checkpoint path/to/resnet50_cifar100.pt --device cuda

Quick look (trains only the classifier head for a few steps on a subset):
    PYTHONPATH=src python examples/analyze_resnet50.py --quick --device cuda

CI smoke (no download, no pretrained weights, random data):
    PYTHONPATH=src python examples/analyze_resnet50.py \\
        --output-dir /tmp/out --synthetic --no-xai --device cpu

Note: with the bare ImageNet backbone and an *untrained* 100-class head the
predictions are not meaningful — pass ``--checkpoint`` (or use ``--quick`` to
warm the head) for a report worth reading. The point of this example is the
torchvision → analyze plumbing.

Citation: docs/citation.md in the bnnr repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from bnnr.adapter import SimpleTorchAdapter
from bnnr.analyze import analyze_model
from bnnr.core import BNNRConfig

SEED = 42
NUM_CLASSES = 100
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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

    def __init__(self, n: int, img_size: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, 3, img_size, img_size, generator=gen)
        self.labels = torch.arange(n, dtype=torch.long) % NUM_CLASSES

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx].item())


class NormalizedResNet50(nn.Module):
    """ResNet50 that normalizes [0, 1] inputs with ImageNet stats internally."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.backbone(x)


def _cifar100_loaders(
    data_root: Path,
    *,
    synthetic: bool,
    quick: bool,
    img_size: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    if synthetic:
        train_ds: Dataset = _SyntheticCifarSubset(32, img_size, seed=seed)
        val_ds: Dataset = _SyntheticCifarSubset(24, img_size, seed=seed + 1)
    else:
        transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )
        train_full = datasets.CIFAR100(str(data_root), train=True, download=True, transform=transform)
        val_full = datasets.CIFAR100(str(data_root), train=False, download=True, transform=transform)
        if quick:
            gen = torch.Generator().manual_seed(seed)
            train_idx = torch.randperm(len(train_full), generator=gen)[:512].tolist()
            val_idx = torch.randperm(len(val_full), generator=gen)[:256].tolist()
            train_ds = Subset(train_full, train_idx)
            val_ds = Subset(val_full, val_idx)
        else:
            train_ds, val_ds = train_full, val_full

    train_loader = DataLoader(IndexedDataset(train_ds), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(IndexedDataset(val_ds), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _train_head(adapter: SimpleTorchAdapter, train_loader: DataLoader, device: torch.device, epochs: int) -> None:
    """Warm the classifier head so the report is not pure noise."""
    adapter.model.train()
    for ep in range(epochs):
        for images, labels, _idx in train_loader:
            images, labels = images.to(device), labels.to(device)
            adapter.optimizer.zero_grad()
            loss = adapter.criterion(adapter.model(images), labels)
            loss.backward()
            adapter.optimizer.step()
        print(f"  head-warm epoch {ep + 1}/{epochs} done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torchvision pretrained ResNet50 on CIFAR-100 → BNNR analyze report",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analyze_resnet50_out"))
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224, help="Resize target (224 uses pretrained features)")
    parser.add_argument("--quick", action="store_true", help="Small subset + warm head for a few steps")
    parser.add_argument("--synthetic", action="store_true", help="In-memory data only (no download; for CI)")
    parser.add_argument("--no-xai", action="store_true", help="Skip XAI in analyze")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ImageNet-pretrained backbone (--no-pretrained for CI / offline)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Load a fine-tuned state_dict (.pt)")
    parser.add_argument("--xai-samples", type=int, default=64, help="Probe set size for XAI (lower = faster)")
    args = parser.parse_args()

    if args.synthetic:
        args.pretrained = False
        if args.img_size == 224:
            args.img_size = 64  # keep CI fast

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.output_dir / "cifar_data"

    train_loader, val_loader = _cifar100_loaders(
        data_root,
        synthetic=args.synthetic,
        quick=args.quick or args.synthetic,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = NormalizedResNet50(num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.backbone.layer4[-1]],
        device=str(device),
    )

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
    elif args.quick and not args.synthetic:
        print("Warming classifier head for 1 epoch on a CIFAR-100 subset …")
        _train_head(adapter, train_loader, device, epochs=1)
    elif args.synthetic:
        print("Synthetic mode: skipping training (random weights)")
    else:
        print(
            "Note: using the bare ImageNet backbone with an untrained 100-class head. "
            "Predictions are not meaningful — pass --checkpoint or --quick for a real report."
        )

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
