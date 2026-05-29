"""Minimal ICD plug-in example — saliency-guided augmentation without BNNRTrainer.

Precomputes Grad-CAM saliency (via BNNR / pytorch-grad-cam), applies ICD in a hand-written
training loop on MNIST.

Run:
    PYTHONPATH=src python examples/classification/icd_plugin_minimal.py

Quick:
    PYTHONPATH=src python examples/classification/icd_plugin_minimal.py --epochs 1 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from bnnr.icd import ICD
from bnnr.xai_cache import XAICache

N_TRAIN = 128
N_VAL = 32
BATCH_SIZE = 16
SEED = 42


class IndexedDataset(Dataset):
    """Wrap (image, label) → (image, label, index) for XAICache / ICD."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, label, idx


class MnistCNN(nn.Module):
    """Small CNN for 1×28×28 MNIST (matches built-in demo scale)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def build_model(device: torch.device) -> tuple[MnistCNN, list[nn.Module]]:
    torch.manual_seed(SEED)
    model = MnistCNN().to(device)
    target_layers = [model.conv2]
    return model, target_layers


def _tensor_batch_to_uint8_hwc(images: torch.Tensor) -> np.ndarray:
    """BCHW float [0,1] or [0,255] → NHWC uint8 (1 or 3 channels)."""
    arr = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def _uint8_hwc_to_tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
    if t.max() > 1.0:
        t = t / 255.0
    return t.to(device)


def apply_icd_batch(
    icd: ICD,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    imgs_np = _tensor_batch_to_uint8_hwc(images)
    labels_np = labels.detach().cpu().numpy()
    idx_np = indices.detach().cpu().numpy()
    icd.probability = 1.0
    aug_np = icd.apply_batch_with_labels(imgs_np, labels_np, sample_indices=idx_np)
    return _uint8_hwc_to_tensor(aug_np, images.device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    icd: ICD,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, labels, indices in loader:
        images = images.to(device)
        labels = labels.to(device)
        indices = indices.to(device)

        images = apply_icd_batch(icd, images, labels, indices)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICD plug-in minimal training loop (no BNNRTrainer)"
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("icd_plugin_out"))
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "xai_cache"

    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(
        root=str(args.output_dir / "mnist_data"), train=True, download=True, transform=transform
    )
    full_val = datasets.MNIST(
        root=str(args.output_dir / "mnist_data"), train=False, download=True, transform=transform
    )

    train_base = torch.utils.data.Subset(full_train, range(N_TRAIN))
    val_base = torch.utils.data.Subset(full_val, range(N_VAL))
    train_loader = DataLoader(IndexedDataset(train_base), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(IndexedDataset(val_base), batch_size=BATCH_SIZE, shuffle=False)

    model, target_layers = build_model(device)
    xai_cache = XAICache(cache_dir)
    print(f"[1/3] Precomputing XAI cache (method=gradcam) → {cache_dir}")
    written = xai_cache.precompute_cache(
        model=model,
        train_loader=train_loader,
        target_layers=target_layers,
        n_samples=N_TRAIN,
        method="gradcam",
        show_progress=False,
    )
    print(f"      Cached {written} saliency maps")

    icd = ICD(
        model=model,
        target_layers=target_layers,
        cache=xai_cache,
        explainer="gradcam",
        probability=1.0,
        random_state=SEED,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"[2/3] Training {args.epochs} epoch(s) with ICD augmentation …")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, icd, optimizer, criterion, device)
        acc = evaluate(model, val_loader, device)
        print(f"      epoch {epoch + 1}: train_loss={loss:.4f}  val_acc={acc:.3f}")

    n_maps = len(list(cache_dir.glob("*.npy")))
    print(f"[3/3] Done. Cache files: {n_maps} under {cache_dir}")
    print("      See docs/plugin_icd.md for integration details.")


if __name__ == "__main__":
    main()
