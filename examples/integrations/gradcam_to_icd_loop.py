"""Grad-CAM heatmaps → BNNR ICD on the same batch (pytorch-grad-cam integration bridge).

Install:
    pip install bnnr

Run:
    PYTHONPATH=src python examples/integrations/gradcam_to_icd_loop.py

Outputs:
    gradcam_icd_out/gradcam_overlay.png
    gradcam_icd_out/icd_augmented.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from bnnr.icd import ICD
from bnnr.xai_cache import XAICache

SEED = 42
BATCH_SIZE = 4


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, label, idx


def _rgb_float_from_tensor(image: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] → HWC float [0,1] RGB."""
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM vs BNNR ICD on one CIFAR-10 batch")
    parser.add_argument("--output-dir", type=Path, default=Path("gradcam_icd_out"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    val_set = datasets.CIFAR10(
        root=str(args.output_dir / "cifar_data"),
        train=False,
        download=True,
        transform=transform,
    )
    subset = Subset(val_set, range(BATCH_SIZE))
    loader = DataLoader(IndexedDataset(subset), batch_size=BATCH_SIZE, shuffle=False)

    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()
    target_layers = [model.layer4[-1]]

    images, labels, indices = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    # ── Step 1: raw pytorch-grad-cam ─────────────────────────────────────
    targets = [ClassifierOutputTarget(int(labels[i])) for i in range(BATCH_SIZE)]
    cam_kwargs: dict = {"model": model, "target_layers": target_layers}
    with GradCAM(**cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=images, targets=targets)
    rgb = _rgb_float_from_tensor(images[0])
    cam_image = show_cam_on_image(rgb, grayscale_cam[0], use_rgb=True)

    # ── Step 2: BNNR ICD with gradcam saliency (precomputed cache) ─────
    cache_dir = args.output_dir / "xai_cache"
    cache = XAICache(cache_dir)
    cache.precompute_cache(
        model=model,
        train_loader=loader,
        target_layers=target_layers,
        n_samples=BATCH_SIZE,
        method="gradcam",
        force_recompute=True,
        show_progress=False,
    )
    icd = ICD(
        model=model,
        target_layers=target_layers,
        cache=cache,
        explainer="gradcam",
        probability=1.0,
        random_state=args.seed,
    )
    img0_uint8 = (rgb * 255.0).astype(np.uint8)
    icd_out = icd.apply_with_label(
        img0_uint8,
        int(labels[0].item()),
        sample_index=int(indices[0].item()),
    )
    if icd_out.ndim == 2:
        icd_rgb = np.stack([icd_out] * 3, axis=-1)
    elif icd_out.shape[-1] == 1:
        icd_rgb = np.repeat(icd_out, 3, axis=2)
    else:
        icd_rgb = icd_out
    icd_rgb = icd_rgb.astype(np.float32) / 255.0

    # ── Save side-by-side figures ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cam_image)
    axes[0].set_title("Grad-CAM overlay (pytorch-grad-cam)")
    axes[0].axis("off")
    axes[1].imshow(np.clip(icd_rgb, 0, 1))
    axes[1].set_title("After BNNR ICD (gradcam saliency)")
    axes[1].axis("off")
    fig.tight_layout()
    overlay_path = args.output_dir / "gradcam_overlay.png"
    icd_path = args.output_dir / "icd_augmented.png"
    fig.savefig(overlay_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    plt.imsave(icd_path, np.clip(icd_rgb, 0, 1))

    print(f"Saved {overlay_path}")
    print(f"Saved {icd_path}")
    print("See docs/plugin_icd.md for a full training-loop example.")


if __name__ == "__main__":
    main()
