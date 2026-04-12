"""
Pascal VOC 2007 comprehensive detection showcase — demonstrates BNNR with
ALL detection augmentation types, XAI saliency maps, and long training runs.

VOC 2007: 20 classes of real-world objects with bounding box annotations —
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor.

5,011 trainval images + 4,952 test images.  Auto-downloads on first run
(~460 MB, cached afterwards).

9 augmentation types available (all BNNR detection types):
  • 4 geometric bbox-aware     (HFlip, VFlip, Rotate90, RandomScale)
  • 2 XAI-driven               (DetectionICD, DetectionAICD)
  • 2 composite                (Mosaic, MixUp)
  • 1 colour jitter via Albu   (optional, if albumentations installed)

Full mode registers 13 candidates (14 with albumentations) including
parameter variants.  Quick mode registers 6 (geometric + ICD/AICD).

Run with dashboard (recommended):
    PYTHONPATH=src python examples/detection/showcase_voc.py --with-dashboard

Quick test (CPU-friendly, ~15–20 min):
    PYTHONPATH=src python examples/detection/showcase_voc.py --with-dashboard --quick

Full run (GPU recommended, ~2–3 h):
    PYTHONPATH=src python examples/detection/showcase_voc.py --with-dashboard \\
        --m-epochs 5 --decisions 6

Without dashboard:
    PYTHONPATH=src python examples/detection/showcase_voc.py --without-dashboard
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import shutil
import tarfile
import urllib.request
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VOCDetection

from bnnr import BNNRConfig, BNNRTrainer, start_dashboard
from bnnr.config import load_config
from bnnr.detection_adapter import DetectionAdapter
from bnnr.detection_augmentations import (
    DetectionHorizontalFlip,
    DetectionMixUp,
    DetectionRandomRotate90,
    DetectionRandomScale,
    DetectionVerticalFlip,
    MosaicAugmentation,
)
from bnnr.detection_collate import detection_collate_fn_with_index
from bnnr.detection_icd import DetectionAICD, DetectionICD
from bnnr.reporting import Reporter

# ── VOC class names ──────────────────────────────────────────────────────
VOC_CLASSES: list[str] = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}  # 0 = bg
NUM_CLASSES = len(VOC_CLASSES) + 1  # +1 for background


# ═══════════════════════════════════════════════════════════════════════════
#  VOC Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════

class VOCDetectionDataset(Dataset):
    """Wraps torchvision.VOCDetection for BNNR detection format.

    Returns ``(image_tensor, target_dict, index)`` per sample:
        image_tensor : [3, H, W] float32 ∈ [0, 1]
        target_dict  : {"boxes": [N, 4] xyxy float32, "labels": [N] int64}
        index        : int
    """

    def __init__(
        self,
        root: str = "./data",
        year: str = "2007",
        image_set: str = "trainval",
        target_size: int = 300,
    ) -> None:
        self.root = Path(root)
        self.target_size = target_size

        # Strategy 1 — data already exists locally
        try:
            self.voc = VOCDetection(
                root=str(self.root), year=year,
                image_set=image_set, download=False,
            )
            return
        except RuntimeError:
            pass

        # Strategy 2 — torchvision built-in download
        try:
            print("  ↳ Trying torchvision download …")
            self.voc = VOCDetection(
                root=str(self.root), year=year,
                image_set=image_set, download=True,
            )
            return
        except Exception:
            pass

        # Strategy 3 — custom mirrors with User-Agent header
        try:
            print("  ↳ Trying alternative mirrors …")
            _download_voc_from_mirrors(self.root, image_set)
            self.voc = VOCDetection(
                root=str(self.root), year=year,
                image_set=image_set, download=False,
            )
            return
        except Exception:
            pass

        raise RuntimeError(
            "VOC 2007 could not be loaded — all download mirrors are unavailable."
        )

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor], int]:
        img, annotation = self.voc[index]
        orig_w, orig_h = img.size

        img_tensor = TF.resize(img, [self.target_size, self.target_size])
        img_tensor = TF.to_tensor(img_tensor)

        objs = annotation["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]

        boxes: list[list[float]] = []
        labels: list[int] = []
        for obj in objs:
            if obj.get("difficult", "0") == "1":
                continue
            name = obj["name"]
            if name not in VOC_CLASS_TO_IDX:
                continue
            bbox = obj["bndbox"]
            x1 = float(bbox["xmin"]) * self.target_size / orig_w
            y1 = float(bbox["ymin"]) * self.target_size / orig_h
            x2 = float(bbox["xmax"]) * self.target_size / orig_w
            y2 = float(bbox["ymax"]) * self.target_size / orig_h
            boxes.append([x1, y1, x2, y2])
            labels.append(VOC_CLASS_TO_IDX[name])

        if not boxes:
            boxes.append([0, 0, 1, 1])
            labels.append(0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_tensor, target, index


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════

class VOCDetectionModel(nn.Module):
    """Faster R-CNN with MobileNet-v3 backbone for VOC 2007.

    Uses ImageNet-pretrained backbone (``weights_backbone="DEFAULT"``)
    for meaningful feature extraction from the start.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone="DEFAULT",
            num_classes=num_classes,
        )

    def forward(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        return self.model(images, targets)


# ═══════════════════════════════════════════════════════════════════════════
#  Augmentations — ALL BNNR detection types + parameter variants
# ═══════════════════════════════════════════════════════════════════════════

def _build_all_detection_augmentations(
    seed: int,
    target_size: int = 300,
    *,
    quick: bool = False,
) -> list:
    """Build augmentation candidates.

    Available BNNR detection augmentation types (9):
      DetectionHorizontalFlip, DetectionVerticalFlip,
      DetectionRandomRotate90, DetectionRandomScale,
      DetectionICD, DetectionAICD,
      MosaicAugmentation, DetectionMixUp,
      AlbumentationsBboxAugmentation (colour jitter, optional)

    Full mode (14 candidates):
      • 4 geometric bbox-aware at default strength
      • 2 XAI-driven (DetectionICD, DetectionAICD)
      • 2 composite (Mosaic, MixUp)
      • 1 Albumentations colour jitter (if installed)
      • 5 parameter variants (stronger/different settings)

    Quick mode (6 candidates):
      • 4 geometric + DetectionICD + DetectionAICD
    """
    augs: list = []

    # ── Wave 1 — geometric bbox-aware augmentations ──────────────────
    augs.extend([
        DetectionHorizontalFlip(
            probability=0.5,
            name_override="det_hflip",
            random_state=seed,
        ),
        DetectionVerticalFlip(
            probability=0.5,
            name_override="det_vflip",
            random_state=seed + 1,
        ),
        DetectionRandomRotate90(
            probability=0.5,
            name_override="det_rotate90",
            random_state=seed + 2,
        ),
        DetectionRandomScale(
            scale_range=(0.85, 1.15),
            probability=0.5,
            name_override="det_scale",
            random_state=seed + 3,
        ),
    ])

    # ── Wave 2 — XAI-driven detection augmentations ──────────────────
    icd = DetectionICD(
        probability=0.5,
        threshold_percentile=70,
        tile_size=8,
        fill_strategy="gaussian_blur",
        name_override="det_icd",
        random_state=seed + 10,
    )
    augs.append(icd)

    aicd = DetectionAICD(
        probability=0.5,
        threshold_percentile=70,
        tile_size=8,
        fill_strategy="gaussian_blur",
        name_override="det_aicd",
        random_state=seed + 11,
    )
    augs.append(aicd)

    if quick:
        return augs

    # ── Wave 3 — composite augmentations ─────────────────────────────
    augs.extend([
        MosaicAugmentation(
            output_size=(target_size, target_size),
            probability=0.5,
            name_override="det_mosaic",
            random_state=seed + 20,
        ),
        DetectionMixUp(
            alpha_range=(0.3, 0.7),
            probability=0.5,
            name_override="det_mixup",
            random_state=seed + 21,
        ),
    ])

    # ── Wave 4 — Albumentations colour jitter (optional) ─────────────
    try:
        import albumentations as alb
        from bnnr.detection_augmentations import AlbumentationsBboxAugmentation

        augs.append(
            AlbumentationsBboxAugmentation(
                alb.Compose(
                    [
                        alb.RandomBrightnessContrast(p=0.5),
                        alb.HueSaturationValue(p=0.5),
                    ],
                    bbox_params=alb.BboxParams(
                        format="pascal_voc",
                        label_fields=["labels"],
                        min_visibility=0.2,
                    ),
                ),
                name_override="det_color_jitter",
                probability=0.5,
                random_state=seed + 30,
            )
        )
    except ImportError:
        pass

    # ── Wave 5 — parameter variants (stronger settings) ──────────────
    augs.extend([
        DetectionHorizontalFlip(
            probability=0.5,
            name_override="det_hflip_strong",
            random_state=seed + 40,
        ),
        DetectionRandomScale(
            scale_range=(0.7, 1.3),
            probability=0.5,
            name_override="det_scale_wide",
            random_state=seed + 41,
        ),
        DetectionICD(
            probability=0.5,
            threshold_percentile=60,
            tile_size=12,
            fill_strategy="noise",
            name_override="det_icd_aggressive",
            random_state=seed + 42,
        ),
        DetectionAICD(
            probability=0.5,
            threshold_percentile=60,
            tile_size=12,
            fill_strategy="gaussian_blur",
            name_override="det_aicd_strong",
            random_state=seed + 43,
        ),
        DetectionRandomRotate90(
            probability=0.5,
            name_override="det_rotate90_freq",
            random_state=seed + 44,
        ),
    ])

    return augs


# ═══════════════════════════════════════════════════════════════════════════
#  Robust VOC download — multiple mirrors with proper headers
# ═══════════════════════════════════════════════════════════════════════════

_VOC_MIRRORS: dict[str, list[str]] = {
    "VOCdevkit_08-Jun-2007.tar": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar",
        "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar",
        "https://pjreddie.com/media/files/VOCdevkit_08-Jun-2007.tar",
    ],
    "VOCtrainval_06-Nov-2007.tar": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
    ],
    "VOCtest_06-Nov-2007.tar": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
    ],
}

_SPLIT_TO_ARCHIVES: dict[str, list[str]] = {
    "trainval": ["VOCdevkit_08-Jun-2007.tar", "VOCtrainval_06-Nov-2007.tar"],
    "train":    ["VOCdevkit_08-Jun-2007.tar", "VOCtrainval_06-Nov-2007.tar"],
    "val":      ["VOCdevkit_08-Jun-2007.tar", "VOCtrainval_06-Nov-2007.tar"],
    "test":     ["VOCdevkit_08-Jun-2007.tar", "VOCtest_06-Nov-2007.tar"],
}


def _download_voc_from_mirrors(root: Path, image_set: str) -> None:
    """Download VOC 2007 archives from multiple mirrors with User-Agent."""
    root.mkdir(parents=True, exist_ok=True)
    archives = _SPLIT_TO_ARCHIVES.get(image_set, _SPLIT_TO_ARCHIVES["trainval"])

    for filename in archives:
        target = root / filename
        if target.exists():
            try:
                with tarfile.open(target):
                    pass
                with tarfile.open(target) as tar:
                    tar.extractall(path=root)
                print(f"    ✓ {filename} (cached)")
                continue
            except tarfile.TarError:
                target.unlink(missing_ok=True)

        mirrors = _VOC_MIRRORS.get(filename, [])
        downloaded = False
        for url in mirrors:
            try:
                print(f"    ↳ {url}")
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (BNNR)"},
                )
                with (
                    urllib.request.urlopen(req, timeout=180) as resp,
                    target.open("wb") as out,
                ):
                    shutil.copyfileobj(resp, out)
                # Verify & extract
                with tarfile.open(target) as tar:
                    tar.extractall(path=root)
                print(f"    ✓ {filename}")
                downloaded = True
                break
            except Exception as exc:
                target.unlink(missing_ok=True)
                print(f"      ✗ {exc}")
                continue

        if not downloaded:
            raise RuntimeError(f"All mirrors failed for {filename}")


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic detection dataset — guaranteed fallback (no downloads)
# ═══════════════════════════════════════════════════════════════════════════

# Distinct class colours (R, G, B) — one per shape type
_SHAPE_COLORS: list[tuple[float, float, float]] = [
    (0.90, 0.22, 0.21),  # red     — square
    (0.30, 0.69, 0.31),  # green   — circle
    (0.25, 0.32, 0.91),  # blue    — triangle
    (0.98, 0.75, 0.18),  # yellow  — cross
    (0.73, 0.33, 0.83),  # purple  — diamond
]


class SyntheticDetectionDataset(Dataset):
    """Procedural detection dataset — zero downloads, works everywhere.

    Generates RGB images with coloured geometric shapes and ground-truth
    bounding boxes.  Used automatically when VOC 2007 cannot be downloaded.

    Returns ``(image_tensor, target_dict, index)`` — same interface as
    ``VOCDetectionDataset``.
    """

    CLASS_NAMES: list[str] = ["square", "circle", "triangle", "cross", "diamond"]

    def __init__(
        self,
        n_samples: int = 500,
        target_size: int = 300,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.target_size = target_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    # ── drawing helpers ───────────────────────────────────────────────

    @staticmethod
    def _draw_square(
        img: Tensor, y1: int, y2: int, x1: int, x2: int, r: float, g: float, b: float,
    ) -> None:
        img[0, y1:y2, x1:x2] = r
        img[1, y1:y2, x1:x2] = g
        img[2, y1:y2, x1:x2] = b

    @staticmethod
    def _draw_circle(
        img: Tensor, y1: int, y2: int, x1: int, x2: int, r: float, g: float, b: float,
    ) -> None:
        S = img.shape[-1]
        cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
        ry, rx = (y2 - y1) / 2.0, (x2 - x1) / 2.0
        yy, xx = torch.meshgrid(
            torch.arange(S, dtype=torch.float32),
            torch.arange(S, dtype=torch.float32),
            indexing="ij",
        )
        mask = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
        img[0, mask] = r
        img[1, mask] = g
        img[2, mask] = b

    @staticmethod
    def _draw_triangle(
        img: Tensor, y1: int, y2: int, x1: int, x2: int, r: float, g: float, b: float,
    ) -> None:
        h = y2 - y1
        cx = (x1 + x2) // 2
        w = x2 - x1
        for row in range(y1, y2):
            frac = (row - y1) / max(h, 1)
            half = int(frac * w / 2)
            left = max(0, cx - half)
            right = min(img.shape[-1], cx + half + 1)
            img[0, row, left:right] = r
            img[1, row, left:right] = g
            img[2, row, left:right] = b

    @staticmethod
    def _draw_cross(
        img: Tensor, y1: int, y2: int, x1: int, x2: int, r: float, g: float, b: float,
    ) -> None:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        arm_w = max(2, (x2 - x1) // 5)
        arm_h = max(2, (y2 - y1) // 5)
        # vertical bar
        img[0, y1:y2, cx - arm_w : cx + arm_w] = r
        img[1, y1:y2, cx - arm_w : cx + arm_w] = g
        img[2, y1:y2, cx - arm_w : cx + arm_w] = b
        # horizontal bar
        img[0, cy - arm_h : cy + arm_h, x1:x2] = r
        img[1, cy - arm_h : cy + arm_h, x1:x2] = g
        img[2, cy - arm_h : cy + arm_h, x1:x2] = b

    @staticmethod
    def _draw_diamond(
        img: Tensor, y1: int, y2: int, x1: int, x2: int, r: float, g: float, b: float,
    ) -> None:
        S = img.shape[-1]
        cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
        hw, hh = (x2 - x1) / 2.0, (y2 - y1) / 2.0
        yy, xx = torch.meshgrid(
            torch.arange(S, dtype=torch.float32),
            torch.arange(S, dtype=torch.float32),
            indexing="ij",
        )
        mask = (torch.abs(xx - cx) / max(hw, 1) + torch.abs(yy - cy) / max(hh, 1)) <= 1.0
        img[0, mask] = r
        img[1, mask] = g
        img[2, mask] = b

    _DRAWERS = [_draw_square, _draw_circle, _draw_triangle, _draw_cross, _draw_diamond]

    # ── __getitem__ ───────────────────────────────────────────────────

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor], int]:
        rng = random.Random(self.seed + index)
        S = self.target_size

        # Gradient background
        img = torch.zeros(3, S, S)
        br, bg, bb = rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2)
        axis = 0 if rng.random() > 0.5 else 1
        for ch, base in enumerate([br, bg, bb]):
            grad = torch.linspace(base, base + rng.uniform(0.05, 0.15), S)
            img[ch] = grad.unsqueeze(axis).expand(S, S)

        # Add subtle noise
        img += torch.randn_like(img) * 0.02
        img.clamp_(0.0, 1.0)

        n_objects = rng.randint(1, 4)
        boxes: list[list[float]] = []
        labels: list[int] = []

        for _ in range(n_objects):
            cls_idx = rng.randint(0, len(self.CLASS_NAMES) - 1)
            obj_w = rng.randint(S // 8, S // 3)
            obj_h = rng.randint(S // 8, S // 3)
            x1 = rng.randint(0, max(0, S - obj_w - 1))
            y1 = rng.randint(0, max(0, S - obj_h - 1))
            x2, y2 = x1 + obj_w, y1 + obj_h

            cr, cg, cb = _SHAPE_COLORS[cls_idx]
            cr += rng.uniform(-0.08, 0.08)
            cg += rng.uniform(-0.08, 0.08)
            cb += rng.uniform(-0.08, 0.08)

            self._DRAWERS[cls_idx].__func__(img, y1, y2, x1, x2, cr, cg, cb)  # type: ignore[attr-defined]

            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(cls_idx + 1)  # 0 = background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target, index


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "BNNR VOC 2007 comprehensive detection showcase — all augmentation "
            "types, detection XAI, live dashboard."
        ),
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("examples/configs/detection/voc_showcase.yaml"),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Limit training set size (default: full 5 011)",
    )
    p.add_argument(
        "--max-val-samples", type=int, default=None,
        help="Limit validation set size (default: full 4 952)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--m-epochs", type=int, default=5,
        help="Epochs per branch per decision round (default: 5)",
    )
    p.add_argument(
        "--decisions", type=int, default=6,
        help="Number of decision rounds (default: 6 → up to 35 main-path epochs)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: auto -> 0.001 on CPU, 0.005 on CUDA)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick test: 500 samples, 3 epochs × 3 decisions (~20 min GPU)",
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

    p.add_argument("--target-size", type=int, default=300,
                    help="Resize images to this size (default: 300)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

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

    # ── Load YAML config (if available) ──────────────────────────────
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = BNNRConfig(task="detection")

    # ── Quick-test overrides ─────────────────────────────────────────
    if args.quick:
        args.max_train_samples = args.max_train_samples or 500
        args.max_val_samples = args.max_val_samples or 250
        args.m_epochs = 3
        args.decisions = 3

    # ── Determine device early ────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.device != "auto":
        device = config.device
    if args.lr is None:
        args.lr = 0.001 if device == "cpu" else 0.005

    # ── Data ─────────────────────────────────────────────────────────
    #   Try VOC 2007 → automatic fallback to synthetic shapes dataset
    class_names: list[str] = VOC_CLASSES
    num_classes: int = NUM_CLASSES
    dataset_name = "VOC 2007"
    use_synthetic = False

    print("[data] Loading Pascal VOC 2007 (auto-downloads ~460 MB on first run)…")
    try:
        train_ds: Dataset = VOCDetectionDataset(
            root=str(args.data_dir),
            image_set="trainval",
            target_size=args.target_size,
        )
        val_ds: Dataset = VOCDetectionDataset(
            root=str(args.data_dir),
            image_set="test",
            target_size=args.target_size,
        )
    except Exception as exc:
        print(f"\n  ⚠ VOC 2007 unavailable: {exc}")
        print("  → Falling back to synthetic detection dataset (shapes with bboxes)")
        print("    Tip: to use real VOC data, manually extract VOCdevkit/ into ./data/\n")
        use_synthetic = True
        class_names = SyntheticDetectionDataset.CLASS_NAMES
        num_classes = len(class_names) + 1
        dataset_name = "Synthetic Shapes"
        n_synth_train = args.max_train_samples or 1000
        n_synth_val = args.max_val_samples or 300
        train_ds = SyntheticDetectionDataset(
            n_samples=n_synth_train,
            target_size=args.target_size,
            seed=42,
        )
        val_ds = SyntheticDetectionDataset(
            n_samples=n_synth_val,
            target_size=args.target_size,
            seed=99,
        )

    train_ds = _subset(train_ds, args.max_train_samples)
    val_ds = _subset(val_ds, args.max_val_samples)

    # ── CLI → config ─────────────────────────────────────────────────
    overrides: dict[str, object] = {
        "task": "detection",
        "m_epochs": args.m_epochs,
        "max_iterations": args.decisions,
        "event_log_enabled": args.with_dashboard,
        # Detection labels are 1-based (0 is background), so include
        # explicit background at index 0 for correct dashboard/report mapping.
        "detection_class_names": ["background", *class_names],
        "report_preview_size": 512,
        "report_xai_size": 512,
        "device": device,
    }
    if args.quick:
        overrides["candidate_pruning_enabled"] = True
        overrides["candidate_pruning_relative_threshold"] = 0.7
        overrides["candidate_pruning_warmup_epochs"] = 2
    config = config.model_copy(update=overrides)

    total_epochs = args.m_epochs * (args.decisions + 1)
    print()
    print("=" * 68)
    print(f"  BNNR  ·  {dataset_name}  Detection Showcase")
    print("-" * 68)
    print(f"  Dataset                : {dataset_name}"
          f"{'  (VOC mirrors down — auto-fallback)' if use_synthetic else ''}")
    print(f"  Max main-path epochs   : ~{total_epochs}")
    print(f"  Decision rounds        : {args.decisions}")
    print(f"  Epochs per branch      : {args.m_epochs}")
    n_candidates = 6 if args.quick else "13-14"
    print(f"  Augmentation candidates: {n_candidates} "
          f"({'geometric + ICD/AICD' if args.quick else 'all detection types + variants (14 with albumentations)'})")
    print(f"  Image size             : {args.target_size}×{args.target_size}")
    print(f"  XAI method             : activation (detection)")
    print(f"  Preview/XAI size       : 512×512")
    print(f"  Device                 : {config.device}")
    print(f"  Learning rate          : {args.lr}")
    if args.quick:
        print("  Mode                   : QUICK TEST")
    print("=" * 68)
    print()

    # ── Dashboard ────────────────────────────────────────────────────
    dashboard_url: str | None = None
    if args.with_dashboard:
        dashboard_url = start_dashboard(
            config.report_dir,
            port=args.dashboard_port,
            auto_open=args.dashboard_auto_open,
            build_frontend=args.dashboard_build_frontend,
        )

    n_train = len(cast(Sized, train_ds))
    n_val = len(cast(Sized, val_ds))
    print(f"[data] Train: {n_train:,}   Val: {n_val:,}   "
          f"Classes: {len(class_names)} ({', '.join(class_names[:5])} …)")

    num_workers = _pick_num_workers(2)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=detection_collate_fn_with_index,
        num_workers=num_workers,
        pin_memory=(config.device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=detection_collate_fn_with_index,
        num_workers=num_workers,
        pin_memory=(config.device != "cpu"),
    )

    # ── Model ────────────────────────────────────────────────────────
    model = VOCDetectionModel(num_classes=num_classes)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] Faster R-CNN MobileNet-v3 FPN (~{params:,} params, "
          f"ImageNet-pretrained backbone)")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    adapter = DetectionAdapter(
        model=model,
        optimizer=optimizer,
        device=config.device,
    )

    # ── Augmentations ────────────────────────────────────────────────
    augmentations = _build_all_detection_augmentations(
        seed=config.seed,
        target_size=args.target_size,
        quick=args.quick,
    )
    print(f"\n[augs] Registered {len(augmentations)} augmentation candidates:")
    for i, aug in enumerate(augmentations, 1):
        tag = ""
        if "icd" in aug.name or "aicd" in aug.name:
            tag = " [XAI-driven]"
        elif "mosaic" in aug.name or "mixup" in aug.name:
            tag = " [composite]"
        elif "color" in aug.name or "jitter" in aug.name:
            tag = " [colour]"
        else:
            tag = " [geometric]"
        print(f"       {i:2d}. {aug.name:<26s}  (p={aug.probability:.2f}){tag}")
    print()

    # ── Train ────────────────────────────────────────────────────────
    reporter = Reporter(config.report_dir)
    trainer = BNNRTrainer(
        model=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=augmentations,
        config=config,
        reporter=reporter,
    )
    result = trainer.run()

    # ── Results ──────────────────────────────────────────────────────
    events_path = result.report_json_path.parent / "events.jsonl"
    print()
    print("=" * 68)
    print(f"  {dataset_name} Detection Showcase — Results")
    print("-" * 68)
    print(f"  Best path      : {result.best_path}")
    print(f"  Best metrics   : {result.best_metrics}")
    print(f"  Report JSON    : {result.report_json_path}")
    print(f"  Events (JSONL) : {events_path}")
    if args.with_dashboard:
        if dashboard_url:
            print(f"  Dashboard      : {dashboard_url}")
        else:
            print(
                "  Dashboard      : not started (install optional deps: pip install \"bnnr[dashboard]\"). "
                "Events were still logged."
            )
    print("=" * 68)
    print()

    if args.with_dashboard and dashboard_url:
        print("Dashboard is still running — press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
