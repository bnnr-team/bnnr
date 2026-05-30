"""COCO128 YOLO layout → BNNR detection batch format (for Ultralytics YOLOv8 + BNNR).

Extracted from examples/detection/bnnr_detection_demo.ipynb for reuse by integration scripts.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF  # noqa: N812 — match notebook convention
import yaml
from PIL import Image
from torch.utils.data import Dataset


def yolo_label_has_any_line(label_path: Path) -> bool:
    """True if the YOLO label file has at least one ``cls cx cy w h`` row."""
    if not label_path.is_file():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            return True
    return False


def coco80_names() -> list[str]:
    """MS COCO class names in YOLO/COCO id order 0..79 (from Ultralytics cfg)."""
    import ultralytics

    p = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco.yaml"
    spec = yaml.safe_load(p.read_text(encoding="utf-8"))
    raw = spec["names"]
    if isinstance(raw, dict):
        return [raw[i] for i in range(80)]
    return [str(x) for x in raw]


class Coco128BnnrDataset(Dataset):
    """COCO128 in YOLO txt format → BNNR ``(image, target, index)``. Labels are 0..79."""

    def __init__(
        self,
        image_paths: list[Path],
        coco128_root: Path,
        target_size: int = 640,
    ) -> None:
        self.image_paths = image_paths
        self.root = coco128_root
        self.labels_dir = self.root / "labels" / "train2017"
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        img = TF.resize(img, [self.target_size, self.target_size])
        img_tensor = TF.to_tensor(img)

        label_path = self.labels_dir / (path.stem + ".txt")
        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1o = (cx - bw / 2.0) * w0
                y1o = (cy - bh / 2.0) * h0
                x2o = (cx + bw / 2.0) * w0
                y2o = (cy + bh / 2.0) * h0
                sx = self.target_size / w0
                sy = self.target_size / h0
                x1, y1, x2, y2 = x1o * sx, y1o * sy, x2o * sx, y2o * sy
                x1 = max(0.0, min(float(self.target_size - 1), x1))
                y1 = max(0.0, min(float(self.target_size - 1), y1))
                x2 = max(0.0, min(float(self.target_size), x2))
                y2 = max(0.0, min(float(self.target_size), y2))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(c)

        if not boxes:
            raise IndexError(
                f"Coco128BnnrDataset: no valid xyxy boxes after resize/clip for {path}. "
                f"Fix labels in {label_path}."
            )

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_tensor, target, index


def split_coco128_paths(coco_root: Path) -> tuple[list[Path], list[Path]]:
    """80/20 train/val split over labeled COCO128 images."""
    img_dir = coco_root / "images" / "train2017"
    all_jpg = sorted(img_dir.glob("*.jpg"))
    if len(all_jpg) < 8:
        raise RuntimeError(f"Expected COCO128 images in {img_dir}, found {len(all_jpg)}.")

    labels_dir = coco_root / "labels" / "train2017"
    n_train = max(1, int(len(all_jpg) * 0.8))
    train_paths = [
        p for p in all_jpg[:n_train] if yolo_label_has_any_line(labels_dir / (p.stem + ".txt"))
    ]
    val_paths = [
        p for p in all_jpg[n_train:] if yolo_label_has_any_line(labels_dir / (p.stem + ".txt"))
    ]
    if not train_paths:
        raise RuntimeError("No COCO128 training images with YOLO label rows.")
    if not val_paths:
        raise RuntimeError("No COCO128 validation images with labels.")
    return train_paths, val_paths
