"""Built-in dataset + model pipelines for the CLI.

Provides ready-to-use combinations so users can run BNNR directly from
the command line without writing Python code.

Supported datasets: mnist, fashion_mnist, cifar10, imagefolder, coco_mini, yolo.
"""

from __future__ import annotations

import logging
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from bnnr.adapter import ModelAdapter, SimpleTorchAdapter
from bnnr.augmentations import BaseAugmentation
from bnnr.core import BNNRConfig
from bnnr.presets import auto_select_augmentations, get_preset

logger = logging.getLogger(__name__)

_SUPPORTED_DATASETS = ("mnist", "fashion_mnist", "cifar10", "imagefolder", "coco_mini", "yolo")


def list_datasets() -> list[str]:
    """Return list of supported dataset names."""
    return list(_SUPPORTED_DATASETS)


# ---------------------------------------------------------------------------
# Indexed wrapper (returns (image, label, index) triples)
# ---------------------------------------------------------------------------


class _IndexedDataset(Dataset):
    """Wraps a dataset to return (image, label, index) triples."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        image, label = self.dataset[index]
        return image, label, index


# ---------------------------------------------------------------------------
# Simple CNN models for built-in datasets
# ---------------------------------------------------------------------------


class _MnistCNN(nn.Module):
    """Minimal CNN for MNIST / Fashion-MNIST (1-channel 28×28)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        return cast(Tensor, self.fc2(x))


class _CifarCNN(nn.Module):
    """Minimal CNN for CIFAR-10 (3-channel 32×32)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.reshape(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        return cast(Tensor, self.fc2(x))


class _ImageFolderCNN(nn.Module):
    """Flexible CNN for ImageFolder datasets (3-channel, resized to 64×64)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.reshape(-1, 128 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        return cast(Tensor, self.fc2(x))


# ---------------------------------------------------------------------------
# Pipeline builders per dataset
# ---------------------------------------------------------------------------


def _resolve_augmentations(preset: str, seed: int) -> list[BaseAugmentation]:
    """Resolve augmentation preset to a list of augmentations."""
    preset_name = preset.lower().strip()

    if preset_name == "none":
        return []

    if preset_name == "auto":
        augs = auto_select_augmentations(random_state=seed)
    else:
        try:
            augs = get_preset(preset_name, random_state=seed)
        except ValueError:
            logger.warning("Unknown preset '%s', falling back to 'auto'", preset_name)
            augs = auto_select_augmentations(random_state=seed)

    return augs


def _maybe_subset(
    dataset: Dataset,
    max_samples: int | None,
) -> Dataset:
    """Optionally truncate a dataset to ``max_samples``."""
    if max_samples is not None:
        n = len(cast(Sized, dataset))
        indices = list(range(min(max_samples, n)))
        return Subset(dataset, indices)
    return dataset


def build_mnist_pipeline(
    config: BNNRConfig,
    data_dir: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    augmentation_preset: str = "auto",
) -> tuple[SimpleTorchAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build a ready-to-train pipeline for MNIST.

    Note: We use only ``ToTensor()`` (no ``Normalize()``) because BNNR
    augmentations convert to uint8 internally.  The demo CNN uses
    BatchNorm instead.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    train_ds = _maybe_subset(train_ds, max_train_samples)
    val_ds = _maybe_subset(val_ds, max_val_samples)

    train_loader = DataLoader(_IndexedDataset(train_ds), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_IndexedDataset(val_ds), batch_size=batch_size, shuffle=False)

    model = _MnistCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=[model.conv2],
        device=config.device,
    )
    augmentations = _resolve_augmentations(augmentation_preset, config.seed)
    return adapter, train_loader, val_loader, augmentations


def build_fashion_mnist_pipeline(
    config: BNNRConfig,
    data_dir: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    augmentation_preset: str = "auto",
) -> tuple[SimpleTorchAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build a ready-to-train pipeline for Fashion-MNIST.

    Note: We use only ``ToTensor()`` (no ``Normalize()``) because BNNR
    augmentations convert to uint8 internally.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=transform)
    val_ds = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=transform)

    train_ds = _maybe_subset(train_ds, max_train_samples)
    val_ds = _maybe_subset(val_ds, max_val_samples)

    train_loader = DataLoader(_IndexedDataset(train_ds), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_IndexedDataset(val_ds), batch_size=batch_size, shuffle=False)

    model = _MnistCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=[model.conv2],
        device=config.device,
    )
    augmentations = _resolve_augmentations(augmentation_preset, config.seed)
    return adapter, train_loader, val_loader, augmentations


def build_cifar10_pipeline(
    config: BNNRConfig,
    data_dir: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    augmentation_preset: str = "auto",
) -> tuple[SimpleTorchAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build a ready-to-train pipeline for CIFAR-10.

    Note: We use only ``ToTensor()`` (no ``Normalize()``) because BNNR
    augmentations convert to uint8 internally.  The demo CNN uses
    BatchNorm instead.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transform_train)
    val_ds = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=transform_val)

    train_ds = _maybe_subset(train_ds, max_train_samples)
    val_ds = _maybe_subset(val_ds, max_val_samples)

    train_loader = DataLoader(_IndexedDataset(train_ds), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_IndexedDataset(val_ds), batch_size=batch_size, shuffle=False)

    model = _CifarCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.m_epochs * config.max_iterations)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=[model.conv3],
        device=config.device,
        scheduler=scheduler,
    )
    augmentations = _resolve_augmentations(augmentation_preset, config.seed)
    return adapter, train_loader, val_loader, augmentations


def build_imagefolder_pipeline(
    config: BNNRConfig,
    data_path: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    augmentation_preset: str = "auto",
    num_classes: int | None = None,
) -> tuple[SimpleTorchAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build a ready-to-train pipeline for ImageFolder datasets.

    Note: We use only ``ToTensor()`` (no ``Normalize()``) because BNNR
    augmentations convert to uint8 internally.  The demo CNN uses
    BatchNorm instead.

    Expects a directory structure like::

        data_path/
        ├── train/
        │   ├── class_a/
        │   ├── class_b/
        │   └── ...
        └── val/  (or test/ or validation/ or valid/)
            ├── class_a/
            ├── class_b/
            └── ...

    Parameters
    ----------
    data_path:
        Root directory containing ``train/`` and a validation subdirectory
        (one of ``val/``, ``test/``, ``validation/``, ``valid/``).
    num_classes:
        Number of classes. Auto-detected if not provided.
    """
    train_dir = data_path / "train"
    # Accept common validation directory names: val, test, validation
    val_dir = None
    for val_name in ("val", "test", "validation", "valid"):
        candidate = data_path / val_name
        if candidate.is_dir():
            val_dir = candidate
            break

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train directory at {train_dir}. "
            f"ImageFolder requires data_path/train/ and one of data_path/{{val,test,validation}}/ subdirectories."
        )
    if val_dir is None:
        raise FileNotFoundError(
            f"No validation directory found in {data_path}. "
            f"Expected one of: val/, test/, validation/, valid/. "
            f"Found: {sorted(p.name for p in data_path.iterdir() if p.is_dir())}"
        )
    logger.info("Using train_dir=%s, val_dir=%s", train_dir, val_dir)

    image_size = 64
    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform_train)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform_val)

    if num_classes is None:
        num_classes = len(train_ds.classes)
        logger.info("Auto-detected %d classes: %s", num_classes, train_ds.classes)

    train_ds = _maybe_subset(train_ds, max_train_samples)
    val_ds = _maybe_subset(val_ds, max_val_samples)

    train_loader = DataLoader(_IndexedDataset(train_ds), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_IndexedDataset(val_ds), batch_size=batch_size, shuffle=False)

    model = _ImageFolderCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=[model.conv3],
        device=config.device,
    )
    augmentations = _resolve_augmentations(augmentation_preset, config.seed)
    return adapter, train_loader, val_loader, augmentations


# ---------------------------------------------------------------------------
# Detection dataset wrapper
# ---------------------------------------------------------------------------


class _IndexedCocoDetection(Dataset):
    """Wraps a COCO-format detection dataset to return (image, target, index).

    ``target`` is a dict with keys ``boxes`` (FloatTensor[N,4] xyxy) and
    ``labels`` (Int64Tensor[N]).  Images are converted to tensors via
    ``ToTensor()``.
    """

    def __init__(self, root: str, ann_file: str, image_size: int = 480) -> None:
        from torchvision.datasets import CocoDetection

        self.ds = CocoDetection(root=root, annFile=ann_file)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor], int]:
        img, anns = self.ds[index]

        # Original image size for box rescaling
        orig_w, orig_h = img.size  # PIL (W, H)

        img_tensor = self.transform(img)

        # Build xyxy boxes + labels from COCO annotations
        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, w, h]
            # Rescale to resized image coordinates
            sx = self.image_size / orig_w
            sy = self.image_size / orig_h
            boxes.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])
            labels.append(ann["category_id"])

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}
        return img_tensor, target, index


def _detection_collate_fn(
    batch: list[tuple[Tensor, dict[str, Tensor], int]],
) -> tuple[Tensor, list[dict[str, Tensor]], list[int]]:
    """Collate function for detection: stack images, keep targets as list."""
    images = torch.stack([item[0] for item in batch], 0)
    targets = [item[1] for item in batch]
    indices = [item[2] for item in batch]
    return images, targets, indices


def build_coco_mini_pipeline(
    config: BNNRConfig,
    data_path: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    augmentation_preset: str = "none",
    num_classes: int | None = None,
    image_size: int = 480,
) -> tuple[ModelAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build a pipeline for COCO-format object detection.

    Expects directory structure::

        data_path/
        ├── train2017/       (or train/)
        │   ├── 000000.jpg
        │   └── ...
        ├── val2017/         (or val/)
        │   └── ...
        └── annotations/
            ├── instances_train2017.json   (or train.json)
            └── instances_val2017.json     (or val.json)

    The function will attempt multiple naming conventions for the annotation
    and image directories.
    """
    from bnnr.detection_adapter import DetectionAdapter

    # --- Locate annotation files and image dirs --------------------------
    ann_dir = data_path / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Expected annotations/ directory at {ann_dir}")

    # Train image dir
    train_img_dir: Path | None = None
    for name in ("train2017", "train"):
        candidate = data_path / name
        if candidate.is_dir():
            train_img_dir = candidate
            break
    if train_img_dir is None:
        raise FileNotFoundError(
            f"No train image directory found in {data_path}. "
            f"Expected train2017/ or train/."
        )

    # Val image dir
    val_img_dir: Path | None = None
    for name in ("val2017", "val", "validation"):
        candidate = data_path / name
        if candidate.is_dir():
            val_img_dir = candidate
            break
    if val_img_dir is None:
        raise FileNotFoundError(
            f"No val image directory found in {data_path}. "
            f"Expected val2017/ or val/."
        )

    # Annotation files
    train_ann: Path | None = None
    for name in ("instances_train2017.json", "train.json"):
        candidate = ann_dir / name
        if candidate.is_file():
            train_ann = candidate
            break
    if train_ann is None:
        raise FileNotFoundError(
            f"No train annotation file found in {ann_dir}. "
            f"Expected instances_train2017.json or train.json."
        )

    val_ann: Path | None = None
    for name in ("instances_val2017.json", "val.json"):
        candidate = ann_dir / name
        if candidate.is_file():
            val_ann = candidate
            break
    if val_ann is None:
        raise FileNotFoundError(
            f"No val annotation file found in {ann_dir}. "
            f"Expected instances_val2017.json or val.json."
        )

    logger.info(
        "COCO detection: train_imgs=%s, val_imgs=%s, train_ann=%s, val_ann=%s",
        train_img_dir,
        val_img_dir,
        train_ann,
        val_ann,
    )

    # --- Build datasets --------------------------------------------------
    train_ds: Dataset = _IndexedCocoDetection(
        root=str(train_img_dir), ann_file=str(train_ann), image_size=image_size,
    )
    val_ds: Dataset = _IndexedCocoDetection(
        root=str(val_img_dir), ann_file=str(val_ann), image_size=image_size,
    )

    train_ds = _maybe_subset(train_ds, max_train_samples)
    val_ds = _maybe_subset(val_ds, max_val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_detection_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_detection_collate_fn,
        num_workers=0,
    )

    # --- Build model + adapter -------------------------------------------
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    if num_classes is None:
        # Default to 91 (COCO categories) – user should override for custom sets
        num_classes = 91
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    adapter = DetectionAdapter(
        model=model,
        optimizer=optimizer,
        device=config.device,
        scheduler=scheduler,
        use_amp=getattr(config, "use_amp", False),
    )

    augmentations = _resolve_augmentations(augmentation_preset, config.seed)
    return adapter, train_loader, val_loader, augmentations


class _IndexedYoloDetection(Dataset):
    """YOLO-format detection dataset wrapper.

    Expected structure:
    - images in an image directory (from data.yaml train/val)
    - labels in sibling ``labels`` directory, same stem and ``.txt`` extension
      with rows: ``class cx cy w h`` normalized to [0,1].

    Parameters
    ----------
    torchvision_label_offset:
        If ``True`` (default), add 1 to each class id so label 0 is reserved for
        torchvision detection backgrounds. Set ``False`` for Ultralytics YOLO,
        which expects class ids ``0 .. nc-1``.
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int = 480,
        *,
        torchvision_label_offset: bool = True,
    ) -> None:
        from PIL import Image

        self._image_paths = image_paths
        self._image_size = image_size
        self._torchvision_label_offset = torchvision_label_offset
        self._to_tensor = transforms.ToTensor()
        self._resize = transforms.Resize((image_size, image_size))
        self._image_cls = Image

    def __len__(self) -> int:
        return len(self._image_paths)

    def _label_path(self, image_path: Path) -> Path:
        # .../images/.../name.jpg -> .../labels/.../name.txt
        parts = list(image_path.parts)
        for i, p in enumerate(parts):
            if p == "images":
                parts[i] = "labels"
                break
        return Path(*parts).with_suffix(".txt")

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor], int]:
        image_path = self._image_paths[index]
        label_path = self._label_path(image_path)

        img = self._image_cls.open(image_path).convert("RGB")
        img = self._resize(img)
        img_tensor = self._to_tensor(img)

        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.is_file():
            lines = label_path.read_text(encoding="utf-8").splitlines()
            for row in lines:
                row = row.strip()
                if not row:
                    continue
                parts = row.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = parts
                cls_raw = int(float(cls))
                cls_id = cls_raw + 1 if self._torchvision_label_offset else cls_raw
                cx_f = float(cx) * self._image_size
                cy_f = float(cy) * self._image_size
                w_f = float(w) * self._image_size
                h_f = float(h) * self._image_size
                x1 = cx_f - w_f / 2.0
                y1 = cy_f - h_f / 2.0
                x2 = cx_f + w_f / 2.0
                y2 = cy_f + h_f / 2.0
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}
        return img_tensor, target, index


def _resolve_yolo_images_from_entry(entry: str | list[str], data_root: Path) -> list[Path]:
    entries = [entry] if isinstance(entry, str) else list(entry)
    paths: list[Path] = []
    for item in entries:
        candidate = Path(item)
        if not candidate.is_absolute():
            raw = str(item).strip()
            candidate = (data_root / raw).resolve()
            if not candidate.exists():
                # Common Roboflow edge-case: data.yaml in dataset root but
                # paths still use ../train/images, ../valid/images.
                trimmed = raw
                while trimmed.startswith("../"):
                    trimmed = trimmed[3:]
                alt_candidate = (data_root / trimmed).resolve()
                if alt_candidate.exists():
                    candidate = alt_candidate
        if candidate.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                paths.extend(sorted(candidate.rglob(ext)))
        elif candidate.is_file():
            paths.extend([Path(line.strip()) for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()])
    return paths


def build_yolo_pipeline(
    config: BNNRConfig,
    data_path: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    augmentation_preset: str = "none",
    num_classes: int | None = None,
    image_size: int = 480,
    *,
    torchvision_label_offset: bool = True,
) -> tuple[ModelAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build detection pipeline from classic YOLO dataset format.

    ``data_path`` points to ``data.yaml`` or a directory containing it.

    ``torchvision_label_offset`` (default ``True``): add +1 to YOLO class ids for
    torchvision detection (background = 0). Use ``False`` for Ultralytics training
    with the same loaders (classes ``0 .. nc-1``).
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    from bnnr.detection_adapter import DetectionAdapter

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("PyYAML is required for YOLO dataset support. Install with: pip install pyyaml") from exc

    yaml_path = data_path if data_path.is_file() else data_path / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Could not find YOLO data.yaml at: {yaml_path}")

    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or "train" not in spec or "val" not in spec:
        raise ValueError("YOLO data.yaml must contain at least 'train' and 'val' entries.")

    base = yaml_path.parent
    train_images = _resolve_yolo_images_from_entry(spec["train"], base)
    val_images = _resolve_yolo_images_from_entry(spec["val"], base)
    if not train_images or not val_images:
        raise FileNotFoundError("YOLO train/val image lists resolved to empty sets.")

    train_ds: Dataset = _IndexedYoloDetection(
        train_images,
        image_size=image_size,
        torchvision_label_offset=torchvision_label_offset,
    )
    val_ds: Dataset = _IndexedYoloDetection(
        val_images,
        image_size=image_size,
        torchvision_label_offset=torchvision_label_offset,
    )
    train_ds = _maybe_subset(train_ds, max_train_samples)
    val_ds = _maybe_subset(val_ds, max_val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_detection_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_detection_collate_fn,
        num_workers=0,
    )

    if num_classes is None:
        if isinstance(spec.get("names"), list):
            num_classes = len(spec["names"]) + 1  # + background
        elif isinstance(spec.get("nc"), int):
            num_classes = int(spec["nc"]) + 1
        else:
            raise ValueError("Could not infer class count from YOLO data.yaml ('names' or 'nc').")

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    adapter = DetectionAdapter(
        model=model,
        optimizer=optimizer,
        device=config.device,
        scheduler=scheduler,
        use_amp=getattr(config, "use_amp", False),
    )
    augmentations = _resolve_augmentations(augmentation_preset, config.seed)
    return adapter, train_loader, val_loader, augmentations


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_pipeline(
    dataset_name: str,
    config: BNNRConfig,
    data_dir: Path = Path("data"),
    batch_size: int = 64,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    augmentation_preset: str = "auto",
    custom_data_path: Path | None = None,
    num_classes: int | None = None,
    **kwargs: Any,
) -> tuple[ModelAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build a complete pipeline for the given dataset.

    Parameters
    ----------
    dataset_name:
        One of: mnist, fashion_mnist, cifar10, imagefolder, coco_mini, yolo.
    config:
        BNNR configuration.
    data_dir:
        Directory for dataset download/storage (torchvision datasets).
    custom_data_path:
        Path to custom dataset (required for imagefolder).
    augmentation_preset:
        Augmentation preset name (auto, gpu, standard, aggressive, light, none).
    num_classes:
        Number of classes (required for imagefolder if auto-detection fails).

    Raises
    ------
    ValueError
        If dataset_name is not recognized.
    FileNotFoundError
        If imagefolder paths don't exist.
    """
    ds = dataset_name.lower().strip()

    if ds == "mnist":
        return build_mnist_pipeline(config, data_dir, batch_size, max_train_samples, max_val_samples, augmentation_preset)
    elif ds == "fashion_mnist":
        return build_fashion_mnist_pipeline(config, data_dir, batch_size, max_train_samples, max_val_samples, augmentation_preset)
    elif ds == "cifar10":
        return build_cifar10_pipeline(config, data_dir, batch_size, max_train_samples, max_val_samples, augmentation_preset)
    elif ds == "imagefolder":
        if custom_data_path is None:
            raise ValueError("--data-path is required for imagefolder dataset. Provide path to directory with train/val subdirs.")
        return build_imagefolder_pipeline(config, custom_data_path, batch_size, max_train_samples, max_val_samples, augmentation_preset, num_classes)
    elif ds == "coco_mini":
        if custom_data_path is None:
            raise ValueError(
                "--data-path is required for coco_mini dataset. "
                "Provide path to directory with train2017/ (or train/), "
                "val2017/ (or val/), and annotations/ subdirectories."
            )
        return build_coco_mini_pipeline(config, custom_data_path, batch_size, max_train_samples, max_val_samples, augmentation_preset, num_classes)
    elif ds == "yolo":
        if custom_data_path is None:
            raise ValueError(
                "--data-path is required for yolo dataset. "
                "Provide path to data.yaml (or its parent directory)."
            )
        return build_yolo_pipeline(
            config,
            custom_data_path,
            batch_size,
            max_train_samples,
            max_val_samples,
            augmentation_preset,
            num_classes,
            image_size=int(kwargs.get("image_size", 480)),
            torchvision_label_offset=bool(kwargs.get("torchvision_label_offset", True)),
        )
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Supported datasets: {', '.join(_SUPPORTED_DATASETS)}"
        )
