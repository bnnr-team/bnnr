"""Built-in dataset + model pipelines for the CLI.

Provides ready-to-use combinations so users can run BNNR directly from
the command line without writing Python code.

Supported datasets: mnist, fashion_mnist, cifar10, stl10, imagefolder.
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

_SUPPORTED_DATASETS = ("mnist", "fashion_mnist", "cifar10", "stl10", "imagefolder")


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


class _STL10Net(nn.Module):
    """VGG-style CNN for STL-10 full runs (3-channel 96x96, ~2.5M params).

    Architecture must exactly match ``examples/classification/showcase_stl10.py::STL10Net``
    so that checkpoints trained with the example are loadable from the CLI.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.10),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(self.features(x)))

    @property
    def target_layer(self) -> nn.Module:
        return self.features[24]


class _STL10NetQuick(nn.Module):
    """Lightweight CNN for STL-10 quick demos (48x48 resized, ~150K params).

    Architecture must exactly match ``examples/classification/showcase_stl10.py::STL10NetQuick``
    so that checkpoints trained with the example are loadable from the CLI.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(self.features(x)))

    @property
    def target_layer(self) -> nn.Module:
        return self.features[8]


def _try_stl10_models(
    state_dict: dict[str, Any],
    num_classes: int = 10,
) -> tuple[nn.Module, list[nn.Module]] | None:
    """Try known STL-10 architectures against *state_dict*.

    Returns ``(model, target_layers)`` on the first architecture that loads
    with ``strict=True``, or ``None`` if nothing matches.
    """
    for model_cls in (_STL10NetQuick, _STL10Net):
        model = model_cls(num_classes=num_classes)
        try:
            model.load_state_dict(state_dict, strict=True)
            return model, [model.target_layer]
        except RuntimeError:
            continue
    return None


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


def build_stl10_pipeline(
    config: BNNRConfig,
    data_dir: Path,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    augmentation_preset: str = "auto",
) -> tuple[SimpleTorchAdapter, DataLoader, DataLoader, list[BaseAugmentation]]:
    """Build a ready-to-train pipeline for STL-10.

    STL-10 is downloaded via ``torchvision.datasets.STL10`` (binary format).
    Images are 96x96 RGB.  No ``Normalize()`` — BatchNorm handles it.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.STL10(str(data_dir), split="train", download=True, transform=transform)
    val_ds = datasets.STL10(str(data_dir), split="test", download=True, transform=transform)

    train_ds = _maybe_subset(train_ds, max_train_samples)
    val_ds = _maybe_subset(val_ds, max_val_samples)

    train_loader = DataLoader(_IndexedDataset(train_ds), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_IndexedDataset(val_ds), batch_size=batch_size, shuffle=False)

    model = _STL10NetQuick(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.m_epochs * config.max_iterations,
    )
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=[model.target_layer],
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
        One of: mnist, fashion_mnist, cifar10, stl10, imagefolder.
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
    elif ds == "stl10":
        return build_stl10_pipeline(config, data_dir, batch_size, max_train_samples, max_val_samples, augmentation_preset)
    elif ds == "imagefolder":
        if custom_data_path is None:
            raise ValueError("--data-path is required for imagefolder dataset. Provide path to directory with train/val subdirs.")
        return build_imagefolder_pipeline(config, custom_data_path, batch_size, max_train_samples, max_val_samples, augmentation_preset, num_classes)
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Supported datasets: {', '.join(_SUPPORTED_DATASETS)}"
        )
