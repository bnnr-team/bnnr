"""Unified dataset loader registry for the grand benchmark.

Every loader returns ``(train_loader, selection_val_loader, held_out_test_loader)``
as a 3-tuple of DataLoaders whose datasets are wrapped with ``_IndexedDataset``.

Supported datasets
------------------
imagewoof, pets, flowers102, dtd, aircraft, eurosat
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCHMARKS_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IMAGEWOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
_IMAGEWOOF_DIRNAME = "imagewoof2-160"


def _balanced_subset(dataset: Any, n_per_class: int | None, seed: int) -> Any:
    """Return a class-balanced Subset with at most *n_per_class* samples each."""
    if n_per_class is None:
        return dataset

    from torch.utils.data import Subset

    targets = _get_targets(dataset)
    by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(idx)

    rng = random.Random(seed)
    chosen: list[int] = []
    for _cls, idxs in sorted(by_class.items()):
        pool = idxs[:]
        rng.shuffle(pool)
        chosen.extend(pool[:n_per_class])
    chosen.sort()
    return Subset(dataset, chosen)


def _split_dataset_50_50(dataset: Any) -> tuple[Any, Any]:
    """Split a dataset 50/50 per class (deterministic, sorted-index).

    Returns ``(selection_val_subset, held_out_test_subset)``.
    """
    from torch.utils.data import Subset

    targets = _get_targets(dataset)
    by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(idx)

    selection_indices: list[int] = []
    held_out_indices: list[int] = []
    for _cls, idxs in sorted(by_class.items()):
        sorted_idxs = sorted(idxs)
        split_point = len(sorted_idxs) // 2
        selection_indices.extend(sorted_idxs[:split_point])
        held_out_indices.extend(sorted_idxs[split_point:])

    selection_indices.sort()
    held_out_indices.sort()
    return Subset(dataset, selection_indices), Subset(dataset, held_out_indices)


def _get_targets(dataset: Any) -> list[int]:
    """Extract class labels from a dataset.  Tries common attribute names first."""
    # Standard torchvision datasets
    for attr in ("targets", "_labels"):
        val = getattr(dataset, attr, None)
        if val is not None:
            return [int(t) for t in val]
    # Wrapped Subset
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        inner_targets = _get_targets(dataset.dataset)
        return [inner_targets[i] for i in dataset.indices]
    # Last resort: iterate (slow for large datasets)
    return [int(dataset[i][1]) for i in range(len(dataset))]  # type: ignore[arg-type]


def _train_transform(img_size: int, policy: str, is_eurosat: bool = False):
    """Build a training transform for the given augmentation policy."""
    from torchvision import transforms

    base = [
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if policy == "randaugment":
        base.append(transforms.RandAugment())
    elif policy == "trivialaugment":
        base.append(transforms.TrivialAugmentWide())
    elif policy == "autoaugment":
        base.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    elif policy != "base":
        raise ValueError(f"Unknown policy {policy!r}")
    base.append(transforms.ToTensor())
    return transforms.Compose(base)


def _val_transform(img_size: int, is_eurosat: bool = False):
    """Build the standard val/test transform."""
    from torchvision import transforms

    if is_eurosat:
        # EuroSAT images are 64 px native; no over-enlargement
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def _make_loaders(
    train_ds: Any,
    selection_val_ds: Any,
    held_out_test_ds: Any,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    """Wrap datasets in _IndexedDataset and return DataLoaders."""
    import torch
    from torch.utils.data import DataLoader

    from bnnr.pipelines import _IndexedDataset

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        _IndexedDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    selection_val_loader = DataLoader(
        _IndexedDataset(selection_val_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    held_out_test_loader = DataLoader(
        _IndexedDataset(held_out_test_ds),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, selection_val_loader, held_out_test_loader


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------


def _imagewoof_loaders(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    from torchvision.datasets import ImageFolder
    from torchvision.datasets.utils import download_and_extract_archive

    root = data_dir / _IMAGEWOOF_DIRNAME
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Imagewoof2 (160px) -> {data_dir} ...", flush=True)
        try:
            download_and_extract_archive(_IMAGEWOOF_URL, download_root=str(data_dir))
        except Exception as exc:
            print(f"Download failed: {exc}")
            raise
        if not (root / "train").is_dir():
            raise FileNotFoundError(
                f"Expected {root}/train after extraction; got {list(data_dir.iterdir())}"
            )

    train_ds = ImageFolder(str(root / "train"), transform=_train_transform(img_size, policy))
    val_ds_full = ImageFolder(str(root / "val"), transform=_val_transform(img_size))

    train_ds = _balanced_subset(train_ds, n_per_class_train, seed)
    selection_val_ds, held_out_test_ds = _split_dataset_50_50(val_ds_full)

    return _make_loaders(
        train_ds, selection_val_ds, held_out_test_ds,
        batch_size=batch_size, seed=seed, num_workers=num_workers,
    )


def _pets_loaders(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    from torchvision.datasets import OxfordIIITPet

    train_tf = _train_transform(img_size, policy)
    val_tf = _val_transform(img_size)

    try:
        trainval_ds = OxfordIIITPet(
            str(data_dir), split="trainval", download=True, transform=train_tf,
        )
        test_ds = OxfordIIITPet(
            str(data_dir), split="test", download=True, transform=val_tf,
        )
    except Exception as exc:
        print(f"OxfordIIITPet download failed: {exc}")
        raise

    # Get targets for trainval: try _labels then fallback
    trainval_targets = _get_targets(trainval_ds)

    # Build a balanced training subset
    # Then the remaining trainval images form selection_val
    if n_per_class_train is not None:
        from torch.utils.data import Subset

        by_class: dict[int, list[int]] = {}
        for idx, target in enumerate(trainval_targets):
            by_class.setdefault(int(target), []).append(idx)

        rng = random.Random(seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        for _cls, idxs in sorted(by_class.items()):
            pool = idxs[:]
            rng.shuffle(pool)
            train_indices.extend(pool[:n_per_class_train])
            val_indices.extend(pool[n_per_class_train:])
        train_indices.sort()
        val_indices.sort()

        train_ds = Subset(trainval_ds, train_indices)
        # selection_val uses val transform
        trainval_val_ds = OxfordIIITPet(
            str(data_dir), split="trainval", download=False, transform=val_tf,
        )
        selection_val_ds = Subset(trainval_val_ds, val_indices)
    else:
        # No subsetting: use trainval 50/50 split for selection_val
        trainval_val_ds = OxfordIIITPet(
            str(data_dir), split="trainval", download=False, transform=val_tf,
        )
        train_ds = trainval_ds
        selection_val_ds, _ = _split_dataset_50_50(trainval_val_ds)

    held_out_test_ds = test_ds

    return _make_loaders(
        train_ds, selection_val_ds, held_out_test_ds,
        batch_size=batch_size, seed=seed, num_workers=num_workers,
    )


def _flowers102_loaders(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    from torchvision.datasets import Flowers102

    train_tf = _train_transform(img_size, policy)
    val_tf = _val_transform(img_size)

    try:
        train_ds = Flowers102(str(data_dir), split="train", download=True, transform=train_tf)
        selection_val_ds = Flowers102(str(data_dir), split="val", download=True, transform=val_tf)
        held_out_test_ds = Flowers102(str(data_dir), split="test", download=True, transform=val_tf)
    except Exception as exc:
        print(f"Flowers102 download failed: {exc}")
        raise

    # Natural splits already have ~10/class in train; subsetting only if requested
    if n_per_class_train is not None:
        train_ds = _balanced_subset(train_ds, n_per_class_train, seed)

    return _make_loaders(
        train_ds, selection_val_ds, held_out_test_ds,
        batch_size=batch_size, seed=seed, num_workers=num_workers,
    )


def _dtd_loaders(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    from torchvision.datasets import DTD

    train_tf = _train_transform(img_size, policy)
    val_tf = _val_transform(img_size)

    try:
        train_ds = DTD(str(data_dir), split="train", partition=1, download=True, transform=train_tf)
        selection_val_ds = DTD(str(data_dir), split="val", partition=1, download=True, transform=val_tf)
        held_out_test_ds = DTD(str(data_dir), split="test", partition=1, download=True, transform=val_tf)
    except Exception as exc:
        print(f"DTD download failed: {exc}")
        raise

    if n_per_class_train is not None:
        train_ds = _balanced_subset(train_ds, n_per_class_train, seed)

    return _make_loaders(
        train_ds, selection_val_ds, held_out_test_ds,
        batch_size=batch_size, seed=seed, num_workers=num_workers,
    )


def _aircraft_loaders(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    from torchvision.datasets import FGVCAircraft

    train_tf = _train_transform(img_size, policy)
    val_tf = _val_transform(img_size)

    try:
        train_ds = FGVCAircraft(
            str(data_dir), split="train", annotation_level="variant",
            download=True, transform=train_tf,
        )
        selection_val_ds = FGVCAircraft(
            str(data_dir), split="val", annotation_level="variant",
            download=True, transform=val_tf,
        )
        held_out_test_ds = FGVCAircraft(
            str(data_dir), split="test", annotation_level="variant",
            download=True, transform=val_tf,
        )
    except Exception as exc:
        print(f"FGVCAircraft download failed: {exc}")
        raise

    if n_per_class_train is not None:
        train_ds = _balanced_subset(train_ds, n_per_class_train, seed)

    return _make_loaders(
        train_ds, selection_val_ds, held_out_test_ds,
        batch_size=batch_size, seed=seed, num_workers=num_workers,
    )


def _eurosat_loaders(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    from torchvision.datasets import EuroSAT

    train_tf = _train_transform(img_size, policy, is_eurosat=True)
    val_tf = _val_transform(img_size, is_eurosat=True)

    try:
        full_ds_train = EuroSAT(str(data_dir), download=True, transform=train_tf)
        full_ds_val = EuroSAT(str(data_dir), download=False, transform=val_tf)
    except Exception as exc:
        print(f"EuroSAT download failed: {exc}")
        raise

    from torch.utils.data import Subset

    targets = _get_targets(full_ds_train)
    by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(idx)

    n_train = n_per_class_train if n_per_class_train is not None else 100
    rng = random.Random(seed)

    train_indices: list[int] = []
    remainder_indices: list[int] = []
    for _cls, idxs in sorted(by_class.items()):
        pool = sorted(idxs)  # deterministic within class
        rng_pool = pool[:]
        rng.shuffle(rng_pool)
        train_indices.extend(rng_pool[:n_train])
        remainder_indices.extend(sorted(rng_pool[n_train:]))

    train_indices.sort()
    remainder_indices.sort()

    # Split remainder 50/50 per class (deterministic)
    rem_by_class: dict[int, list[int]] = {}
    rem_targets = [targets[i] for i in remainder_indices]
    for idx, target in zip(remainder_indices, rem_targets):
        rem_by_class.setdefault(int(target), []).append(idx)

    selection_indices: list[int] = []
    held_out_indices: list[int] = []
    for _cls, idxs in sorted(rem_by_class.items()):
        sorted_idxs = sorted(idxs)
        split_point = len(sorted_idxs) // 2
        selection_indices.extend(sorted_idxs[:split_point])
        held_out_indices.extend(sorted_idxs[split_point:])

    selection_indices.sort()
    held_out_indices.sort()

    train_ds = Subset(full_ds_train, train_indices)
    selection_val_ds = Subset(full_ds_val, selection_indices)
    held_out_test_ds = Subset(full_ds_val, held_out_indices)

    return _make_loaders(
        train_ds, selection_val_ds, held_out_test_ds,
        batch_size=batch_size, seed=seed, num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADER_MAP = {
    "imagewoof": _imagewoof_loaders,
    "pets": _pets_loaders,
    "flowers102": _flowers102_loaders,
    "dtd": _dtd_loaders,
    "aircraft": _aircraft_loaders,
    "eurosat": _eurosat_loaders,
}

SUPPORTED_DATASETS = tuple(_LOADER_MAP.keys())


def get_loaders(
    dataset: str,
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    n_per_class_train: int | None,
    data_dir: Path,
    num_workers: int = 2,
) -> tuple[Any, Any, Any]:
    """Return ``(train_loader, selection_val_loader, held_out_test_loader)``.

    Parameters
    ----------
    dataset:
        One of ``imagewoof | pets | flowers102 | dtd | aircraft | eurosat``.
    img_size:
        Square resize/crop target (dataset-specific default in
        ``run_grand_benchmark.DATASET_DEFAULTS``).
    batch_size:
        DataLoader batch size.
    seed:
        Random seed for balanced subsetting and DataLoader generator.
    policy:
        Train augmentation policy: ``base | randaugment | trivialaugment | autoaugment``.
    n_per_class_train:
        Balanced samples per class in the training split.  ``None`` = all.
    data_dir:
        Root directory where datasets are downloaded/cached.
    num_workers:
        DataLoader worker count.  Use 0 in smoke/debug mode.
    """
    if dataset not in _LOADER_MAP:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Supported: {', '.join(SUPPORTED_DATASETS)}"
        )
    loader_fn = _LOADER_MAP[dataset]
    is_eurosat = dataset == "eurosat"
    return loader_fn(
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        policy=policy,
        n_per_class_train=n_per_class_train,
        data_dir=data_dir,
        num_workers=num_workers,
    )
