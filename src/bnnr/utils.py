"""General utilities for metrics, filesystem paths, and runtime helpers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    zero_one_loss,
)
from torch import Tensor


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for production-friendly structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(name: str = "bnnr", log_file: Path | None = None, json_format: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if json_format:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed.
    deterministic : bool
        When ``True`` (default, backward-compatible) cuDNN is forced into
        deterministic mode (``cudnn.deterministic=True``, ``benchmark=False``).
        When ``False`` cuDNN benchmark mode is enabled, which auto-tunes
        convolution algorithms and can yield 10-30 % faster training at the
        cost of non-bitwise-reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(device: str = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    device = device.lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}")


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: str = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.as_tensor(array, dtype=dtype, device=get_device(device))


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0

    image = image.astype(np.float32)
    min_v, max_v = image.min(), image.max()
    if min_v < 0.0 or max_v > 1.0:
        denom = (max_v - min_v) + 1e-8
        image = (image - min_v) / denom
    return image


def denormalize_image(
    image: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    image = image.astype(np.float32)
    out = image.copy()
    out = out * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def portable_path(p: Path | str) -> str:
    """Convert a path to a forward-slash string for cross-platform compatibility.

    On Linux/macOS this is a no-op.  On Windows it replaces backslashes so
    that artifact references stored in JSON / event logs always use ``/``
    and work correctly in URLs served by the dashboard.
    """
    return str(p).replace("\\", "/")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    from datetime import datetime

    return datetime.now().strftime(fmt)


def _parse_fbeta(metric_name: str) -> float | None:
    """Parse beta value from metric name like ``fbeta_0.5``, ``fbeta_2``.

    Returns the beta as a float, or ``None`` if the name doesn't match.
    """
    if not metric_name.startswith("fbeta_"):
        return None
    try:
        return float(metric_name[len("fbeta_"):])
    except ValueError:
        return None


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, metrics: list[str] | None = None) -> dict[str, float]:
    if metrics is None:
        metrics = ["accuracy", "f1_macro"]

    # Multi-label path: predictions and labels are 2D (B, num_classes)
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        y_pred = predictions
        y_true = labels

        result: dict[str, float] = {}
        for metric in metrics:
            if metric == "accuracy":
                # Subset (exact-match) accuracy — works natively with 2D arrays
                result[metric] = float(accuracy_score(y_true, y_pred))
            elif metric == "f1_samples":
                result[metric] = float(f1_score(y_true, y_pred, average="samples", zero_division=0))
            elif metric == "f1_macro":
                result[metric] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            elif metric == "f1_micro":
                result[metric] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            elif metric == "f1_weighted":
                result[metric] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            elif metric == "precision":
                result[metric] = float(precision_score(y_true, y_pred, average="samples", zero_division=0))
            elif metric == "precision_macro":
                result[metric] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            elif metric == "precision_micro":
                result[metric] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
            elif metric == "precision_weighted":
                result[metric] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            elif metric == "recall":
                result[metric] = float(recall_score(y_true, y_pred, average="samples", zero_division=0))
            elif metric == "recall_macro":
                result[metric] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            elif metric == "recall_micro":
                result[metric] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
            elif metric == "recall_weighted":
                result[metric] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            elif metric == "hamming":
                result[metric] = float(1.0 - hamming_loss(y_true, y_pred))
            elif metric == "jaccard_samples":
                result[metric] = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
            elif metric == "jaccard_macro":
                result[metric] = float(jaccard_score(y_true, y_pred, average="macro", zero_division=0))
            elif metric == "jaccard_micro":
                result[metric] = float(jaccard_score(y_true, y_pred, average="micro", zero_division=0))
            elif metric == "jaccard_weighted":
                result[metric] = float(jaccard_score(y_true, y_pred, average="weighted", zero_division=0))
            elif metric == "zero_one_loss":
                result[metric] = float(zero_one_loss(y_true, y_pred))
            elif (beta := _parse_fbeta(metric)) is not None:
                result[metric] = float(fbeta_score(y_true, y_pred, beta=beta, average="samples", zero_division=0))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return result

    # Single-label path (unchanged)
    y_pred = predictions.reshape(-1)
    y_true = labels.reshape(-1)

    result = {}
    for metric in metrics:
        if metric == "accuracy":
            result[metric] = float(accuracy_score(y_true, y_pred))
        elif metric == "f1_macro":
            result[metric] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "f1_micro":
            result[metric] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        elif metric == "f1_weighted":
            result[metric] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "precision":
            result[metric] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "precision_macro":
            result[metric] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "precision_micro":
            result[metric] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        elif metric == "precision_weighted":
            result[metric] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "recall":
            result[metric] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "recall_macro":
            result[metric] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "recall_micro":
            result[metric] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        elif metric == "recall_weighted":
            result[metric] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "cohen_kappa":
            result[metric] = float(cohen_kappa_score(y_true, y_pred))
        elif metric == "mcc":
            result[metric] = float(matthews_corrcoef(y_true, y_pred))
        elif metric == "balanced_accuracy":
            result[metric] = float(balanced_accuracy_score(y_true, y_pred))
        elif metric == "hamming":
            result[metric] = float(1.0 - hamming_loss(y_true, y_pred))
        elif metric == "jaccard_macro":
            result[metric] = float(jaccard_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "jaccard_micro":
            result[metric] = float(jaccard_score(y_true, y_pred, average="micro", zero_division=0))
        elif metric == "jaccard_weighted":
            result[metric] = float(jaccard_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "zero_one_loss":
            result[metric] = float(zero_one_loss(y_true, y_pred))
        elif (beta := _parse_fbeta(metric)) is not None:
            result[metric] = float(fbeta_score(y_true, y_pred, beta=beta, average="macro", zero_division=0))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return result
