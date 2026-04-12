"""Core BNNR training loop, configuration model, and iterative selection engine."""

from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import cv2
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from bnnr.adapter import (  # noqa: F401 — re-exported for backward compat
    ModelAdapter,
    SimpleTorchAdapter,
    XAICapableModel,
)
from bnnr.augmentation_runner import AugmentationRunner
from bnnr.augmentations import BaseAugmentation
from bnnr.data_quality import run_data_quality_analysis
from bnnr.icd import AICD, ICD
from bnnr.utils import set_seed, setup_logging
from bnnr.xai import generate_saliency_maps, save_xai_visualization
from bnnr.xai_analysis import (
    analyze_xai_batch_rich,
)
from bnnr.xai_cache import XAICache

if TYPE_CHECKING:
    from bnnr.reporting import BNNRRunResult, Reporter


class BNNRConfig(BaseModel):
    """Immutable runtime configuration for a full BNNR training run.

    Defines training budget, metrics/selection policy, reporting paths,
    XAI behavior, and task-specific options for classification, detection,
    and multilabel workflows.
    """
    model_config = ConfigDict(frozen=True)

    m_epochs: int = 5
    max_iterations: int = 10
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "f1_macro", "loss"])
    selection_metric: str = "accuracy"
    selection_mode: str = "max"

    # NOTE: For detection tasks, use selection_metric="map_50" (or "map_50_95")
    # and metrics=["map_50", "map_50_95", "loss"].  The model_validator below
    # auto-adjusts these when they are left at classification defaults.
    checkpoint_dir: Path = Path("checkpoints")
    report_dir: Path = Path("reports")
    xai_enabled: bool = True
    xai_samples: int = 4
    xai_method: str = "opticam"
    device: str = "cuda"
    seed: int = 42
    save_checkpoints: bool = True
    verbose: bool = True
    early_stopping_patience: int = 2
    xai_cache_dir: Optional[Path] = None  # noqa: UP045 – pydantic evaluates at runtime; Path | None breaks on 3.9
    xai_cache_samples: int = 0  # 0 = cache entire dataset
    xai_cache_max_samples: int = 50000
    log_file: Optional[Path] = None  # noqa: UP045
    report_preview_size: int = 224
    report_xai_size: int = 512
    dual_xai_report: bool = False
    report_probe_images_per_class: int = 3
    report_probe_max_classes: int = 10
    candidate_pruning_enabled: bool = True
    candidate_pruning_relative_threshold: float = 0.9
    candidate_pruning_warmup_epochs: int = 1
    xai_selection_weight: float = 0.0
    xai_pruning_threshold: float = 0.0
    adaptive_icd_threshold: bool = False
    xai_cache_force_recompute: bool = False
    duplicate_hamming_threshold: int = 10
    xai_cache_progress: bool = True
    event_log_enabled: bool = True
    event_sample_every_epochs: int = 1
    event_xai_every_epochs: int = 1
    event_min_interval_seconds: float = 0.0
    denormalization_mean: Optional[list[float]] = None  # noqa: UP045
    denormalization_std: Optional[list[float]] = None  # noqa: UP045

    # ── Optional baseline re-evaluation per iteration ──
    reeval_baseline_per_iteration: bool = False

    # ── Multi-label-specific fields (ignored when task!="multilabel") ──
    multilabel_threshold: float = 0.5

    # ── Detection-specific fields (ignored when task="classification") ──
    task: str = "classification"
    detection_bbox_format: str = "xyxy"
    detection_targets_mode: str = "auto"  # auto | image_only | bbox_aware
    detection_score_threshold: float = 0.5
    detection_nms_threshold: float = 0.5
    detection_min_box_area: float = 16.0
    detection_max_truncation: float = 0.7
    detection_xai_method: str = "activation"  # activation | occlusion
    detection_xai_grid_size: int = 3
    detection_xai_max_gt_boxes: int = 1
    detection_xai_max_pred_boxes: int = 1
    detection_class_names: Optional[list[str]] = None  # noqa: UP045

    @field_validator("multilabel_threshold")
    @classmethod
    def validate_multilabel_threshold(cls, value: float) -> float:
        if value <= 0.0 or value >= 1.0:
            raise ValueError("multilabel_threshold must be in (0, 1)")
        return value

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        if value not in {"classification", "detection", "multilabel"}:
            raise ValueError("task must be 'classification', 'detection' or 'multilabel'")
        return value

    @field_validator("detection_bbox_format")
    @classmethod
    def validate_detection_bbox_format(cls, value: str) -> str:
        if value not in {"xyxy", "xywh", "cxcywh"}:
            raise ValueError("detection_bbox_format must be 'xyxy', 'xywh' or 'cxcywh'")
        return value

    @field_validator("detection_targets_mode")
    @classmethod
    def validate_detection_targets_mode(cls, value: str) -> str:
        if value not in {"auto", "image_only", "bbox_aware"}:
            raise ValueError("detection_targets_mode must be 'auto', 'image_only' or 'bbox_aware'")
        return value

    @field_validator("detection_score_threshold", "detection_nms_threshold", "detection_max_truncation")
    @classmethod
    def validate_detection_thresholds(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("detection threshold fields must be in [0, 1]")
        return value

    @field_validator("detection_min_box_area")
    @classmethod
    def validate_detection_min_box_area(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("detection_min_box_area must be >= 0")
        return value

    @field_validator("detection_xai_grid_size", "detection_xai_max_gt_boxes", "detection_xai_max_pred_boxes")
    @classmethod
    def validate_detection_xai_controls(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("detection_xai_* controls must be > 0")
        return value

    @field_validator("selection_mode")
    @classmethod
    def validate_selection_mode(cls, value: str) -> str:
        if value not in {"max", "min"}:
            raise ValueError("selection_mode must be 'max' or 'min'")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value not in {"cuda", "cpu", "auto"}:
            raise ValueError("device must be 'cuda', 'cpu' or 'auto'")
        return value

    @field_validator("report_preview_size", "report_xai_size")
    @classmethod
    def validate_report_image_sizes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("report image size fields must be > 0")
        return value

    @field_validator("report_probe_images_per_class", "report_probe_max_classes")
    @classmethod
    def validate_report_probe_controls(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("report probe controls must be > 0")
        return value

    @field_validator("candidate_pruning_relative_threshold")
    @classmethod
    def validate_candidate_pruning_threshold(cls, value: float) -> float:
        if value <= 0.0 or value > 1.0:
            raise ValueError("candidate_pruning_relative_threshold must be in (0, 1]")
        return value

    @field_validator("candidate_pruning_warmup_epochs")
    @classmethod
    def validate_candidate_pruning_warmup(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("candidate_pruning_warmup_epochs must be > 0")
        return value

    @field_validator("event_sample_every_epochs", "event_xai_every_epochs")
    @classmethod
    def validate_event_epoch_sampling(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("event_*_every_epochs fields must be > 0")
        return value

    @field_validator("event_min_interval_seconds")
    @classmethod
    def validate_event_min_interval(cls, value: float) -> float:
        if value < 0:
            raise ValueError("event_min_interval_seconds must be >= 0")
        return value

    @field_validator("xai_selection_weight")
    @classmethod
    def validate_xai_selection_weight(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("xai_selection_weight must be in [0, 1]")
        return value

    @field_validator("xai_pruning_threshold")
    @classmethod
    def validate_xai_pruning_threshold(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("xai_pruning_threshold must be in [0, 1]")
        return value

    @model_validator(mode="before")
    @classmethod
    def _auto_detection_defaults(cls, data: Any) -> Any:
        """Auto-adjust selection_metric and metrics for detection tasks.

        When ``task="detection"`` and the user hasn't explicitly overridden
        ``selection_metric`` / ``metrics``, we switch to detection-appropriate
        defaults (``map_50`` and ``["map_50", "map_50_95", "loss"]``).
        """
        if not isinstance(data, dict):
            return data
        task = data.get("task", "classification")
        if task != "detection":
            return data

        cls_default_metric = "accuracy"
        cls_default_metrics = ["accuracy", "f1_macro", "loss"]

        if data.get("selection_metric", cls_default_metric) == cls_default_metric:
            data["selection_metric"] = "map_50"

        if data.get("metrics", cls_default_metrics) == cls_default_metrics:
            data["metrics"] = ["map_50", "map_50_95", "loss"]

        return data

    @model_validator(mode="before")
    @classmethod
    def _auto_multilabel_defaults(cls, data: Any) -> Any:
        """Auto-adjust selection_metric and metrics for multilabel tasks.

        When ``task="multilabel"`` and the user hasn't explicitly overridden
        ``selection_metric`` / ``metrics``, we switch to multilabel-appropriate
        defaults (``f1_samples`` and ``["f1_samples", "f1_macro", "accuracy", "loss"]``).
        """
        if not isinstance(data, dict):
            return data
        task = data.get("task", "classification")
        if task != "multilabel":
            return data

        cls_default_metric = "accuracy"
        cls_default_metrics = ["accuracy", "f1_macro", "loss"]

        if data.get("selection_metric", cls_default_metric) == cls_default_metric:
            data["selection_metric"] = "f1_samples"

        if data.get("metrics", cls_default_metrics) == cls_default_metrics:
            data["metrics"] = ["f1_samples", "f1_macro", "accuracy", "loss"]

        return data

# ModelAdapter, XAICapableModel, SimpleTorchAdapter are now in bnnr.adapter
# Re-exported here for backward compatibility.
# (imported at module top level from bnnr.adapter)


@dataclass
class _TrainerState:
    current_iteration: int
    active_augmentations: list[str]
    baseline_metrics: dict[str, float]


@dataclass
class _RuntimeState:
    """Mutable runtime flags that should NOT live on the frozen BNNRConfig."""

    xai_disabled: bool = False


# ── State-dict helpers (module-level, used by BNNRTrainer) ──────────────


def clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Create a detached clone of a state dict (initial allocation).

    Handles nested dicts (e.g. adapter state with ``model``, ``optimizer``
    sub-dicts) by recursing into them and cloning tensor leaves.
    """
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = clone_state_dict(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def copy_state_dict_inplace(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Copy *src* state dict values into *dst* buffers **in-place**.

    Avoids the overhead of ``copy.deepcopy`` by reusing pre-allocated
    tensors instead of creating new ones each time.  Handles nested
    dicts by recursing, copying tensor leaves in-place.  Non-tensor
    leaves are replaced directly (e.g. step counters).
    """
    for k, v in src.items():
        if isinstance(v, Tensor) and isinstance(dst.get(k), Tensor):
            dst[k].copy_(v)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            copy_state_dict_inplace(dst[k], v)
        else:
            # Non-tensor leaf (int, list, etc.) — replace directly.
            dst[k] = copy.deepcopy(v)


def _is_ultralytics_tasks_backbone(model: Any) -> bool:
    """Ultralytics task modules use a BCHW tensor forward, not a list of CHW tensors like torchvision detection."""
    mod = type(model).__module__
    name = type(model).__name__
    return mod.startswith("ultralytics.nn.tasks") and name in {
        "DetectionModel",
        "OBBModel",
        "SegmentationModel",
        "PoseModel",
        "YOLOEModel",
    }


class BNNRTrainer:
    """Execute iterative BNNR training, evaluation, and augmentation selection.

    Coordinates baseline/candidate epochs, checkpointing, event emission,
    XAI collection, and final report artifacts while preserving the best
    path under the configured selection metric.
    """
    def __init__(
        self,
        model: ModelAdapter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        augmentations: list[BaseAugmentation],
        config: BNNRConfig,
        reporter: Reporter | None = None,
        custom_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.augmentations = augmentations
        self.config = config
        self._runtime = _RuntimeState()

        if reporter is None:
            from bnnr.reporting import Reporter

            reporter = Reporter(config.report_dir)
        self.reporter = reporter

        self.current_iteration = 0
        self.best_augmentation: str | None = None
        self._active_augmentations: list[BaseAugmentation] = []
        self._resume_completed_candidates: set[str] = set()
        self._resume_iteration_results: dict[str, dict[str, float]] = {}
        self._resume_baseline_metrics: dict[str, float] | None = None
        self._report_probe_images: Tensor | None = None
        self._report_probe_labels: Tensor | None = None
        self._report_probe_targets: list[dict[str, Tensor]] | None = None
        self._report_probe_sample_ids: list[str] = []
        self._latest_detection_xai_details: list[dict[str, Any]] = []
        self._fast_probe_originals: dict[tuple[int, str, int], list[Path]] = {}
        self._prev_xai_batch_stats: dict[str, list[dict[str, float]]] = {}
        self._baseline_xai_stats: dict[str, list[dict[str, float]]] = {}
        # Cache for merged _evaluate + _compute_eval_class_details (classification only)
        self._last_eval_preds: np.ndarray | None = None
        self._last_eval_labels: np.ndarray | None = None
        self.logger = setup_logging("bnnr", config.log_file, json_format=True)
        self._custom_metrics: dict[str, Any] = custom_metrics or {}

        # ── Multi-label criterion sanity check ────────────────────────
        if self._is_multilabel and isinstance(model, SimpleTorchAdapter):
            criterion = model.criterion
            criterion_name = type(criterion).__name__
            if criterion_name in {"CrossEntropyLoss", "NLLLoss"}:
                import warnings as _w
                _w.warn(
                    f"task='multilabel' but criterion is {criterion_name}. "
                    f"Multi-label classification typically requires "
                    f"BCEWithLogitsLoss. Current criterion may produce "
                    f"incorrect gradients for multi-label targets.",
                    UserWarning,
                    stacklevel=2,
                )
            if criterion_name == "BCELoss":
                import warnings as _w
                _w.warn(
                    "task='multilabel' with BCELoss detected. BCELoss requires "
                    "sigmoid-activated inputs but the adapter passes raw logits. "
                    "Use nn.BCEWithLogitsLoss instead for numerical stability.",
                    UserWarning,
                    stacklevel=2,
                )

    @property
    def _xai_enabled(self) -> bool:
        """Check if XAI is effectively enabled (config + runtime)."""
        return self.config.xai_enabled and not self._runtime.xai_disabled

    @property
    def _is_detection(self) -> bool:
        """Check if this run is object detection task."""
        return self.config.task == "detection"

    @property
    def _is_multilabel(self) -> bool:
        """Check if this run is multi-label classification task."""
        return self.config.task == "multilabel"

    def _log(self, message: str) -> None:
        if self.config.verbose:
            self.logger.info(message)

    def _emit_pipeline_phase(
        self,
        phase: str,
        status: str,
        message: str = "",
    ) -> None:
        """Emit a ``pipeline_phase`` event so the dashboard can show pre-training activity."""
        emit_fn = getattr(self.reporter, "_event_sink", None)
        if emit_fn is not None and hasattr(emit_fn, "emit"):
            emit_fn.emit(
                "pipeline_phase",
                {"phase": phase, "status": status, "message": message},
                force=True,
            )

    def _emit_pipeline_complete(self) -> None:
        """Emit a ``pipeline_complete`` event so the dashboard knows training finished."""
        emit_fn = getattr(self.reporter, "_event_sink", None)
        if emit_fn is not None and hasattr(emit_fn, "emit"):
            emit_fn.emit("pipeline_complete", {}, force=True)

    def _check_pause(self) -> None:
        """Poll for a pause signal file and block until it is removed.

        The dashboard creates ``<run_dir>/.bnnr_pause`` when the user clicks
        Pause.  This method is called at epoch boundaries so training
        can be suspended without altering the core training logic.
        """
        from bnnr.dashboard.backend import PAUSE_SIGNAL_FILENAME

        pause_file = self.reporter.run_dir / PAUSE_SIGNAL_FILENAME
        if not pause_file.exists():
            return
        print("\n  ⏸  Training paused — waiting for resume signal ...", flush=True)
        self._log("Training paused via dashboard signal")
        while pause_file.exists():
            time.sleep(0.5)
        print("  ▶  Training resumed\n", flush=True)
        self._log("Training resumed")

    def _average_metrics(self, all_metrics: list[dict[str, float]]) -> dict[str, float]:
        """Average metric dicts across batches/epochs.

        Batches may omit keys (e.g. detection skips a bad batch and returns only
        ``loss`` + ``loss_non_finite``). Averaging must use the union of keys and
        only include finite values present for each key.
        """
        if not all_metrics:
            return {}
        keys: set[str] = set()
        for m in all_metrics:
            keys.update(m.keys())
        out: dict[str, float] = {}
        for k in keys:
            vals: list[float] = []
            for m in all_metrics:
                if k not in m:
                    continue
                v = float(m[k])
                if math.isfinite(v):
                    vals.append(v)
            if vals:
                out[k] = float(sum(vals) / len(vals))
        return out

    # Delegate to module-level helpers for backward compat
    _clone_state_dict = staticmethod(clone_state_dict)
    _copy_state_dict_inplace = staticmethod(copy_state_dict_inplace)

    def _tensor_to_uint8(self, images: Tensor) -> np.ndarray:
        """Convert a (B, C, H, W) float tensor to a (B, H, W, C) uint8 array.

        Handles three input ranges:
        * [0, 255] → direct cast
        * [0, 1]   → multiply by 255
        * negative / >1 (normalised) → CLAMP to [0, 1] first **and warn**
        """
        np_images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        lo, hi = float(np_images.min()), float(np_images.max())

        if lo < -0.01 or (hi > 1.05 and hi < 200):
            # Likely normalised (mean/std) data — clamp to [0, 1] as best-effort
            if not getattr(self, "_norm_warning_emitted", False):
                import warnings

                warnings.warn(
                    "BNNR detected input tensors with values outside [0, 1] "
                    f"(range [{lo:.2f}, {hi:.2f}]). This usually means "
                    "transforms.Normalize() was applied BEFORE BNNR augmentations. "
                    "BNNR augmentations convert images to uint8 internally — "
                    "pre-normalised data will be corrupted. Remove Normalize() "
                    "from your DataLoader transforms and rely on BatchNorm in "
                    "the model instead.",
                    RuntimeWarning,
                    stacklevel=4,
                )
                self._norm_warning_emitted = True
            np_images = np.clip(np_images, 0.0, 1.0)

        if hi <= 1.05:
            return (np_images * 255.0).astype("uint8")  # type: ignore[no-any-return]
        return np_images.astype("uint8")  # type: ignore[no-any-return]

    @staticmethod
    def _uint8_to_tensor(np_images: np.ndarray, *, ref_batch: Tensor) -> Tensor:
        """Convert (B, H, W, C) uint8 back to (B, C, H, W) float tensor."""
        t = torch.as_tensor(np_images, dtype=ref_batch.dtype, device=ref_batch.device)
        t = t.permute(0, 3, 1, 2)
        if ref_batch.max() <= 1.05:
            t = t / 255.0
        return t

    @staticmethod
    def _det_uint8_batch_to_float01(np_images: np.ndarray, *, ref_batch: Tensor) -> Tensor:
        """Uint8 HWC batch → BCHW float32 in ``[0, 1]`` (detection / YOLO contract).

        Always divides by 255 — unlike :meth:`_uint8_to_tensor`, this does **not**
        depend on ``ref_batch.max()`` so chained bbox-aware augmentations never
        leave a float ``0–255`` batch that breaks Ultralytics.
        """
        t = torch.as_tensor(np_images, dtype=torch.uint8, device=ref_batch.device)
        t = t.permute(0, 3, 1, 2).to(dtype=torch.float32)
        return (t / 255.0).clamp(0.0, 1.0)

    def _apply_augmentation_to_batch(
        self,
        batch: Any,
        augmentation: BaseAugmentation,
        sample_indices: Tensor | None = None,
    ) -> Any:
        # ── Detection path ──────────────────────────────────────────────
        if self._is_detection:
            images, targets = batch
            targets_mode = self.config.detection_targets_mode
            # Bbox-aware augmentations implement apply_with_targets
            can_apply_with_targets = hasattr(augmentation, "apply_with_targets")
            if targets_mode == "bbox_aware" and not can_apply_with_targets:
                # Explicit bbox-aware mode but augmentation has no bbox path:
                # keep image-only semantics to avoid breaking existing augmentations.
                can_apply_with_targets = False
            if targets_mode == "image_only":
                can_apply_with_targets = False

            if can_apply_with_targets:
                np_images = self._tensor_to_uint8(images)
                aug_images_list = []
                aug_targets_list = []
                ref_h = int(images.shape[2])
                ref_w = int(images.shape[3])
                for idx in range(np_images.shape[0]):
                    aug_img, aug_tgt = augmentation.apply_with_targets(  # type: ignore[attr-defined]  # duck-typed via hasattr check above
                        np_images[idx], targets[idx],
                    )
                    # Some detection augs (e.g. random scale) can change image size.
                    # Keep batch shape stable by resizing back to the reference
                    # tensor size and scaling boxes accordingly.
                    if aug_img.shape[0] != ref_h or aug_img.shape[1] != ref_w:
                        src_h, src_w = int(aug_img.shape[0]), int(aug_img.shape[1])
                        sx = ref_w / max(src_w, 1)
                        sy = ref_h / max(src_h, 1)
                        aug_img = cv2.resize(aug_img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

                        boxes_any = aug_tgt.get("boxes")
                        if isinstance(boxes_any, Tensor):
                            boxes_t = boxes_any.clone().to(dtype=torch.float32)
                            if boxes_t.numel() > 0:
                                boxes_t[:, [0, 2]] *= sx
                                boxes_t[:, [1, 3]] *= sy
                                boxes_t[:, [0, 2]] = boxes_t[:, [0, 2]].clamp_(0, ref_w)
                                boxes_t[:, [1, 3]] = boxes_t[:, [1, 3]].clamp_(0, ref_h)
                            aug_tgt["boxes"] = boxes_t
                        elif boxes_any is not None:
                            boxes_np = np.asarray(boxes_any, dtype=np.float32).copy()
                            if boxes_np.size > 0:
                                boxes_np[:, [0, 2]] *= sx
                                boxes_np[:, [1, 3]] *= sy
                                boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, ref_w)
                                boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, ref_h)
                            aug_tgt["boxes"] = boxes_np

                    aug_images_list.append(aug_img)
                    aug_targets_list.append(aug_tgt)
                aug_images_np = np.stack(aug_images_list, axis=0)
                return self._det_uint8_batch_to_float01(aug_images_np, ref_batch=images), aug_targets_list

            # Pixel-only augmentations: apply to images, pass targets through
            try:
                images = augmentation.apply_tensor(images)
                return images, targets
            except (NotImplementedError, RuntimeError, TypeError):
                np_images = self._tensor_to_uint8(images)
                aug_images = augmentation.apply_batch(np_images)
                return self._det_uint8_batch_to_float01(aug_images, ref_batch=images), targets

        # ── Classification path (unchanged) ─────────────────────────────
        images, labels = batch
        if hasattr(augmentation, "apply_batch_with_labels"):
            np_images = self._tensor_to_uint8(images)
            np_labels = labels.detach().cpu().numpy()
            np_indices = sample_indices.detach().cpu().numpy() if sample_indices is not None else None
            aug_images = augmentation.apply_batch_with_labels(np_images, np_labels, sample_indices=np_indices)
            return self._uint8_to_tensor(aug_images, ref_batch=images), labels

        try:
            images = augmentation.apply_tensor(images)
            return images, labels
        except (NotImplementedError, RuntimeError, TypeError):
            # Fallback to numpy path if tensor augmentation fails
            np_images = self._tensor_to_uint8(images)
            aug_images = augmentation.apply_batch(np_images)
            return self._uint8_to_tensor(aug_images, ref_batch=images), labels

    def _train_epoch(self, loader: DataLoader, augmentations: list[BaseAugmentation] | None = None) -> dict[str, float]:
        epoch_metrics: list[dict[str, float]] = []

        # ── Detection path (unchanged — AugmentationRunner doesn't
        #    support apply_with_targets for bbox-aware augmentations) ──
        if self._is_detection:
            for raw_batch in loader:
                if len(raw_batch) == 3:
                    images, targets, sample_indices = raw_batch
                else:
                    images, targets = raw_batch
                    sample_indices = None
                batch: Any = (images, targets)
                if augmentations:
                    for aug in augmentations:
                        batch = self._apply_augmentation_to_batch(batch, aug, sample_indices=sample_indices)
                metrics = self.model.train_step(batch)
                epoch_metrics.append(metrics)
            return self._average_metrics(epoch_metrics)

        # ── Classification path — use AugmentationRunner for batched
        #    numpy conversions and optional async prefetch ─────────────
        if augmentations:
            runner = AugmentationRunner(augmentations, async_prefetch=False)
            for raw_batch in loader:
                if len(raw_batch) == 3:
                    images, labels, sample_indices = raw_batch
                else:
                    images, labels = raw_batch
                    sample_indices = None
                images, labels = runner.apply_batch(images, labels, sample_indices=sample_indices)
                metrics = self.model.train_step((images, labels))
                epoch_metrics.append(metrics)
        else:
            for raw_batch in loader:
                if len(raw_batch) == 3:
                    images, labels, _ = raw_batch
                    batch = (images, labels)
                else:
                    batch = raw_batch
                metrics = self.model.train_step(batch)
                epoch_metrics.append(metrics)
        return self._average_metrics(epoch_metrics)

    def _evaluate(self, loader: DataLoader, *, cache_predictions: bool = False) -> dict[str, float]:
        """Evaluate the model on *loader*.

        Parameters
        ----------
        cache_predictions : bool
            When ``True`` **and** the task is classification **and** the
            model exposes ``get_model()`` (i.e. is :class:`XAICapableModel`),
            raw per-sample predictions and labels are cached in
            ``self._last_eval_preds`` / ``self._last_eval_labels`` so that
            :meth:`_compute_eval_class_details` can reuse them without a
            second forward pass.
        """
        all_metrics: list[dict[str, float]] = []

        # Determine whether we can cache predictions in this call.
        can_cache = (
            cache_predictions
            and not self._is_detection
            and isinstance(self.model, XAICapableModel)
        )
        preds_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []

        # Use a forward hook to capture logits from eval_step's own forward
        # pass, eliminating the need for a separate (second) forward pass.
        _captured_logits: list[torch.Tensor] = []
        _hook_handle = None
        if can_cache:
            model_impl = self.model.get_model()  # type: ignore[attr-defined]  # XAICapableModel checked via can_cache
            model_impl.eval()

            def _capture_hook(_module: Any, _inp: Any, output: Any) -> None:
                _captured_logits.append(output.detach())

            _hook_handle = model_impl.register_forward_hook(_capture_hook)

        try:
            for raw_batch in loader:
                # ── Detection path ─────────────────────────────────────
                if self._is_detection:
                    if len(raw_batch) == 3:
                        images, targets, _ = raw_batch
                    else:
                        images, targets = raw_batch
                    batch: Any = (images, targets)
                    all_metrics.append(self.model.eval_step(batch))
                    continue

                # ── Classification path ────────────────────────────────
                if len(raw_batch) == 3:
                    images, labels, _ = raw_batch
                    batch = (images, labels)
                else:
                    batch = raw_batch
                    images, labels = raw_batch[0], raw_batch[1]

                _captured_logits.clear()
                all_metrics.append(self.model.eval_step(batch))

                # Extract predictions from the logits captured by the hook
                # during eval_step — no second forward pass needed.
                if can_cache and _captured_logits:
                    logits = _captured_logits[-1]
                    if self._is_multilabel:
                        preds_rows.append(
                            (torch.sigmoid(logits) >= self.config.multilabel_threshold).int().cpu()
                        )
                    else:
                        preds_rows.append(torch.argmax(logits, dim=1).cpu())
                    label_rows.append(labels.cpu())
        finally:
            if _hook_handle is not None:
                _hook_handle.remove()

        # Store cached predictions so _compute_eval_class_details can reuse.
        if can_cache and preds_rows:
            self._last_eval_preds = torch.cat(preds_rows).numpy().astype(np.int64)
            self._last_eval_labels = torch.cat(label_rows).numpy().astype(np.int64)
        else:
            self._last_eval_preds = None
            self._last_eval_labels = None

        # For detection: call epoch_end_eval to compute mAP over all batches
        epoch_end_eval_fn = getattr(self.model, "epoch_end_eval", None)
        if self._is_detection and callable(epoch_end_eval_fn):
            epoch_level_metrics = epoch_end_eval_fn()
            # Merge epoch-level metrics; remove the dummy loss=0.0 from
            # eval_step since detection models don't produce eval loss —
            # the real training loss comes from train_metrics.
            avg = self._average_metrics(all_metrics)
            avg.pop("loss", None)  # remove dummy eval loss
            avg.update(epoch_level_metrics)
            result = avg
        else:
            result = self._average_metrics(all_metrics)

        # ── Custom callable metrics ──────────────────────────────────
        if self._custom_metrics and self._last_eval_preds is not None and self._last_eval_labels is not None:
            for name, fn in self._custom_metrics.items():
                try:
                    result[name] = float(fn(self._last_eval_preds, self._last_eval_labels))
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        "Custom metric %r failed and was omitted from results: %s",
                        name,
                        exc,
                        exc_info=True,
                    )

        return result

    def _save_checkpoint(
        self,
        iteration: int,
        augmentation_name: str,
        metrics: dict[str, float],
        baseline_metrics: dict[str, float] | None = None,
        completed_candidates: list[str] | None = None,
        current_best_metric: float | None = None,
        iteration_results: dict[str, dict[str, float]] | None = None,
    ) -> Path:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", augmentation_name)
        checkpoint_path = self.config.checkpoint_dir / f"iter_{iteration}_{safe_name}.pt"
        payload = {
            "model_state": self.model.state_dict(),
            "iteration": iteration,
            "augmentation_name": augmentation_name,
            "metrics": metrics,
            "active_augmentations": [aug.name for aug in self._active_augmentations],
            "baseline_metrics": baseline_metrics or metrics,
            "completed_candidates": completed_candidates or [],
            "current_best_metric": current_best_metric,
            "iteration_results": iteration_results or {},
            # Full RNG state for deterministic resume
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else []
                ),
            },
            "config_snapshot": self.config.model_dump(mode="json"),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        # weights_only=False is required because checkpoints contain Python/numpy
        # RNG state objects for deterministic resume. Checkpoints are self-generated
        # by BNNR, so pickle deserialization risk is acceptable here.
        #
        # Safety check: verify checkpoint is a dict with expected BNNR keys
        # before applying its contents to the running model.
        state = cast(dict[str, Any], torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        expected_checkpoint_keys = {"model_state", "iteration", "augmentation_name"}
        if not isinstance(state, dict) or not expected_checkpoint_keys.issubset(state.keys()):
            raise ValueError(
                f"Checkpoint at {checkpoint_path} does not appear to be a valid BNNR checkpoint. "
                f"Expected keys {expected_checkpoint_keys}, got {set(state.keys()) if isinstance(state, dict) else type(state).__name__}."
            )
        self.model.load_state_dict(state["model_state"])
        # Restore full RNG state for deterministic resume
        rng_state = state.get("rng_state")
        if rng_state is not None:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch_cpu" in rng_state:
                torch.random.set_rng_state(rng_state["torch_cpu"])
            if "torch_cuda" in rng_state and torch.cuda.is_available():
                cuda_states = rng_state["torch_cuda"]
                if cuda_states:
                    torch.cuda.set_rng_state_all(cuda_states)
        return state

    def _select_best_path(
        self,
        results: dict[str, dict[str, float]],
        baseline_metrics: dict[str, float],
        xai_scores: dict[str, float] | None = None,
    ) -> str | None:
        metric = self.config.selection_metric
        mode = self.config.selection_mode
        w = self.config.xai_selection_weight

        baseline_value = baseline_metrics.get(metric)

        # When XAI weight is zero (default) or no XAI scores, use pure metric selection
        if w <= 0 or not xai_scores:
            best_name: str | None = None
            best_value = None
            for aug_name, aug_metrics in results.items():
                val = aug_metrics.get(metric)
                if val is None:
                    continue
                if best_value is None or (mode == "max" and val > best_value) or (mode == "min" and val < best_value):
                    best_name = aug_name
                    best_value = val
            if best_name is None or baseline_value is None or best_value is None:
                return None
            improved = (best_value > baseline_value) if mode == "max" else (best_value < baseline_value)
            return best_name if improved else None

        # Composite selection: (1-w)*normalized_metric + w*xai_quality
        metric_vals = {name: m.get(metric) for name, m in results.items() if m.get(metric) is not None}
        if not metric_vals:
            return None

        all_vals = list(metric_vals.values())
        min_m = min(v for v in all_vals if v is not None)
        max_m = max(v for v in all_vals if v is not None)
        m_range = max_m - min_m if max_m != min_m else 1.0  # type: ignore[operator]

        best_name = None
        best_composite: float | None = None
        for aug_name, val in metric_vals.items():
            if val is None:
                continue
            # Normalise metric to [0, 1]
            if mode == "max":
                norm_m = (float(val) - float(min_m)) / float(m_range)  # type: ignore[arg-type]
            else:
                norm_m = (float(max_m) - float(val)) / float(m_range)  # type: ignore[arg-type]
            xai_q = xai_scores.get(aug_name, 0.0)
            composite = (1.0 - w) * norm_m + w * xai_q
            if best_composite is None or composite > best_composite:
                best_composite = composite
                best_name = aug_name

        if best_name is None or baseline_value is None:
            return None

        best_value = results[best_name].get(metric)
        if best_value is None:
            return None
        improved = (best_value > baseline_value) if mode == "max" else (best_value < baseline_value)
        return best_name if improved else None

    def _should_prune_candidate(
        self,
        candidate_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        xai_quality: float | None = None,
    ) -> bool:
        if not self.config.candidate_pruning_enabled:
            return False
        metric = self.config.selection_metric
        candidate_value = candidate_metrics.get(metric)
        baseline_value = baseline_metrics.get(metric)
        if candidate_value is None or baseline_value is None:
            return False

        threshold = self.config.candidate_pruning_relative_threshold
        if self.config.selection_mode == "max":
            metric_prune = float(candidate_value) < float(baseline_value) * threshold
        else:
            # For "min" mode, threshold=0.9 means prune when candidate is 10% worse.
            metric_prune = float(candidate_value) > float(baseline_value) * (2.0 - threshold)

        if metric_prune:
            return True

        # XAI-quality based pruning (opt-in)
        xai_thresh = self.config.xai_pruning_threshold
        if xai_thresh > 0 and xai_quality is not None and xai_quality < xai_thresh:
            return True

        return False

    def _get_current_best_metric(self, results: dict[str, dict[str, float]]) -> float | None:
        metric = self.config.selection_metric
        values = [v[metric] for v in results.values() if metric in v]
        if not values:
            return None
        return float(max(values) if self.config.selection_mode == "max" else min(values))

    def _top_k_candidate_names(self, results: dict[str, dict[str, float]], k: int = 3) -> list[str]:
        metric = self.config.selection_metric
        sorted_items = sorted(
            ((name, metrics) for name, metrics in results.items() if metric in metrics),
            key=lambda item: float(item[1][metric]),
            reverse=(self.config.selection_mode == "max"),
        )
        return [name for name, _ in sorted_items[:k]]

    def _generate_xai(
        self,
        iteration: int,
        augmentation_name: str,
        confusion: dict[str, Any] | None = None,
    ) -> tuple[list[Path], dict[str, str], dict[str, dict[str, Any]]]:
        """Generate XAI saliency maps and per-class textual insights.

        Parameters
        ----------
        confusion : dict | None
            Confusion matrix dict (``{"labels": [...], "matrix": [[...]]}``).
            When provided, insights are enriched with confusion-pair info and
            cross-checkpoint trend indicators.

        Returns
        -------
        paths : list[Path]
            Saved overlay PNG paths.
        insights : dict[str, str]
            Mapping ``class_id`` → human-readable insight text.
        diagnoses : dict[str, dict]
            Structured per-class diagnosis (severity, quality_score, …).
            Empty dict when XAI is disabled.
        """
        if not self._xai_enabled:
            return [], {}, {}

        if self._is_detection:
            model_getter = getattr(self.model, "get_model", None)
            if not callable(model_getter):
                return [], {}, {}

            from bnnr.detection_xai import (
                compute_detection_box_saliency_occlusion,
                generate_detection_saliency,
                save_detection_xai_panels,
            )

            self._initialize_report_probe_samples()
            if self._report_probe_images is None or self._report_probe_targets is None:
                return [], {}, {}

            model_impl = model_getter()
            device = next(model_impl.parameters()).device
            predict_ultra = getattr(self.model, "predict_detection_dicts", None)
            use_ultra_xai = callable(predict_ultra)
            if _is_ultralytics_tasks_backbone(model_impl) and not use_ultra_xai:
                self._log(
                    "Detection XAI skipped: Ultralytics backbone requires "
                    "UltralyticsDetectionAdapter.predict_detection_dicts (raw task module only)."
                )
                return [], {}, {}

            # Generate XAI for ALL probe samples (activation method is
            # a single batched forward pass, so the cost is negligible).
            images = self._report_probe_images.to(device)
            targets = self._report_probe_targets

            xai_method = self.config.detection_xai_method
            forward_layout: Literal["torchvision_list", "ultralytics_bchw"] = (
                "ultralytics_bchw" if use_ultra_xai else "torchvision_list"
            )

            with torch.no_grad():
                if use_ultra_xai and predict_ultra is not None:
                    preds = predict_ultra(images)
                else:
                    preds = model_impl([img for img in images])

            # ── Activation-based saliency (fast: single batched forward pass) ──
            activation_saliency: list[np.ndarray] | None = None
            if xai_method == "activation":
                target_layers_fn = getattr(self.model, "get_target_layers", None)
                target_layers = target_layers_fn() if callable(target_layers_fn) else []
                if target_layers:
                    activation_saliency = generate_detection_saliency(
                        model_impl,
                        images,
                        target_layers,
                        device=device,
                        forward_layout=forward_layout,
                    )

            np_images = self._tensor_batch_to_preview_uint8(images.detach().cpu())
            save_dir = self.config.report_dir / "xai" / f"iter_{iteration}_{augmentation_name}"
            save_dir.mkdir(parents=True, exist_ok=True)

            class_names = self.config.detection_class_names
            paths: list[Path] = []
            self._latest_detection_xai_details = []
            per_class_scores: dict[str, list[float]] = defaultdict(list)

            ultra_predict_chw: Callable[[Tensor], dict[str, Tensor]] | None = None
            if use_ultra_xai and predict_ultra is not None:
                _pu = predict_ultra

                def _ultra_predict_chw(im: Tensor) -> dict[str, Tensor]:
                    d = _pu(im.unsqueeze(0))[0]
                    return {
                        k: v.to(device) if isinstance(v, Tensor) else v
                        for k, v in d.items()
                    }

                ultra_predict_chw = _ultra_predict_chw

            for idx in range(np_images.shape[0]):
                gt = targets[idx]
                pred = preds[idx]
                pred_scores = pred.get("scores", torch.zeros(0))
                pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long))
                # Keep top-k predictions for readability
                top_k = min(5, int(pred_scores.numel()))
                if top_k > 0:
                    top_idx = torch.argsort(pred_scores, descending=True)[:top_k]
                    pred_boxes = pred["boxes"][top_idx].detach().cpu()
                    pred_labels = pred_labels[top_idx].detach().cpu()
                    pred_scores = pred_scores[top_idx].detach().cpu()
                else:
                    pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
                    pred_labels = torch.zeros((0,), dtype=torch.long)
                    pred_scores = torch.zeros((0,), dtype=torch.float32)

                for cls, score in zip(pred_labels.tolist(), pred_scores.tolist()):
                    per_class_scores[str(int(cls))].append(float(score))

                gt_boxes = gt.get("boxes", torch.zeros((0, 4))).detach().cpu()
                gt_labels = gt.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu()

                # ── Choose saliency method ──
                sal: np.ndarray | None = None
                used_method = xai_method

                if xai_method == "activation" and activation_saliency is not None:
                    # Use the pre-computed activation saliency (fast path).
                    sal = activation_saliency[idx]
                elif xai_method == "occlusion":
                    # Per-box saliency via detection-conditioned occlusion sensitivity.
                    pred_boxes_cpu = pred_boxes.detach().cpu()
                    pred_labels_cpu = pred_labels.detach().cpu()
                    max_gt_boxes = int(self.config.detection_xai_max_gt_boxes)
                    max_pred_boxes = int(self.config.detection_xai_max_pred_boxes)
                    gt_sel = torch.arange(min(max_gt_boxes, int(gt_boxes.shape[0])))
                    pred_sel = torch.arange(min(max_pred_boxes, int(pred_boxes_cpu.shape[0])))
                    query_boxes_parts: list[Tensor] = []
                    query_labels_parts: list[Tensor] = []
                    if gt_sel.numel() > 0:
                        query_boxes_parts.append(gt_boxes[gt_sel])
                        query_labels_parts.append(gt_labels[gt_sel])
                    if pred_sel.numel() > 0:
                        query_boxes_parts.append(pred_boxes_cpu[pred_sel])
                        query_labels_parts.append(pred_labels_cpu[pred_sel])

                    if query_boxes_parts:
                        q_boxes = torch.cat(query_boxes_parts, dim=0)
                        q_labels = torch.cat(query_labels_parts, dim=0)
                        pred_for_occ = (
                            {
                                k: v.to(device) if isinstance(v, Tensor) else v
                                for k, v in pred.items()
                            }
                            if use_ultra_xai
                            else pred
                        )
                        sal_maps, _baseline_scores = compute_detection_box_saliency_occlusion(
                            model=model_impl,
                            image=images[idx],
                            query_boxes=q_boxes.to(device),
                            query_labels=q_labels.to(device),
                            baseline_pred=pred_for_occ,
                            predict_chw=ultra_predict_chw,
                            device=device,
                            grid_size=int(self.config.detection_xai_grid_size),
                            iou_threshold=0.3,
                        )
                        if sal_maps:
                            sal = np.mean(np.stack(sal_maps, axis=0), axis=0).astype(np.float32)

                img_h, img_w = images.shape[2], images.shape[3]

                def _box_saliency_stats(
                    boxes_tensor: Tensor,
                    labels_tensor: Tensor,
                    scores_tensor: Tensor | None = None,
                    saliency_map: np.ndarray | None = sal,
                    _img_h: int = img_h,
                    _img_w: int = img_w,
                ) -> list[dict[str, Any]]:
                    out: list[dict[str, Any]] = []
                    if saliency_map is None or boxes_tensor.numel() == 0:
                        return out
                    for bi in range(int(boxes_tensor.shape[0])):
                        bx1 = int(max(0, min(_img_w - 1, float(boxes_tensor[bi, 0].item()))))
                        by1 = int(max(0, min(_img_h - 1, float(boxes_tensor[bi, 1].item()))))
                        bx2 = int(max(0, min(_img_w, float(boxes_tensor[bi, 2].item()))))
                        by2 = int(max(0, min(_img_h, float(boxes_tensor[bi, 3].item()))))
                        if bx2 <= bx1 or by2 <= by1:
                            continue
                        patch = saliency_map[by1:by2, bx1:bx2]
                        mean_sal = float(np.mean(patch)) if patch.size > 0 else 0.0
                        max_sal = float(np.max(patch)) if patch.size > 0 else 0.0
                        row: dict[str, Any] = {
                            "box": [bx1, by1, bx2, by2],
                            "label": int(labels_tensor[bi].item()) if bi < int(labels_tensor.numel()) else -1,
                            "saliency_mean": mean_sal,
                            "saliency_max": max_sal,
                        }
                        if scores_tensor is not None and bi < int(scores_tensor.numel()):
                            row["score"] = float(scores_tensor[bi].item())
                        out.append(row)
                    return out

                details_row = {
                    "image_size": [int(img_h), int(img_w)],
                    "xai_method": used_method,
                    "gt": _box_saliency_stats(
                        gt.get("boxes", torch.zeros((0, 4))).detach().cpu(),
                        gt.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu(),
                    ),
                    "pred": _box_saliency_stats(
                        pred_boxes,
                        pred_labels,
                        pred_scores,
                    ),
                }
                self._latest_detection_xai_details.append(details_row)

                p_gt, p_sal, p_pred = save_detection_xai_panels(
                    image=np_images[idx],
                    saliency=sal,
                    boxes_gt=gt.get("boxes"),
                    labels_gt=gt.get("labels"),
                    boxes_pred=pred_boxes,
                    labels_pred=pred_labels,
                    scores_pred=pred_scores,
                    class_names=class_names,
                    save_path=save_dir / f"xai_{idx}.png",
                    size=self.config.report_xai_size,
                )
                paths.extend((p_gt, p_sal, p_pred))

            # Enrich insight text with per-class AP for better detection diagnostics.
            # Prefer cached full-eval predictions when available (from epoch_end_eval).
            from bnnr.detection_metrics import calculate_per_class_ap

            cached_preds = getattr(self.model, "last_eval_preds", [])
            cached_targets = getattr(self.model, "last_eval_targets", [])
            if cached_preds and cached_targets:
                ap_preds = cached_preds
                ap_targets = cached_targets
            else:
                ap_preds = [
                    {
                        "boxes": p.get("boxes", torch.zeros((0, 4))).detach().cpu(),
                        "scores": p.get("scores", torch.zeros((0,))).detach().cpu(),
                        "labels": p.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu(),
                    }
                    for p in preds
                ]
                ap_targets = [
                    {
                        "boxes": t.get("boxes", torch.zeros((0, 4))).detach().cpu(),
                        "labels": t.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu(),
                    }
                    for t in targets
                ]

            per_class_ap = calculate_per_class_ap(
                ap_preds, ap_targets, class_names=class_names,
            )

            xai_insights: dict[str, str] = {}
            xai_diagnoses: dict[str, dict[str, Any]] = {}
            for cls_id, scores in per_class_scores.items():
                if not scores:
                    continue
                avg_score = float(np.mean(scores))
                support = int(len(scores))
                ap_info = per_class_ap.get(cls_id, {"ap": 0.0, "support": 0})
                ap_val = float(ap_info.get("ap", 0.0))
                cls_name = (
                    class_names[int(cls_id)]
                    if class_names is not None and int(cls_id) < len(class_names)
                    else f"class_{cls_id}"
                )
                xai_insights[cls_id] = (
                    f"{cls_name}: AP@0.5={ap_val:.2f}, mean detection confidence={avg_score:.2f}, "
                    f"detections={support} ({used_method} XAI overlays available)."
                )
                xai_diagnoses[cls_id] = {
                    "severity": "ok" if ap_val >= 0.5 else ("warning" if ap_val >= 0.2 else "critical"),
                    "quality_score": ap_val,
                    "trend": "stable",
                    "short_text": xai_insights[cls_id],
                    "confused_with": [],
                    "ap_50": ap_val,
                    "detections": support,
                    "mean_confidence": avg_score,
                }
            return paths, xai_insights, xai_diagnoses

        if not isinstance(self.model, XAICapableModel):
            return [], {}, {}

        self._initialize_report_probe_samples()
        if self._report_probe_images is None or self._report_probe_labels is None:
            return [], {}, {}
        images = self._report_probe_images
        labels = self._report_probe_labels

        model_impl = self.model.get_model()
        device = next(model_impl.parameters()).device
        images = images.to(device)
        labels = labels.to(device)

        maps = generate_saliency_maps(
            model=model_impl,
            images=images,
            labels=labels,
            target_layers=self.model.get_target_layers(),
            method=self.config.xai_method,
        )

        # Lightweight predictions for insight context (no extra forward pass cost
        # when the model is already in eval mode from generate_saliency_maps).
        # For multi-label: XAI maps target one class, so we use dominant (argmax).
        with torch.no_grad():
            logits = model_impl(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        true_labels = labels.detach().cpu().numpy().tolist()

        # Resolve class names from run metadata if available
        class_names = None  # reassignment: list[str] | None
        if hasattr(self.reporter, "_event_sink") and self.reporter._event_sink is not None:
            # Try to get class names from the dataset profile event
            pass
        run_meta = getattr(self.reporter, "_run_class_names", None)
        if run_meta is None:
            # Best-effort: check the reporter's internal state
            pass

        # Enriched analysis when confusion data is available
        confusion_matrix: list[list[int]] | None = None
        if confusion and isinstance(confusion.get("matrix"), list):
            confusion_matrix = confusion["matrix"]

        xai_insights, xai_diagnoses, batch_stats = analyze_xai_batch_rich(
            maps,
            true_labels,
            preds,
            class_names=class_names,
            xai_method=self.config.xai_method,
            confusion_matrix=confusion_matrix,
            prev_batch_stats=self._prev_xai_batch_stats,
            augmentation_name=augmentation_name if augmentation_name != "baseline" else None,
            baseline_batch_stats=self._baseline_xai_stats or None,
        )
        # Store batch stats for the next checkpoint's trend comparison
        self._prev_xai_batch_stats = batch_stats

        np_images = self._tensor_batch_to_preview_uint8(images.detach().cpu())

        save_dir = self.config.report_dir / "xai" / f"iter_{iteration}_{augmentation_name}"
        paths = save_xai_visualization(
            np_images,
            maps,
            save_dir=save_dir,
            prefix="xai",
            output_size=self.config.report_xai_size,
        )
        return paths, xai_insights, xai_diagnoses

    @staticmethod
    def _xai_mean_quality(xai_diagnoses: dict[str, dict[str, Any]]) -> float | None:
        """Compute mean XAI quality score across all diagnosed classes.

        Returns ``None`` when no quality scores are available (e.g. XAI
        is disabled or no diagnoses were produced).
        """
        if not xai_diagnoses:
            return None
        scores = [
            float(d["quality_score"])
            for d in xai_diagnoses.values()
            if isinstance(d, dict) and "quality_score" in d
        ]
        return float(np.mean(scores)) if scores else None

    def _generate_xai_lightweight(
        self,
        iteration: int,
        augmentation_name: str,
        confusion: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, list[dict[str, float]]]]:
        """Lightweight XAI probe: analysis only, no PNG saving.

        Returns (insights, diagnoses, batch_stats).  Cost: one forward pass
        on the probe set + saliency computation (~50–100 ms per call).
        """
        if not self._xai_enabled or not isinstance(self.model, XAICapableModel):
            return {}, {}, {}

        # Detection XAI is handled in PR3; skip for now
        if self._is_detection:
            return {}, {}, {}

        self._initialize_report_probe_samples()
        if self._report_probe_images is None or self._report_probe_labels is None:
            return {}, {}, {}
        images = self._report_probe_images
        labels = self._report_probe_labels

        model_impl = self.model.get_model()
        device = next(model_impl.parameters()).device
        images = images.to(device)
        labels = labels.to(device)

        maps = generate_saliency_maps(
            model=model_impl,
            images=images,
            labels=labels,
            target_layers=self.model.get_target_layers(),
            method=self.config.xai_method,
        )

        # For multi-label: XAI maps target one class, so we use dominant (argmax).
        with torch.no_grad():
            logits = model_impl(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        true_labels = labels.detach().cpu().numpy().tolist()

        confusion_matrix: list[list[int]] | None = None
        if confusion and isinstance(confusion.get("matrix"), list):
            confusion_matrix = confusion["matrix"]

        xai_insights, xai_diagnoses, batch_stats = analyze_xai_batch_rich(
            maps,
            true_labels,
            preds,
            class_names=None,
            xai_method=self.config.xai_method,
            confusion_matrix=confusion_matrix,
            prev_batch_stats=self._prev_xai_batch_stats,
            augmentation_name=augmentation_name,
            baseline_batch_stats=self._baseline_xai_stats or None,
        )
        return xai_insights, xai_diagnoses, batch_stats

    def _tensor_batch_to_preview_uint8(self, images: Tensor) -> np.ndarray:
        np_images: np.ndarray = images.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
        out: np.ndarray = np.zeros_like(np_images, dtype=np.uint8)

        for idx in range(np_images.shape[0]):
            sample = np_images[idx]
            converted: np.ndarray | None = None

            # Fast path for already image-like tensors.
            if float(sample.min()) >= 0.0 and float(sample.max()) <= 1.0:
                converted = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
            elif float(sample.min()) >= 0.0 and float(sample.max()) <= 255.0:
                converted = np.clip(sample, 0, 255).astype(np.uint8)
            else:
                # Preferred: user-configured denorm, then generic fallback.
                cfg_mean = self.config.denormalization_mean
                cfg_std = self.config.denormalization_std
                if cfg_mean is not None and cfg_std is not None:
                    mean = np.array(cfg_mean, dtype=np.float32)
                    std = np.array(cfg_std, dtype=np.float32)
                    # Broadcast for any number of channels
                    denorm = sample * std + mean
                    denorm = np.clip(denorm, 0.0, 1.0)
                    if float(denorm.max() - denorm.min()) > 1e-6:
                        converted = (denorm * 255.0).astype(np.uint8)

            if converted is None:
                min_val = float(sample.min())
                max_val = float(sample.max())
                if max_val - min_val < 1e-8:
                    converted = np.zeros_like(sample, dtype=np.uint8)
                else:
                    normed = (sample - min_val) / (max_val - min_val)
                    converted = np.clip(normed * 255.0, 0, 255).astype(np.uint8)

            out[idx] = converted
        return out  # type: ignore[no-any-return]

    def _generate_augmentation_previews(
        self,
        iteration: int,
        augmentation_name: str,
        augmentations: list[BaseAugmentation] | None,
    ) -> list[tuple[Path, Path]]:
        if self._is_detection:
            from bnnr.detection_xai import draw_boxes_on_image

            self._initialize_report_probe_samples()
            if self._report_probe_images is None or self._report_probe_targets is None:
                return []

            images = self._report_probe_images
            targets = copy.deepcopy(self._report_probe_targets)
            batch: Any = (images, targets)

            if augmentations:
                preview_augs = [copy.copy(a) for a in augmentations]
                for pa in preview_augs:
                    pa.probability = 1.0
                for pa in preview_augs:
                    batch = self._apply_augmentation_to_batch(batch, pa)

            aug_images, aug_targets = batch
            np_images = self._tensor_batch_to_preview_uint8(images)
            np_aug_images = self._tensor_batch_to_preview_uint8(aug_images)
            save_dir = self.config.report_dir / "samples" / f"iter_{iteration}_{augmentation_name}"
            save_dir.mkdir(parents=True, exist_ok=True)
            pairs: list[tuple[Path, Path]] = []

            for idx in range(np_images.shape[0]):
                original = np_images[idx]
                augmented = np_aug_images[idx]
                if original.shape[-1] == 1:
                    original = np.repeat(original, 3, axis=2)
                if augmented.shape[-1] == 1:
                    augmented = np.repeat(augmented, 3, axis=2)

                original = draw_boxes_on_image(
                    original,
                    targets[idx].get("boxes", torch.zeros((0, 4))),
                    targets[idx].get("labels", torch.zeros((0,), dtype=torch.long)),
                    class_names=self.config.detection_class_names,
                )
                augmented = draw_boxes_on_image(
                    augmented,
                    aug_targets[idx].get("boxes", torch.zeros((0, 4))),
                    aug_targets[idx].get("labels", torch.zeros((0,), dtype=torch.long)),
                    class_names=self.config.detection_class_names,
                )

                original = cv2.resize(
                    original,
                    (self.config.report_preview_size, self.config.report_preview_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                augmented = cv2.resize(
                    augmented,
                    (self.config.report_preview_size, self.config.report_preview_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                orig_path = save_dir / f"sample_{idx}_original.png"
                aug_path = save_dir / f"sample_{idx}_augmented.png"
                cv2.imwrite(str(orig_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(aug_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                pairs.append((orig_path, aug_path))
            return pairs

        self._initialize_report_probe_samples()
        if self._report_probe_images is None or self._report_probe_labels is None:
            return []
        images = self._report_probe_images
        labels = self._report_probe_labels
        sample_indices_np = None
        np_images = self._tensor_batch_to_preview_uint8(images)

        augmented = np_images.copy()
        if augmentations:
            preview_augs = [copy.copy(a) for a in augmentations]
            for pa in preview_augs:
                pa.probability = 1.0
            np_labels = labels.detach().cpu().numpy()
            for pa in preview_augs:
                if hasattr(pa, "apply_batch_with_labels"):
                    augmented = pa.apply_batch_with_labels(augmented, np_labels, sample_indices=sample_indices_np)
                else:
                    augmented = pa.apply_batch(augmented)

        save_dir = self.config.report_dir / "samples" / f"iter_{iteration}_{augmentation_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        pairs = []  # list[tuple[Path, Path]] — reassigned from earlier scope
        labels_np = labels.detach().cpu().numpy().squeeze()
        if labels_np.ndim == 0:
            labels_np = labels_np.reshape(1)
        for idx in range(np_images.shape[0]):
            original = np_images[idx]
            aug = augmented[idx]  # type: ignore[assignment]
            class_id = int(labels_np[idx])
            if original.shape[-1] == 1:
                original = np.repeat(original, 3, axis=2)
            if aug.shape[-1] == 1:  # type: ignore[union-attr,attr-defined]
                aug = np.repeat(aug, 3, axis=2)  # type: ignore[assignment]
            # Save larger thumbnails for clearer inspection in HTML report.
            original = cv2.resize(
                original,
                (self.config.report_preview_size, self.config.report_preview_size),
                interpolation=cv2.INTER_NEAREST,
            )
            aug = cv2.resize(
                aug,
                (self.config.report_preview_size, self.config.report_preview_size),
                interpolation=cv2.INTER_NEAREST,
            )
            orig_path = save_dir / f"sample_{idx}_class_{class_id}_original.png"
            aug_path = save_dir / f"sample_{idx}_class_{class_id}_augmented.png"
            cv2.imwrite(str(orig_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(aug_path), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
            pairs.append((orig_path, aug_path))
        return pairs

    def _initialize_report_probe_samples(self) -> None:
        if self._report_probe_images is not None and self._report_probe_labels is not None:
            return

        images_by_class: dict[int, list[Tensor]] = defaultdict(list)
        targets_by_class: dict[int, list[dict[str, Tensor] | None]] = defaultdict(list)
        indices_by_class: dict[int, list[int | None]] = defaultdict(list)
        for raw_batch in self.val_loader:
            if self._is_detection:
                # Detection: register each image under ALL classes
                # it contains so that probe selection is distributed
                # evenly across all present classes (not biased toward
                # the largest-box class only).
                if len(raw_batch) == 3:
                    images, targets, sample_indices = raw_batch
                else:
                    images, targets = raw_batch
                    sample_indices = None
                for idx in range(images.shape[0]):
                    target = targets[idx]
                    if len(target.get("labels", [])) == 0:
                        continue
                    # Register under each unique class present in the image
                    unique_classes = set(int(lbl.item()) for lbl in target["labels"])
                    for class_id in unique_classes:
                        images_by_class[class_id].append(images[idx].detach().cpu())
                        targets_by_class[class_id].append({
                            "boxes": target["boxes"].detach().cpu().clone(),
                            "labels": target["labels"].detach().cpu().clone(),
                        })
                        if sample_indices is not None:
                            current_idx = sample_indices[idx]
                            if isinstance(current_idx, Tensor):
                                indices_by_class[class_id].append(int(current_idx.item()))
                            else:
                                indices_by_class[class_id].append(int(current_idx))
                        else:
                            indices_by_class[class_id].append(None)
                continue

            # Classification / multilabel path
            if len(raw_batch) == 3:
                images, labels, sample_indices = raw_batch
            else:
                images, labels = raw_batch
                sample_indices = None

            if self._is_multilabel:
                # Multi-label: register image under ALL active classes
                for idx in range(images.shape[0]):
                    label_vec = labels[idx]
                    active_classes = [
                        cls_idx
                        for cls_idx in range(label_vec.shape[0])
                        if int(label_vec[cls_idx]) == 1
                    ]
                    if not active_classes:
                        active_classes = [int(label_vec.argmax().item())]
                    for class_id in active_classes:
                        images_by_class[class_id].append(images[idx].detach().cpu())
                        if sample_indices is not None:
                            si = sample_indices[idx]
                            indices_by_class[class_id].append(
                                int(si.item()) if isinstance(si, Tensor) else int(si)
                            )
                        else:
                            indices_by_class[class_id].append(None)
            else:
                for idx in range(images.shape[0]):
                    lbl = labels[idx]
                    # Squeeze extra dims (e.g. MedMNIST returns [batch, 1])
                    if lbl.ndim >= 1:
                        lbl = lbl.squeeze()
                    class_id = int(lbl.item())
                    images_by_class[class_id].append(images[idx].detach().cpu())
                    if sample_indices is not None:
                        indices_by_class[class_id].append(int(sample_indices[idx].item()))
                    else:
                        indices_by_class[class_id].append(None)

        rnd = np.random.default_rng(self.config.seed)
        selected_images: list[Tensor] = []
        selected_labels: list[int] = []
        selected_targets: list[dict[str, Tensor]] = []
        selected_sample_ids: list[str] = []
        classes = sorted(images_by_class.keys())[: self.config.report_probe_max_classes]
        per_class = self.config.report_probe_images_per_class
        for class_id in classes:
            class_images = images_by_class[class_id]
            class_indices = indices_by_class[class_id]
            if not class_images:
                continue
            sample_count = min(per_class, len(class_images))
            chosen_indices = rnd.choice(len(class_images), size=sample_count, replace=False)
            for probe_rank, index in enumerate(chosen_indices):
                real_index = int(index)
                selected_images.append(class_images[real_index])
                selected_labels.append(class_id)
                if self._is_detection:
                    target_entry = targets_by_class[class_id][real_index]
                    if target_entry is not None:
                        selected_targets.append(target_entry)
                sample_idx = class_indices[real_index]
                if sample_idx is None:
                    selected_sample_ids.append(f"class_{class_id}_probe_{probe_rank}")
                else:
                    selected_sample_ids.append(f"sample_{sample_idx}")

        if not selected_images:
            return
        self._report_probe_images = torch.stack(selected_images)
        self._report_probe_labels = torch.as_tensor(selected_labels, dtype=torch.long)
        self._report_probe_targets = selected_targets if self._is_detection else None
        self._report_probe_sample_ids = selected_sample_ids
        log_probe_set = getattr(self.reporter, "log_probe_set", None)
        if callable(log_probe_set):
            probes = [
                {"sample_id": sample_id, "class_id": label, "index": idx}
                for idx, (sample_id, label) in enumerate(zip(self._report_probe_sample_ids, selected_labels))
            ]
            log_probe_set(probes)

    def _probe_labels(self) -> list[int]:
        if self._report_probe_labels is None:
            return []
        return [int(v) for v in self._report_probe_labels.detach().cpu().numpy().tolist()]

    def _probe_sample_ids(self) -> list[str]:
        return list(self._report_probe_sample_ids)

    def _collect_val_logits(
        self,
        *,
        post_process: str = "argmax",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Run a forward pass over the val set and collect predictions/labels.

        ``post_process``:
        * ``"argmax"``  → standard classification (argmax over logits)
        * ``"sigmoid"`` → multi-label (sigmoid ≥ threshold → int)

        Returns ``(preds, labels)`` as int64 numpy arrays, or ``None``
        if the model is incompatible or the val set is empty.
        """
        if self._last_eval_preds is not None and self._last_eval_labels is not None:
            return self._last_eval_preds, self._last_eval_labels

        if not isinstance(self.model, XAICapableModel):
            return None
        model_impl = self.model.get_model()
        device = next(model_impl.parameters()).device
        model_impl.eval()
        preds_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        with torch.no_grad():
            for raw_batch in self.val_loader:
                if len(raw_batch) == 3:
                    images, labels_batch, _ = raw_batch
                else:
                    images, labels_batch = raw_batch
                logits = model_impl(images.to(device))
                if post_process == "sigmoid":
                    preds_rows.append(
                        (torch.sigmoid(logits) >= self.config.multilabel_threshold).int().cpu()
                    )
                else:
                    preds_rows.append(torch.argmax(logits, dim=1).cpu())
                label_rows.append(labels_batch.cpu())
        if not preds_rows or not label_rows:
            return None
        preds = torch.cat(preds_rows).numpy().astype(np.int64)
        labels = torch.cat(label_rows).numpy().astype(np.int64)
        return preds, labels

    def _compute_eval_class_details(self) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
        # ── Detection path ──────────────────────────────────────────────
        if self._is_detection:
            return self._compute_eval_class_details_detection()

        # ── Multi-label path ────────────────────────────────────────────
        if self._is_multilabel:
            return self._compute_eval_class_details_multilabel()

        # ── Classification path ─────────────────────────────────────────
        result = self._collect_val_logits(post_process="argmax")
        if result is None:
            return {}, {}
        preds, labels = result

        n_classes = int(max(int(np.max(preds)), int(np.max(labels)))) + 1
        per_class: dict[str, dict[str, float | int]] = {}
        for class_id in range(n_classes):
            mask = labels == class_id
            support = int(np.sum(mask))
            if support == 0:
                continue
            acc = float(np.mean(preds[mask] == labels[mask]))
            per_class[str(class_id)] = {"accuracy": acc, "support": support}
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
            matrix[int(true_label), int(pred_label)] += 1
        confusion = {
            "labels": list(range(n_classes)),
            "matrix": matrix.tolist(),
        }
        return per_class, confusion

    def _compute_eval_class_details_multilabel(self) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
        """Compute per-label precision/recall/f1 for multi-label classification.

        Returns per-label metrics and a per-label TP/FP/FN summary instead of
        a full NxN confusion matrix (which is not meaningful for multi-label).
        """
        result = self._collect_val_logits(post_process="sigmoid")
        if result is None:
            return {}, {}
        preds, labels = result

        n_labels = preds.shape[1] if preds.ndim == 2 else 0
        if n_labels == 0:
            return {}, {}

        per_label: dict[str, dict[str, float | int]] = {}
        confusion_per_label: list[dict[str, int]] = []
        for label_idx in range(n_labels):
            y_true = labels[:, label_idx]
            y_pred = preds[:, label_idx]
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            support = int(np.sum(y_true == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_label[str(label_idx)] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }
            confusion_per_label.append({"tp": tp, "fp": fp, "fn": fn})

        confusion = {
            "type": "multilabel_per_label",
            "labels": list(range(n_labels)),
            "per_label": confusion_per_label,
        }
        return per_label, confusion

    def _compute_eval_class_details_detection(self) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
        """Compute per-class AP for detection task.

        Reuses ``last_eval_preds`` / ``last_eval_targets`` saved by
        ``DetectionAdapter.epoch_end_eval()`` to avoid a redundant full
        evaluation pass.  Falls back to a single forward pass only when
        the cached data is empty.
        """
        from bnnr.detection_metrics import (
            calculate_detection_confusion_matrix,
            calculate_per_class_ap,
        )

        # Try to reuse cached eval results from the adapter
        all_preds: list[dict[str, Tensor]] = getattr(self.model, "last_eval_preds", [])
        all_targets: list[dict[str, Tensor]] = getattr(self.model, "last_eval_targets", [])

        # Fallback: run a single forward pass if no cached data
        if not all_preds or not all_targets:
            all_preds = []
            all_targets = []
            model_obj = self.model.get_model() if hasattr(self.model, "get_model") else None
            if model_obj is None:
                return {}, {}

            # Ultralytics task models expect BCHW tensors, not a list like torchvision detection.
            use_ultra_fallback = _is_ultralytics_tasks_backbone(model_obj) and hasattr(
                self.model, "eval_step",
            )
            if use_ultra_fallback:
                self.model._eval_preds = []  # type: ignore[attr-defined]
                self.model._eval_targets = []  # type: ignore[attr-defined]
                with torch.no_grad():
                    for raw_batch in self.val_loader:
                        if len(raw_batch) == 3:
                            images, targets, _ = raw_batch
                        else:
                            images, targets = raw_batch
                        self.model.eval_step((images, targets))
                all_preds = copy.deepcopy(self.model._eval_preds)  # type: ignore[attr-defined]
                all_targets = copy.deepcopy(self.model._eval_targets)  # type: ignore[attr-defined]
                self.model._eval_preds = []  # type: ignore[attr-defined]
                self.model._eval_targets = []  # type: ignore[attr-defined]
            else:
                model_obj.eval()
                with torch.no_grad():
                    for raw_batch in self.val_loader:
                        if len(raw_batch) == 3:
                            images, targets, _ = raw_batch
                        else:
                            images, targets = raw_batch
                        device = next(model_obj.parameters()).device
                        images_list = [img.to(device) for img in images]
                        preds = model_obj(images_list)
                        for pred in preds:
                            all_preds.append({
                                "boxes": pred["boxes"].cpu(),
                                "scores": pred["scores"].cpu(),
                                "labels": pred["labels"].cpu(),
                            })
                        for target in targets:
                            all_targets.append({
                                "boxes": target["boxes"].cpu() if isinstance(target["boxes"], Tensor) else target["boxes"],
                                "labels": target["labels"].cpu() if isinstance(target["labels"], Tensor) else target["labels"],
                            })

        if not all_preds or not all_targets:
            return {}, {}

        class_names = self.config.detection_class_names

        # Determine known class IDs from dataset (GT labels only)
        known_classes: set[int] = set()
        for t in all_targets:
            if t.get("labels") is not None and len(t["labels"]) > 0:
                known_classes.update(int(x) for x in t["labels"].cpu().tolist())

        per_class_ap = calculate_per_class_ap(all_preds, all_targets, class_names=class_names)

        # Convert to the expected format with proper detection naming
        per_class: dict[str, dict[str, float | int]] = {}
        for cls_id, info in per_class_ap.items():
            per_class[cls_id] = {
                "accuracy": info["ap"],
                "ap_50": info["ap"],
                "support": info["support"],
            }

        # Build class-level confusion matrix for dashboard diagnostics.
        # Filter to known GT classes + background (class 0) to avoid
        # phantom predicted-only classes cluttering the matrix.
        confusion = calculate_detection_confusion_matrix(
            predictions=all_preds,
            targets=all_targets,
            num_classes=(
                (
                    len(self.config.detection_class_names)
                    if (
                        self.config.detection_class_names
                        and str(self.config.detection_class_names[0]).strip().lower()
                        in {"background", "bg", "__background__"}
                    )
                    else len(self.config.detection_class_names) + 1
                )
                if self.config.detection_class_names is not None else None
            ),
            iou_threshold=0.5,
        )

        # Filter confusion matrix to known classes only
        if known_classes and confusion.get("labels"):
            allowed = known_classes | {0}  # always keep background
            old_labels = confusion["labels"]
            old_matrix = confusion["matrix"]
            keep_indices = [i for i, lbl in enumerate(old_labels) if lbl in allowed]
            if len(keep_indices) < len(old_labels):
                new_labels = [old_labels[i] for i in keep_indices]
                new_matrix = [
                    [old_matrix[r][c] for c in keep_indices]
                    for r in keep_indices
                ]
                confusion = {"labels": new_labels, "matrix": new_matrix}

        return per_class, confusion

    def _emit_probe_prediction_snapshots(
        self,
        *,
        iteration: int,
        epoch: int,
        branch: str,
        preview_pairs: list[tuple[Path, Path]],
        xai_paths: list[Path],
    ) -> None:
        log_prediction = getattr(self.reporter, "log_sample_prediction", None)
        if not callable(log_prediction):
            return
        if self._is_detection:
            model_getter = getattr(self.model, "get_model", None)
            if not callable(model_getter):
                return
            self._initialize_report_probe_samples()
            if self._report_probe_images is None or self._report_probe_targets is None:
                return
            model = model_getter()
            device = next(model.parameters()).device
            images = self._report_probe_images.to(device)
            predict_ultra = getattr(self.model, "predict_detection_dicts", None)
            if _is_ultralytics_tasks_backbone(model) and not callable(predict_ultra):
                self._log(
                    "Probe prediction snapshots skipped: Ultralytics backbone without "
                    "UltralyticsDetectionAdapter.predict_detection_dicts."
                )
                return
            with torch.no_grad():
                if callable(predict_ultra):
                    preds = predict_ultra(images)
                else:
                    preds = model([img for img in images])

            sample_ids = self._probe_sample_ids()
            reporter_run_dir = getattr(self.reporter, "run_dir", None)
            fallback_originals = self._ensure_fast_probe_originals(
                iteration=iteration,
                branch=branch,
                epoch=epoch,
                run_dir=reporter_run_dir,
            )
            normalized_preview_pairs = preview_pairs
            normalized_xai_paths = xai_paths
            checkpoints = getattr(self.reporter, "_checkpoints", None)
            if isinstance(checkpoints, list) and checkpoints:
                latest = checkpoints[-1]
                if getattr(latest, "iteration", None) == iteration and getattr(latest, "augmentation", None) == branch:
                    normalized_preview_pairs = list(getattr(latest, "preview_pairs", preview_pairs))
                    normalized_xai_paths = list(getattr(latest, "xai_paths", xai_paths))

            n_probes = len(sample_ids)
            xai_triplet_mode = (
                n_probes > 0
                and len(normalized_xai_paths) == 3 * n_probes
            )
            for idx, sample_id in enumerate(sample_ids):
                gt = self._report_probe_targets[idx]
                pred = preds[idx]
                pred_scores = pred.get("scores", torch.zeros(0))
                pred_labels = pred.get("labels", torch.zeros(0, dtype=torch.long))
                if pred_scores.numel() > 0:
                    top = int(torch.argmax(pred_scores).item())
                    predicted_class = int(pred_labels[top].item())
                    confidence = float(pred_scores[top].item())
                else:
                    predicted_class = -1
                    confidence = 0.0

                if gt.get("labels") is not None and int(gt["labels"].numel()) > 0:
                    true_class = int(gt["labels"][0].item())
                else:
                    true_class = -1

                original_artifact = None
                augmented_artifact = None
                xai_artifact = None
                xai_gt_artifact = None
                xai_saliency_artifact = None
                xai_pred_artifact = None
                if idx < len(normalized_preview_pairs):
                    original_artifact = self._to_artifact_reference(normalized_preview_pairs[idx][0], reporter_run_dir)
                    augmented_artifact = self._to_artifact_reference(normalized_preview_pairs[idx][1], reporter_run_dir)
                elif idx < len(fallback_originals):
                    # Keep Samples tab usable immediately after epoch_end:
                    # if full previews are not ready yet, expose a fast original.
                    original_artifact = self._to_artifact_reference(fallback_originals[idx], reporter_run_dir)
                if xai_triplet_mode:
                    base = idx * 3
                    xai_gt_artifact = self._to_artifact_reference(normalized_xai_paths[base], reporter_run_dir)
                    xai_saliency_artifact = self._to_artifact_reference(
                        normalized_xai_paths[base + 1], reporter_run_dir,
                    )
                    xai_pred_artifact = self._to_artifact_reference(
                        normalized_xai_paths[base + 2], reporter_run_dir,
                    )
                    xai_artifact = xai_saliency_artifact
                elif idx < len(normalized_xai_paths):
                    xai_artifact = self._to_artifact_reference(normalized_xai_paths[idx], reporter_run_dir)

                detection_details: dict[str, Any] = {}
                if idx < len(self._latest_detection_xai_details):
                    detection_details = self._latest_detection_xai_details[idx]

                log_prediction(
                    sample_id=sample_id,
                    iteration=iteration,
                    epoch=epoch,
                    branch=branch,
                    true_class=true_class,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    loss_local=None,
                    original_artifact=original_artifact,
                    augmented_artifact=augmented_artifact,
                    xai_artifact=xai_artifact,
                    xai_gt_artifact=xai_gt_artifact,
                    xai_saliency_artifact=xai_saliency_artifact,
                    xai_pred_artifact=xai_pred_artifact,
                    detection_details=detection_details,
                )
            return

        if not isinstance(self.model, XAICapableModel):
            return
        self._initialize_report_probe_samples()
        if self._report_probe_images is None or self._report_probe_labels is None:
            return
        model = self.model.get_model()
        device = next(model.parameters()).device
        images = self._report_probe_images.to(device)
        labels = self._report_probe_labels.to(device)
        with torch.no_grad():
            logits = model(images)
            if self._is_multilabel:
                # Multi-label: use sigmoid + threshold; report dominant class
                probs = torch.sigmoid(logits)
                conf, pred = torch.max(probs, dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
        sample_ids = self._probe_sample_ids()
        true_labels = labels.detach().cpu().numpy().tolist()
        pred_labels = pred.detach().cpu().numpy().tolist()
        conf_scores = conf.detach().cpu().numpy().tolist()
        reporter_run_dir = getattr(self.reporter, "run_dir", None)
        normalized_preview_pairs = preview_pairs
        normalized_xai_paths = xai_paths
        checkpoints = getattr(self.reporter, "_checkpoints", None)
        if isinstance(checkpoints, list) and checkpoints:
            latest = checkpoints[-1]
            if getattr(latest, "iteration", None) == iteration and getattr(latest, "augmentation", None) == branch:
                normalized_preview_pairs = list(getattr(latest, "preview_pairs", preview_pairs))
                normalized_xai_paths = list(getattr(latest, "xai_paths", xai_paths))
        for idx, sample_id in enumerate(sample_ids):
            original_artifact = None
            augmented_artifact = None
            xai_artifact = None
            if idx < len(normalized_preview_pairs):
                original_artifact = self._to_artifact_reference(normalized_preview_pairs[idx][0], reporter_run_dir)
                augmented_artifact = self._to_artifact_reference(normalized_preview_pairs[idx][1], reporter_run_dir)
            if idx < len(normalized_xai_paths):
                xai_artifact = self._to_artifact_reference(normalized_xai_paths[idx], reporter_run_dir)
            log_prediction(
                sample_id=sample_id,
                iteration=iteration,
                epoch=epoch,
                branch=branch,
                true_class=int(true_labels[idx]),
                predicted_class=int(pred_labels[idx]),
                confidence=float(conf_scores[idx]),
                loss_local=None,
                original_artifact=original_artifact,
                augmented_artifact=augmented_artifact,
                xai_artifact=xai_artifact,
            )

    def _ensure_fast_probe_originals(
        self,
        *,
        iteration: int,
        branch: str,
        epoch: int,
        run_dir: Path | None,
    ) -> list[Path]:
        """Create lightweight original probe previews for immediate dashboard updates."""
        key = (iteration, branch, epoch)
        cached = self._fast_probe_originals.get(key)
        if cached:
            return cached
        if self._report_probe_images is None:
            return []

        save_root = (
            Path(run_dir) / "artifacts"
            if run_dir is not None
            else self.config.report_dir / "artifacts"
        )
        save_dir = save_root / "samples" / f"iter_{iteration}_{branch}" / f"epoch_{epoch}" / "fast_originals"
        save_dir.mkdir(parents=True, exist_ok=True)

        np_images = self._tensor_batch_to_preview_uint8(self._report_probe_images)
        paths: list[Path] = []
        if self._is_detection and self._report_probe_targets is not None:
            from bnnr.detection_xai import draw_boxes_on_image

            for idx in range(np_images.shape[0]):
                original = np_images[idx]
                if original.shape[-1] == 1:
                    original = np.repeat(original, 3, axis=2)
                target = self._report_probe_targets[idx]
                overlay = draw_boxes_on_image(
                    original,
                    target.get("boxes", torch.zeros((0, 4))),
                    target.get("labels", torch.zeros((0,), dtype=torch.long)),
                    class_names=self.config.detection_class_names,
                )
                overlay = cv2.resize(
                    overlay,
                    (self.config.report_preview_size, self.config.report_preview_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                out_path = save_dir / f"sample_{idx}_original.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                paths.append(out_path)
        else:
            for idx in range(np_images.shape[0]):
                original = np_images[idx]
                if original.shape[-1] == 1:
                    original = np.repeat(original, 3, axis=2)
                original = cv2.resize(
                    original,
                    (self.config.report_preview_size, self.config.report_preview_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                out_path = save_dir / f"sample_{idx}_original.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
                paths.append(out_path)

        self._fast_probe_originals[key] = paths
        return paths

    def _to_artifact_reference(self, path: Path, run_dir: Path | None) -> str:
        if run_dir is not None:
            try:
                return str(path.resolve().relative_to(Path(run_dir).resolve())).replace("\\", "/")
            except ValueError:
                pass
        text = str(path).replace("\\", "/")
        marker = "/artifacts/"
        if marker in text:
            suffix = text.split(marker, 1)[1]
            return f"artifacts/{suffix}"
        return text

    def _adapt_icd_thresholds(self, xai_diagnoses: dict[str, dict[str, Any]]) -> None:
        """Adjust ICD/AICD threshold_percentile based on XAI coverage and focus.

        Rules:
        - Hyper-focused (coverage < 5%, gini > 0.8): increase threshold by 5pp
          (mask more, force broader learning).
        - Too scattered (coverage > 30%, gini < 0.4): decrease threshold by 5pp
          (mask less, let model keep broader features).

        Only applies when ``config.adaptive_icd_threshold`` is True.
        """
        # Compute mean coverage and gini across all diagnosed classes
        coverages: list[float] = []
        ginis: list[float] = []
        for diag in xai_diagnoses.values():
            breakdown = diag.get("quality_breakdown", {})
            if isinstance(breakdown, dict):
                # We need raw stats, not scores — use the text heuristic or just
                # look at the breakdown components.
                pass
        # Use batch stats as the source of truth (more reliable)
        for cls_stats in self._prev_xai_batch_stats.values():
            for s in cls_stats:
                coverages.append(s.get("coverage", 0.0))
                ginis.append(s.get("gini", 0.0))

        if not coverages:
            return

        mean_cov = float(np.mean(coverages))
        mean_gini = float(np.mean(ginis))

        adjustment = 0
        if mean_cov < 0.05 and mean_gini > 0.8:
            adjustment = 5  # increase masking
        elif mean_cov > 0.30 and mean_gini < 0.4:
            adjustment = -5  # decrease masking

        if adjustment == 0:
            return

        for aug in self.augmentations:
            if isinstance(aug, (ICD, AICD)):
                old_tp = getattr(aug, "threshold_percentile", 75)
                new_tp = max(50, min(90, old_tp + adjustment))
                aug.threshold_percentile = new_tp
                self._log(
                    f"Adaptive ICD: {aug.name} threshold_percentile "
                    f"{old_tp} → {new_tp} (coverage={mean_cov:.2f}, gini={mean_gini:.2f})"
                )

    def _compute_eval_analysis(self) -> dict[str, Any]:
        # ── Detection path ──────────────────────────────────────────────
        if self._is_detection:
            per_class, _ = self._compute_eval_class_details_detection()
            if not per_class:
                return {}
            ap_values = [v.get("accuracy", 0.0) for v in per_class.values()]
            return {
                "per_class_accuracy": per_class,
                "macro_per_class_accuracy": float(np.mean(ap_values)) if ap_values else None,
            }

        # ── Multi-label path ────────────────────────────────────────────
        if self._is_multilabel:
            per_label, _ = self._compute_eval_class_details_multilabel()
            if not per_label:
                return {}
            f1_values = [v.get("f1", 0.0) for v in per_label.values()]
            return {
                "per_label_f1": per_label,
                "macro_per_label_f1": float(np.mean(f1_values)) if f1_values else None,
            }

        # ── Classification path (unchanged) ─────────────────────────────
        if not isinstance(self.model, XAICapableModel):
            return {}

        model = self.model.get_model()
        device = next(model.parameters()).device
        model.eval()
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for raw_batch in self.val_loader:
                if len(raw_batch) == 3:
                    images, labels, _ = raw_batch
                else:
                    images, labels = raw_batch
                logits = model(images.to(device))
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(labels.cpu())

        if not all_preds or not all_labels:
            return {}

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        n_classes = int(max(int(preds.max().item()), int(labels.max().item()))) + 1

        per_class_accuracy: dict[str, dict[str, float | int]] = {}
        class_acc_values: list[float] = []
        for class_id in range(n_classes):
            mask = labels == class_id
            support = int(mask.sum().item())
            if support == 0:
                continue
            class_acc = float((preds[mask] == labels[mask]).float().mean().item())
            per_class_accuracy[str(class_id)] = {
                "accuracy": class_acc,
                "support": support,
            }
            class_acc_values.append(class_acc)

        return {
            "per_class_accuracy": per_class_accuracy,
            "macro_per_class_accuracy": float(np.mean(class_acc_values)) if class_acc_values else None,
        }

    def _build_xai_insights(
        self,
        baseline_metrics: dict[str, float],
        best_metrics: dict[str, float],
        selected_augmentations: list[str],
    ) -> list[str]:
        metric = self.config.selection_metric
        baseline_value = baseline_metrics.get(metric)
        best_value = best_metrics.get(metric)
        insights: list[str] = []

        if isinstance(baseline_value, (int, float)) and isinstance(best_value, (int, float)):
            delta = float(best_value) - float(baseline_value)
            direction = "higher" if self.config.selection_mode == "max" else "lower"
            insights.append(f"Best {metric} is {delta:+.4f} vs baseline ({direction} is better).")

        if any("icd" == name for name in selected_augmentations):
            insights.append("ICD was selected: model benefits from dropping highly salient regions.")
        if any("aicd" == name for name in selected_augmentations):
            insights.append("AICD was selected: model benefits from suppressing low-saliency background context.")
        if self.config.xai_method.lower() in {"craft", "nmf", "nmf_concepts"}:
            insights.append("NMF concept maps were used for saliency estimation in this run.")
        else:
            insights.append("OptiCAM heatmaps were used for saliency estimation in this run.")

        return insights

    @staticmethod
    def _saliency_recommendations(
        batch_stats: dict[str, list[dict[str, float]]],
        xai_diagnoses: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        """Generate augmentation recommendations from saliency statistics.

        Shared logic used by both ``_build_xai_summary`` (final report)
        and ``_generate_augmentation_hints`` (live training logs).

        Parameters
        ----------
        batch_stats
            Per-class saliency stats (``_prev_xai_batch_stats``).
        xai_diagnoses
            Optional per-class diagnoses for severity-based hints.

        Returns
        -------
        list[str]
            Human-readable recommendation strings.
        """
        if not batch_stats:
            return []

        all_coverages: list[float] = []
        all_ginis: list[float] = []
        all_edges: list[float] = []
        all_coherences: list[float] = []
        for stats_list in batch_stats.values():
            for s in stats_list:
                all_coverages.append(s.get("coverage", 0.0))
                all_ginis.append(s.get("gini", 0.0))
                all_edges.append(s.get("edge_ratio", 0.0))
                all_coherences.append(s.get("spatial_coherence", 0.0))

        mean_cov = float(np.mean(all_coverages)) if all_coverages else 0.0
        mean_gini = float(np.mean(all_ginis)) if all_ginis else 0.0
        mean_edge = float(np.mean(all_edges)) if all_edges else 0.0
        mean_coh = float(np.mean(all_coherences)) if all_coherences else 0.0

        hints: list[str] = []

        # Coverage
        if mean_cov < 0.05 and all_coverages:
            hints.append(
                f"Very narrow attention (coverage={mean_cov:.1%}). "
                "ICD augmentation may force the model to learn broader features."
            )
        elif mean_cov > 0.40 and all_coverages:
            hints.append(
                f"Diffuse attention (coverage={mean_cov:.1%}). "
                "AICD augmentation may help sharpen focus on discriminative regions."
            )

        # Focus (Gini)
        if mean_gini < 0.3 and all_ginis:
            hints.append(
                f"Low focus (Gini={mean_gini:.2f}). "
                "Model spreads attention uniformly — consider augmentations that "
                "encourage spatial discrimination (e.g. ICD, cutout-style)."
            )
        elif mean_gini > 0.9 and all_ginis:
            hints.append(
                f"Extremely concentrated focus (Gini={mean_gini:.2f}). "
                "Risk of relying on a tiny region. AICD or spatial jitter "
                "may encourage using multiple cues."
            )

        # Edge ratio
        if mean_edge > 0.3 and all_edges:
            hints.append(
                f"High edge-region attention (edge_ratio={mean_edge:.2f}). "
                "Model may be relying on border artifacts — consider random "
                "cropping or padding augmentation."
            )

        # Coherence
        if mean_coh < 0.3 and all_coherences:
            hints.append(
                f"Low spatial coherence ({mean_coh:.2f}). "
                "Saliency is fragmented — noise augmentation or smoothing "
                "transforms may help the model consolidate attention."
            )

        # Per-class critical severity
        if xai_diagnoses:
            critical_classes = [
                cls_id
                for cls_id, diag in xai_diagnoses.items()
                if isinstance(diag, dict) and diag.get("severity") == "critical"
            ]
            if critical_classes:
                cls_list = ", ".join(critical_classes[:5])
                hints.append(
                    f"Critical XAI quality for class(es): {cls_list}. "
                    "Targeted augmentation or data rebalancing may be needed."
                )

        return hints

    def _build_xai_summary(self) -> dict[str, Any]:
        """Build a post-run XAI summary from accumulated batch stats.

        Returns a dict with:
        - ``mean_quality_coverage``: overall mean saliency coverage
        - ``mean_quality_focus``: overall mean gini coefficient (focus)
        - ``quality_trend``: "improving" / "stable" / "declining" / "insufficient_data"
        - ``per_class``: per-class quality breakdown
        - ``recommendations``: list of actionable text hints

        Returns an empty dict when no XAI data is available.
        """
        if not self._prev_xai_batch_stats:
            return {}

        coverages: list[float] = []
        ginis: list[float] = []
        per_class: dict[str, dict[str, float]] = {}

        for cls_id, stats_list in self._prev_xai_batch_stats.items():
            cls_covs = [s.get("coverage", 0.0) for s in stats_list]
            cls_ginis = [s.get("gini", 0.0) for s in stats_list]
            cls_coherence = [s.get("spatial_coherence", 0.0) for s in stats_list]
            cls_edge = [s.get("edge_ratio", 0.0) for s in stats_list]

            coverages.extend(cls_covs)
            ginis.extend(cls_ginis)

            per_class[cls_id] = {
                "coverage": round(float(np.mean(cls_covs)), 4) if cls_covs else 0.0,
                "focus": round(float(np.mean(cls_ginis)), 4) if cls_ginis else 0.0,
                "coherence": round(float(np.mean(cls_coherence)), 4) if cls_coherence else 0.0,
                "edge_ratio": round(float(np.mean(cls_edge)), 4) if cls_edge else 0.0,
            }

        mean_cov = float(np.mean(coverages)) if coverages else 0.0
        mean_gini = float(np.mean(ginis)) if ginis else 0.0

        # Determine quality trend from baseline vs final
        trend = "insufficient_data"
        if self._baseline_xai_stats and self._prev_xai_batch_stats:
            bl_entropies = [
                s.get("entropy", 0.0)
                for stats in self._baseline_xai_stats.values()
                for s in stats
            ]
            cur_entropies = [
                s.get("entropy", 0.0)
                for stats in self._prev_xai_batch_stats.values()
                for s in stats
            ]
            if bl_entropies and cur_entropies:
                bl_mean = float(np.mean(bl_entropies))
                cur_mean = float(np.mean(cur_entropies))
                delta = cur_mean - bl_mean
                if delta < -0.5:
                    trend = "improving"
                elif delta > 0.5:
                    trend = "declining"
                else:
                    trend = "stable"

        return {
            "mean_quality_coverage": round(mean_cov, 4),
            "mean_quality_focus": round(mean_gini, 4),
            "quality_trend": trend,
            "per_class": per_class,
            "recommendations": self._saliency_recommendations(self._prev_xai_batch_stats),
        }

    def _generate_augmentation_hints(
        self,
        xai_diagnoses: dict[str, dict[str, Any]],
        batch_stats: dict[str, list[dict[str, float]]],
        *,
        phase: str = "baseline",
    ) -> list[str]:
        """Derive actionable augmentation hints from XAI diagnoses.

        Uses the shared ``_saliency_recommendations`` helper and adds
        logging / console output.

        Parameters
        ----------
        xai_diagnoses : dict[str, dict]
            Per-class diagnosis dicts produced by ``_generate_xai``.
        batch_stats : dict[str, list[dict]]
            Raw per-class saliency statistics (``_prev_xai_batch_stats``).
        phase : str
            Label for log messages (``"baseline"`` or ``"iteration N"``).

        Returns
        -------
        list[str]
            Human-readable hint strings (empty when no XAI data).
        """
        if not xai_diagnoses and not batch_stats:
            return []

        hints = self._saliency_recommendations(batch_stats, xai_diagnoses)

        # ── Log hints ────────────────────────────────────────────────
        if hints:
            self.logger.info(
                "XAI augmentation hints (%s): %s",
                phase,
                " | ".join(hints),
            )
            if self.config.verbose:
                for hint in hints:
                    print(f"    [XAI hint] {hint}", flush=True)

        return hints

    def _resize_saliency_batch(self, maps: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if maps.ndim != 3:
            return maps
        if maps.shape[1] == target_h and maps.shape[2] == target_w:
            return maps
        resized: list[np.ndarray] = []
        for sal in maps:
            resized_map = cv2.resize(sal.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized.append(resized_map.astype(np.float32))
        return np.stack(resized, axis=0)  # type: ignore[no-any-return]

    def _generate_dual_xai_analysis(self) -> dict[str, Any]:
        if not self._xai_enabled or not self.config.dual_xai_report or not isinstance(self.model, XAICapableModel):
            return {}
        # Detection XAI uses different saliency approach; skip dual analysis
        if self._is_detection:
            return {}

        first_batch = next(iter(self.val_loader))
        if len(first_batch) == 3:
            images, labels, _ = first_batch
        else:
            images, labels = first_batch
        images = images[: self.config.xai_samples]
        labels = labels[: self.config.xai_samples]

        device = next(self.model.get_model().parameters()).device
        images = images.to(device)
        labels = labels.to(device)
        target_layers = self.model.get_target_layers()

        opticam_maps = generate_saliency_maps(
            self.model.get_model(),
            images,
            labels,
            target_layers,
            method="opticam",
        )
        craft_maps = generate_saliency_maps(
            self.model.get_model(),
            images,
            labels,
            target_layers,
            method="craft",
        )
        target_h, target_w = int(images.shape[-2]), int(images.shape[-1])
        opticam_maps = self._resize_saliency_batch(opticam_maps, target_h, target_w)
        craft_maps = self._resize_saliency_batch(craft_maps, target_h, target_w)
        np_images = self._tensor_batch_to_preview_uint8(images.detach().cpu())

        run_dir = getattr(self.reporter, "run_dir", self.config.report_dir)
        opticam_paths = save_xai_visualization(
            np_images,
            opticam_maps,
            save_dir=Path(run_dir) / "assets" / "xai_compare" / "opticam",
            prefix="opticam",
            output_size=self.config.report_xai_size,
        )
        craft_paths = save_xai_visualization(
            np_images,
            craft_maps,
            save_dir=Path(run_dir) / "assets" / "xai_compare" / "craft",
            prefix="craft",
            output_size=self.config.report_xai_size,
        )

        diff = np.abs(opticam_maps - craft_maps)
        return {
            "enabled": True,
            "opticam_paths": [str(path) for path in opticam_paths],
            "craft_paths": [str(path) for path in craft_paths],
            "mean_saliency_opticam": float(np.mean(opticam_maps)),
            "mean_saliency_craft": float(np.mean(craft_maps)),
            "mean_abs_difference": float(np.mean(diff)),
        }

    def _precompute_xai_cache(self) -> XAICache | None:
        # Detection uses bbox-prior ICD (DetectionICD/DetectionAICD) which
        # doesn't need an XAI cache.  Skip entirely for detection tasks.
        if self._is_detection:
            return None

        if not isinstance(self.model, XAICapableModel):
            self._log("Model is not XAICapableModel. Disabling XAI.")
            self._runtime.xai_disabled = True
            return None

        needs_cache = any(isinstance(aug, (ICD, AICD)) for aug in self.augmentations)
        if not needs_cache:
            return None

        cache_dir = self.config.xai_cache_dir or (self.config.checkpoint_dir / "xai_cache")
        cache = XAICache(cache_dir)

        # Resolve n_samples: 0 means "cache all", capped by xai_cache_max_samples
        n_samples = self.config.xai_cache_samples
        if n_samples <= 0:
            dataset_size = len(self.train_loader.dataset)  # type: ignore[arg-type]
            n_samples = min(dataset_size, self.config.xai_cache_max_samples)
            self._log(
                f"XAI cache auto-sized to {n_samples} samples "
                f"(dataset={dataset_size}, cap={self.config.xai_cache_max_samples})"
            )

        written = cache.precompute_cache(
            model=self.model.get_model(),
            train_loader=self.train_loader,
            target_layers=self.model.get_target_layers(),
            n_samples=n_samples,
            method=self.config.xai_method,
            force_recompute=self.config.xai_cache_force_recompute,
            show_progress=self.config.xai_cache_progress and self.config.verbose,
        )
        self._log(f"Precomputed {written} XAI cache maps")

        for aug in self.augmentations:
            if isinstance(aug, (ICD, AICD)):
                aug.cache = cache
        return cache

    def run_single_iteration(
        self,
        augmentation: BaseAugmentation,
        baseline_metrics: dict[str, float] | None = None,
        *,
        iteration: int = 0,
        candidate_idx: int = 0,
        total_candidates: int = 0,
    ) -> tuple[dict[str, float], dict[str, Any], int, bool]:
        """Train one candidate augmentation for m_epochs.

        Returns:
            best_metrics: metrics from the best epoch (per selection_metric).
            best_model_state: deep-copied model state_dict from the best epoch.
            best_epoch: 1-based index of the best epoch.
            pruned: whether the candidate was pruned early.
        """
        active = self._active_augmentations + [augmentation]
        sel_m = self.config.selection_metric
        sel_mode = self.config.selection_mode

        # Track best checkpoint across all epochs
        best_metrics: dict[str, float] = {}
        # Pre-allocate state buffer; updated in-place on each new best.
        best_model_state = self._clone_state_dict(self.model.state_dict())
        best_epoch: int = 0
        best_sel_value: float | None = None

        pruned = False
        for epoch_idx in range(1, self.config.m_epochs + 1):
            train_metrics = self._train_epoch(self.train_loader, augmentations=active)
            epoch_metrics = self._evaluate(self.val_loader)

            # Preserve training loss (eval pops dummy loss for detection).
            if "loss" not in epoch_metrics and "loss" in train_metrics:
                epoch_metrics["loss"] = train_metrics["loss"]

            sel_v = epoch_metrics.get(sel_m, 0)

            # Check if this epoch is the new best
            is_new_best = False
            if best_sel_value is None:
                is_new_best = True
            elif sel_mode == "max" and sel_v > best_sel_value:
                is_new_best = True
            elif sel_mode == "min" and sel_v < best_sel_value:
                is_new_best = True

            if is_new_best:
                best_metrics = epoch_metrics
                self._copy_state_dict_inplace(best_model_state, self.model.state_dict())
                best_epoch = epoch_idx
                best_sel_value = sel_v

            # Step LR scheduler if the adapter supports it
            epoch_end_fn = getattr(self.model, "epoch_end", None)
            if callable(epoch_end_fn):
                epoch_end_fn()

            # Print progress to terminal
            best_marker = " ★" if is_new_best else ""
            print(
                f"    epoch {epoch_idx}/{self.config.m_epochs} "
                f"— {sel_m}={sel_v:.4f}  loss={epoch_metrics.get('loss', 0):.4f}"
                f"  (best: e{best_epoch}={best_sel_value:.4f}){best_marker}",
                flush=True,
            )

            # Emit per-epoch event so the dashboard updates during candidate evaluation
            log_cand_epoch = getattr(self.reporter, "log_candidate_epoch", None)
            if callable(log_cand_epoch):
                log_cand_epoch(
                    iteration=iteration,
                    epoch=epoch_idx,
                    augmentation_name=augmentation.name,
                    metrics=epoch_metrics,
                    is_best=is_new_best,
                )

            self._check_pause()

            # Pruning: use best-so-far metrics (not current epoch)
            # If even the best performance so far is below threshold, prune.
            if (
                baseline_metrics is not None
                and epoch_idx >= self.config.candidate_pruning_warmup_epochs
                and self._should_prune_candidate(best_metrics, baseline_metrics)
            ):
                pruned = True
                print(
                    f"    ✗ Pruned at epoch {epoch_idx} "
                    f"(best-so-far {sel_m}={best_sel_value:.4f} below threshold)",
                    flush=True,
                )
                break
        return best_metrics, best_model_state, best_epoch, pruned

    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        state = self._load_checkpoint(checkpoint_path)
        self.current_iteration = int(state.get("iteration", 0))
        names = state.get("active_augmentations", [])
        self._active_augmentations = [a for a in self.augmentations if a.name in names]
        self._resume_completed_candidates = set(state.get("completed_candidates", []))
        self._resume_iteration_results = cast(dict[str, dict[str, float]], state.get("iteration_results", {}))
        self._resume_baseline_metrics = cast(Optional[dict[str, float]], state.get("baseline_metrics"))

    @staticmethod
    def _count_labels(
        loader: DataLoader,
        *,
        is_detection: bool,
        is_multilabel: bool,
        capture_shape: bool = False,
    ) -> tuple[Counter[int], list[int], int]:
        """Count class labels in *loader* in a single pass.

        Returns ``(counter, image_shape, total_boxes)``.  ``image_shape``
        is only populated when *capture_shape* is True and an image batch
        is found.  ``total_boxes`` is non-zero only for detection.
        """
        counter: Counter[int] = Counter()
        image_shape: list[int] = []
        total_boxes = 0
        for raw_batch in loader:
            if is_detection:
                if len(raw_batch) == 3:
                    images, targets, _ = raw_batch
                else:
                    images, targets = raw_batch
                if capture_shape and not image_shape and images.ndim == 4:
                    image_shape = list(images.shape[1:])
                for target in targets:
                    for label in target["labels"].tolist():
                        counter[int(label)] += 1
                        total_boxes += 1
            elif is_multilabel:
                if len(raw_batch) == 3:
                    images, labels, _ = raw_batch
                else:
                    images, labels = raw_batch
                if capture_shape and not image_shape and images.ndim == 4:
                    image_shape = list(images.shape[1:])
                for sample_labels in labels:
                    for cls_idx in range(sample_labels.shape[0]):
                        if int(sample_labels[cls_idx]) == 1:
                            counter[cls_idx] += 1
            else:
                if len(raw_batch) == 3:
                    images, labels, _ = raw_batch
                else:
                    images, labels = raw_batch
                if capture_shape and not image_shape and images.ndim == 4:
                    image_shape = list(images.shape[1:])
                labels = labels.squeeze()
                if labels.ndim == 0:
                    labels = labels.unsqueeze(0)
                for label in labels.tolist():
                    counter[int(label)] += 1
        return counter, image_shape, total_boxes

    def _compute_dataset_profile(self) -> dict[str, Any]:
        """Compute dataset class distribution, metadata, and data quality.

        Single pass over the training set collects:
        * class distribution & image shape  (near-zero cost)
        * dHash-based near-duplicate detection  (≤ 2 s for 10 k images)
        * image quality flags: NaN/Inf, zero-variance, near-black/white

        A second light pass counts validation labels.
        """
        train_counter, image_shape, total_train_boxes = self._count_labels(
            self.train_loader,
            is_detection=self._is_detection,
            is_multilabel=self._is_multilabel,
            capture_shape=True,
        )
        val_counter, _, total_val_boxes = self._count_labels(
            self.val_loader,
            is_detection=self._is_detection,
            is_multilabel=self._is_multilabel,
        )

        all_classes = sorted(set(train_counter.keys()) | set(val_counter.keys()))
        num_classes = len(all_classes)

        class_distribution = {str(c): train_counter.get(c, 0) for c in all_classes}
        val_class_distribution = {str(c): val_counter.get(c, 0) for c in all_classes}

        total_train = sum(train_counter.values())
        total_val = sum(val_counter.values())

        # Imbalance ratio: max / min (only train)
        counts = list(train_counter.values())
        if counts and min(counts) > 0:
            imbalance_ratio = float(max(counts)) / float(min(counts))
        else:
            imbalance_ratio = float("inf")

        # Build class names from config or class IDs
        if self.config.detection_class_names:
            class_names = list(self.config.detection_class_names)
        else:
            class_names = [f"class_{c}" for c in all_classes]

        profile: dict[str, Any] = {
            "num_classes": num_classes,
            "class_distribution": class_distribution,
            "val_class_distribution": val_class_distribution,
            "total_train_samples": total_train,
            "total_val_samples": total_val,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "image_shape": image_shape,
            "class_names": class_names,
        }

        if self._is_detection:
            profile["task"] = "detection"
            profile["total_train_boxes"] = total_train_boxes
            profile["total_val_boxes"] = total_val_boxes

        # --- Data quality analysis (duplicate detection + sanity checks) ---
        try:
            dq_run_dir = getattr(self.reporter, "run_dir", self.config.report_dir)
            dq_save_dir = dq_run_dir / "artifacts" / "data_quality"
            quality_result = run_data_quality_analysis(
                self.train_loader,
                save_dir=dq_save_dir,
                run_dir=dq_run_dir,
                duplicate_threshold=self.config.duplicate_hamming_threshold,
            )
            profile.update(quality_result)
            dq = quality_result.get("data_quality", {})
            n_warnings = len(dq.get("warnings", []))
            if n_warnings:
                self._log(
                    f"Data quality: {dq.get('summary', '')} "
                    f"({n_warnings} warning type(s))"
                )
            else:
                self._log(f"Data quality: {dq.get('summary', 'OK')}")
        except (ValueError, RuntimeError, OSError, TypeError) as exc:
            self.logger.warning(
                "Data quality analysis failed — skipping: %s",
                exc,
                exc_info=True,
            )

        return profile

    def run(self) -> BNNRRunResult:
        from bnnr.reporting import BNNRRunResult

        set_seed(self.config.seed)
        self.reporter.start(self.config)

        # Emit dataset profile before any training (near-zero cost)
        self._emit_pipeline_phase("dataset_profiling", "started", "Analyzing dataset...")
        dataset_profile = self._compute_dataset_profile()
        log_dataset_profile = getattr(self.reporter, "log_dataset_profile", None)
        if callable(log_dataset_profile):
            log_dataset_profile(dataset_profile)
        self._emit_pipeline_phase("dataset_profiling", "completed")

        self._emit_pipeline_phase("xai_cache", "started", "Precomputing XAI cache...")
        xai_cache = self._precompute_xai_cache()
        if xai_cache is None:
            self._emit_pipeline_phase("xai_cache", "skipped")
        else:
            self._emit_pipeline_phase("xai_cache", "completed")

        baseline_metrics: dict[str, float]
        best_metrics: dict[str, float]
        if self.current_iteration > 0:
            self._log(f"Resuming from iteration {self.current_iteration}")
            baseline_metrics = copy.deepcopy(self._resume_baseline_metrics or self._evaluate(self.val_loader))
            best_metrics = baseline_metrics
            best_path = " -> ".join([aug.name for aug in self._active_augmentations]) or "baseline"
            selected_augmentations = [aug.name for aug in self._active_augmentations]
        else:
            # Baseline phase
            print(
                f"\n{'='*60}\n"
                f"  BASELINE TRAINING ({self.config.m_epochs} epochs)\n"
                f"  Starting...\n"
                f"{'='*60}",
                flush=True,
            )
            for epoch in range(1, self.config.m_epochs + 1):
                train_metrics = self._train_epoch(self.train_loader, augmentations=[])
                val_metrics = self._evaluate(self.val_loader, cache_predictions=True)

                # Step LR scheduler if the adapter supports it
                epoch_end_fn = getattr(self.model, "epoch_end", None)
                if callable(epoch_end_fn):
                    epoch_end_fn()

                sel_m = self.config.selection_metric
                # For detection, loss comes from train_metrics (eval has no loss)
                display_loss = val_metrics.get("loss", train_metrics.get("loss", 0))
                print(
                    f"  baseline epoch {epoch}/{self.config.m_epochs} "
                    f"— {sel_m}={val_metrics.get(sel_m, 0):.4f}  "
                    f"loss={display_loss:.4f}",
                    flush=True,
                )
                per_class_accuracy, confusion = self._compute_eval_class_details()
                cp = self._save_checkpoint(0, "baseline", val_metrics)

                # Generate XAI and previews BEFORE emitting events so
                # dashboard receives complete data in a single flush.
                xai_paths, xai_insights, xai_diagnoses = self._generate_xai(
                    0, "baseline", confusion=confusion,
                )
                preview_pairs = self._generate_augmentation_previews(0, "baseline", [])

                # Inject mean XAI quality score as a trackable metric
                epoch_metrics = {**train_metrics, **val_metrics}
                _xai_q = self._xai_mean_quality(xai_diagnoses)
                if _xai_q is not None:
                    epoch_metrics["xai_quality"] = round(_xai_q, 4)

                # Merge XAI insights into per-class data
                for cls_id, insight_text in xai_insights.items():
                    if cls_id in per_class_accuracy:
                        per_class_accuracy[cls_id]["xai_insight"] = insight_text  # type: ignore[assignment]

                # Emit epoch_end with XAI data included
                log_epoch_metrics = getattr(self.reporter, "log_epoch_metrics", None)
                if callable(log_epoch_metrics):
                    log_epoch_metrics(
                        iteration=0,
                        epoch=epoch,
                        branch="baseline",
                        metrics=epoch_metrics,
                        active_path="baseline",
                        per_class_accuracy=per_class_accuracy,
                        confusion=confusion,
                        xai_insights=xai_insights,
                        xai_diagnoses=xai_diagnoses,
                    )
                self.reporter.log_checkpoint(
                    0,
                    "baseline",
                    epoch,
                    epoch_metrics,
                    cp,
                    xai_paths,
                    preview_pairs=preview_pairs,
                    probe_labels=self._probe_labels(),
                    active_path="baseline",
                    per_class_accuracy=per_class_accuracy,
                    confusion=confusion,
                    xai_insights=xai_insights,
                    xai_diagnoses=xai_diagnoses,
                    emit_epoch_event=False,
                )
                # Emit sample predictions ONCE with all artifacts
                self._emit_probe_prediction_snapshots(
                    iteration=0,
                    epoch=epoch,
                    branch="baseline",
                    preview_pairs=preview_pairs,
                    xai_paths=xai_paths,
                )
                self._check_pause()
            baseline_metrics = self._evaluate(self.val_loader)
            best_metrics = baseline_metrics
            best_path = "baseline"
            selected_augmentations = []
            # Freeze baseline XAI stats for delta-vs-baseline in all future checkpoints
            self._baseline_xai_stats = copy.deepcopy(self._prev_xai_batch_stats)

            sel_m = self.config.selection_metric
            print(
                f"\n  ✓ Baseline complete: {sel_m}={baseline_metrics.get(sel_m, 0):.4f}\n",
                flush=True,
            )

            # Emit XAI-guided augmentation hints after baseline
            if xai_diagnoses and self._prev_xai_batch_stats:
                self._generate_augmentation_hints(
                    xai_diagnoses, self._prev_xai_batch_stats, phase="baseline",
                )

        initial_baseline_metrics = copy.deepcopy(baseline_metrics)
        patience_count = 0
        best_state = self._clone_state_dict(self.model.state_dict())
        current_branch_id = "root:baseline"
        start_iteration = max(1, self.current_iteration or 1)
        resume_iteration = self.current_iteration if self.current_iteration > 0 else None

        for iteration in tqdm(
            range(start_iteration, self.config.max_iterations + 1),
            desc="BNNR iterations",
            disable=not self.config.verbose,
        ):
            self.current_iteration = iteration
            iteration_results: dict[str, dict[str, float]]
            completed_candidates: list[str]

            if resume_iteration is not None and iteration == resume_iteration:
                iteration_results = copy.deepcopy(self._resume_iteration_results)
                completed_candidates = list(self._resume_completed_candidates)
            else:
                iteration_results = {}
                completed_candidates = []

            candidates = [a for a in self.augmentations if a.name not in [x.name for x in self._active_augmentations]]
            if completed_candidates:
                completed_set = set(completed_candidates)
                candidates = [a for a in candidates if a.name not in completed_set]
            if not candidates and not iteration_results:
                self._log("No remaining augmentation candidates; stopping")
                break

            # Store each candidate's best-epoch model state so we can use the winner directly
            candidate_states: dict[str, dict[str, Any]] = {}
            candidate_best_epochs: dict[str, int] = {}

            # ── Optional baseline re-evaluation for this iteration ────
            if self.config.reeval_baseline_per_iteration and iteration > 0:
                print(
                    f"\n  ↻ Baseline re-evaluation (iteration {iteration}, "
                    f"{self.config.m_epochs} epochs, no augmentations)",
                    flush=True,
                )
                saved_state = self._clone_state_dict(self.model.state_dict())
                self.model.load_state_dict(self._clone_state_dict(best_state))

                # Create a no-op augmentation list for the baseline re-evaluation
                from bnnr.augmentations import BasicAugmentation
                noop_aug = BasicAugmentation(probability=1.0)
                reeval_metrics, reeval_state, reeval_epoch, _ = self.run_single_iteration(
                    noop_aug,
                    baseline_metrics=baseline_metrics,
                    iteration=iteration,
                    candidate_idx=0,
                    total_candidates=len(candidates) + 1,
                )

                # Emit as a regular branch so the dashboard picks it up
                reeval_branch_id = f"iter_{iteration}:baseline_reeval"
                _sink = getattr(self.reporter, "_event_sink", None)
                if _sink is not None and hasattr(_sink, "emit"):
                    _sink.emit(
                        "branch_evaluated",
                        {
                            "branch_id": reeval_branch_id,
                            "branch": "baseline_reeval",
                            "augmentation_name": "baseline_reeval",
                            "iteration": iteration,
                            "metrics": reeval_metrics,
                            "epoch": reeval_epoch,
                            "selected": False,
                        },
                    )
                iteration_results["baseline_reeval"] = reeval_metrics
                candidate_states["baseline_reeval"] = reeval_state
                candidate_best_epochs["baseline_reeval"] = reeval_epoch

                # Restore original model state
                self.model.load_state_dict(saved_state)
                print(
                    f"    ✓ Baseline re-eval: "
                    f"{self.config.selection_metric}={reeval_metrics.get(self.config.selection_metric, 0):.4f}",
                    flush=True,
                )

            if candidates:
                sel_m = self.config.selection_metric
                base_val = baseline_metrics.get(sel_m, 0)
                print(
                    f"\n{'='*60}\n"
                    f"  ITERATION {iteration} — Evaluating {len(candidates)} candidates\n"
                    f"  Baseline {sel_m}: {base_val:.4f}\n"
                    f"  Epochs per candidate: {self.config.m_epochs}\n"
                    f"  Selection: best epoch per candidate (not last)\n"
                    f"{'='*60}",
                    flush=True,
                )

                candidate_bar = tqdm(
                    candidates,
                    desc=f"Iteration {iteration} candidates",
                    leave=False,
                    disable=not self.config.verbose,
                )
                per_candidate_durations: list[float] = []
                per_class_by_candidate: dict[str, dict[str, dict[str, float | int]]] = {}
                xai_scores_by_candidate: dict[str, float] = {}
                for idx, augmentation in enumerate(candidate_bar, start=1):
                    t0 = time.perf_counter()
                    print(
                        f"\n  ▶ [{idx}/{len(candidates)}] {augmentation.name} "
                        f"(p={augmentation.probability:.2f})",
                        flush=True,
                    )
                    self.model.load_state_dict(self._clone_state_dict(best_state))
                    cand_best_metrics, cand_best_state, cand_best_epoch, pruned = self.run_single_iteration(
                        augmentation,
                        baseline_metrics=baseline_metrics,
                        iteration=iteration,
                        candidate_idx=idx,
                        total_candidates=len(candidates),
                    )
                    iteration_results[augmentation.name] = cand_best_metrics
                    # Save this candidate's best-epoch model state (already deep-copied)
                    candidate_states[augmentation.name] = cand_best_state
                    candidate_best_epochs[augmentation.name] = cand_best_epoch

                    # Restore best-epoch state to compute per-class details at that point.
                    # Clear cached eval data so _compute_eval_class_details_detection
                    # runs a fresh (single) forward pass with the best-epoch weights.
                    self.model.load_state_dict(cand_best_state)
                    if hasattr(self.model, "last_eval_preds"):
                        self.model.last_eval_preds = []  # type: ignore[attr-defined]  # duck-typed attr on DetectionAdapter
                        self.model.last_eval_targets = []  # type: ignore[attr-defined]  # duck-typed attr on DetectionAdapter
                    # Invalidate classification prediction cache (state changed)
                    self._last_eval_preds = None
                    self._last_eval_labels = None
                    per_class_candidate, confusion_candidate = self._compute_eval_class_details()
                    per_class_by_candidate[augmentation.name] = per_class_candidate

                    # Lightweight XAI probe per candidate (for XAI-aware selection)
                    _, cand_xai_diag, _ = self._generate_xai_lightweight(
                        iteration, augmentation.name, confusion=confusion_candidate,
                    )
                    if cand_xai_diag:
                        avg_q = float(np.mean([
                            d.get("quality_score", 0.0) for d in cand_xai_diag.values()
                        ]))
                        xai_scores_by_candidate[augmentation.name] = avg_q

                    completed_candidates.append(augmentation.name)

                    # Emit real-time events per candidate so dashboard updates live
                    branch_id = f"iter_{iteration}:{augmentation.name}"
                    self.reporter.log_candidate_evaluated(
                        iteration=iteration,
                        branch_id=branch_id,
                        parent_id=current_branch_id,
                        augmentation_name=augmentation.name,
                        metrics=cand_best_metrics,
                        pruned=pruned,
                        per_class=per_class_candidate,
                        confusion=confusion_candidate,
                        best_epoch=cand_best_epoch,
                        candidate_idx=idx,
                        total_candidates=len(candidates),
                    )

                    delta = cand_best_metrics.get(sel_m, 0) - base_val
                    delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                    status = "PRUNED" if pruned else f"Δ{delta_str}"
                    print(
                        f"  ◀ [{idx}/{len(candidates)}] {augmentation.name}: "
                        f"{sel_m}={cand_best_metrics.get(sel_m, 0):.4f} "
                        f"(best@e{cand_best_epoch}, {status})",
                        flush=True,
                    )

                    elapsed = time.perf_counter() - t0
                    per_candidate_durations.append(elapsed)
                    avg_time = sum(per_candidate_durations) / len(per_candidate_durations)
                    remaining = max(len(candidates) - idx, 0)
                    eta = avg_time * remaining
                    candidate_bar.set_postfix_str(f"avg={avg_time:.2f}s eta={eta:.1f}s")

                    if self.config.save_checkpoints:
                        _ = self._save_checkpoint(
                            iteration=iteration,
                            augmentation_name=f"progress_{augmentation.name}",
                            metrics=cand_best_metrics,
                            baseline_metrics=baseline_metrics,
                            completed_candidates=completed_candidates,
                            current_best_metric=self._get_current_best_metric(iteration_results),
                            iteration_results=iteration_results,
                        )

                    self._check_pause()

            selected_name = self._select_best_path(
                iteration_results,
                baseline_metrics,
                xai_scores=xai_scores_by_candidate if candidates else None,
            )
            top_candidate_names = self._top_k_candidate_names(iteration_results, k=3)
            candidate_preview_pairs: dict[str, list[tuple[Path, Path]]] = {}
            aug_by_name = {aug.name: aug for aug in self.augmentations}
            for cand_name in top_candidate_names:
                candidate_aug = aug_by_name.get(cand_name)
                if candidate_aug is None:
                    continue
                preview_pairs = self._generate_augmentation_previews(
                    iteration=iteration,
                    augmentation_name=f"candidate_{cand_name}",
                    augmentations=self._active_augmentations + [candidate_aug],
                )
                candidate_preview_pairs[cand_name] = preview_pairs
            self.reporter.log_iteration_summary(
                iteration,
                iteration_results,
                selected_name or "none",
                baseline_metrics=baseline_metrics,
                top_candidates=top_candidate_names,
                candidate_preview_pairs=candidate_preview_pairs,
                parent_branch_id=current_branch_id,
                metrics_per_class=per_class_by_candidate if candidates else {},
            )

            if selected_name is None:
                patience_count += 1
                print(
                    f"\n  ⚠ No improvement at iteration {iteration} "
                    f"({patience_count}/{self.config.early_stopping_patience})",
                    flush=True,
                )
                if patience_count >= self.config.early_stopping_patience:
                    print("  ⛔ Early stopping triggered", flush=True)
                    break
                continue

            patience_count = 0
            selected_aug = next(a for a in self.augmentations if a.name == selected_name)
            self._active_augmentations.append(selected_aug)
            selected_augmentations.append(selected_name)
            self.best_augmentation = selected_name
            current_branch_id = f"iter_{iteration}:{selected_name}"

            # Use the winner's best-epoch model state directly
            # (already saved from the epoch with the highest selection metric)
            winner_state = candidate_states.get(selected_name)
            winner_best_epoch = candidate_best_epochs.get(selected_name, self.config.m_epochs)
            if winner_state is not None:
                self.model.load_state_dict(winner_state)
                final_metrics = iteration_results[selected_name]
            else:
                # Fallback: retrain (should not happen in normal flow)
                self.model.load_state_dict(self._clone_state_dict(best_state))
                for _ in range(self.config.m_epochs):
                    _ = self._train_epoch(self.train_loader, augmentations=self._active_augmentations)
                final_metrics = self._evaluate(self.val_loader, cache_predictions=True)

            self._copy_state_dict_inplace(best_state, self.model.state_dict())
            best_metrics = final_metrics
            best_path = " -> ".join(selected_augmentations)
            baseline_metrics = final_metrics

            winner_metric = final_metrics.get(self.config.selection_metric, 0)
            base_val = initial_baseline_metrics.get(self.config.selection_metric, 0)
            print(
                f"\n  ★ SELECTED: '{selected_name}' "
                f"(best@epoch {winner_best_epoch}/{self.config.m_epochs}, "
                f"{self.config.selection_metric}={winner_metric:.4f}, "
                f"total gain vs initial baseline: "
                f"{'+' if winner_metric > base_val else ''}{(winner_metric - base_val):.4f})\n"
                f"  Path: {best_path}",
                flush=True,
            )

            cp = self._save_checkpoint(
                iteration,
                selected_name,
                final_metrics,
                baseline_metrics=baseline_metrics,
                completed_candidates=completed_candidates,
                current_best_metric=self._get_current_best_metric(iteration_results),
                iteration_results=iteration_results,
            )
            per_class_accuracy, confusion = self._compute_eval_class_details()

            # Generate XAI and previews BEFORE emitting events
            xai_paths, xai_insights, xai_diagnoses = self._generate_xai(
                iteration, selected_name, confusion=confusion,
            )
            preview_pairs = self._generate_augmentation_previews(
                iteration,
                selected_name,
                self._active_augmentations,
            )

            # Inject mean XAI quality score as a trackable metric
            _xai_q = self._xai_mean_quality(xai_diagnoses)
            if _xai_q is not None:
                final_metrics["xai_quality"] = round(_xai_q, 4)

            # Merge XAI insights into per-class data
            for cls_id, insight_text in xai_insights.items():
                if cls_id in per_class_accuracy:
                    per_class_accuracy[cls_id]["xai_insight"] = insight_text  # type: ignore[assignment]

            # Emit epoch_end with XAI data included
            log_epoch_metrics = getattr(self.reporter, "log_epoch_metrics", None)
            if callable(log_epoch_metrics):
                log_epoch_metrics(
                    iteration=iteration,
                    epoch=self.config.m_epochs,
                    branch=selected_name,
                    metrics=final_metrics,
                    active_path=best_path,
                    per_class_accuracy=per_class_accuracy,
                    confusion=confusion,
                    xai_insights=xai_insights,
                    xai_diagnoses=xai_diagnoses,
                )
            self.reporter.log_checkpoint(
                iteration,
                selected_name,
                self.config.m_epochs,
                final_metrics,
                cp,
                xai_paths,
                preview_pairs=preview_pairs,
                probe_labels=self._probe_labels(),
                active_path=best_path,
                per_class_accuracy=per_class_accuracy,
                confusion=confusion,
                xai_insights=xai_insights,
                xai_diagnoses=xai_diagnoses,
                emit_epoch_event=False,
            )
            # Emit sample predictions ONCE with all artifacts
            self._emit_probe_prediction_snapshots(
                iteration=iteration,
                epoch=self.config.m_epochs,
                branch=selected_name,
                preview_pairs=preview_pairs,
                xai_paths=xai_paths,
            )

            # Emit XAI-guided augmentation hints for this iteration
            if xai_diagnoses and self._prev_xai_batch_stats:
                self._generate_augmentation_hints(
                    xai_diagnoses,
                    self._prev_xai_batch_stats,
                    phase=f"iteration {iteration}",
                )

            # Adaptive ICD/AICD threshold adjustment based on XAI diagnoses
            if self.config.adaptive_icd_threshold and xai_diagnoses:
                self._adapt_icd_thresholds(xai_diagnoses)

        analysis = self._compute_eval_analysis()
        analysis["xai_insights"] = self._build_xai_insights(
            baseline_metrics=initial_baseline_metrics,
            best_metrics=best_metrics,
            selected_augmentations=selected_augmentations,
        )
        xai_summary = self._build_xai_summary()
        if xai_summary:
            analysis["xai_summary"] = xai_summary
            # Print final XAI summary to terminal
            recs = xai_summary.get("recommendations", [])
            if recs:
                trend = xai_summary.get("quality_trend", "unknown")
                cov = xai_summary.get("mean_quality_coverage", 0.0)
                foc = xai_summary.get("mean_quality_focus", 0.0)
                print(
                    f"\n  [XAI SUMMARY] trend={trend}  "
                    f"coverage={cov:.1%}  focus(gini)={foc:.2f}",
                    flush=True,
                )
                for rec in recs:
                    print(f"    → {rec}", flush=True)
        dual_xai_analysis = self._generate_dual_xai_analysis()
        if dual_xai_analysis:
            analysis["dual_xai"] = dual_xai_analysis
        self._emit_pipeline_complete()
        result = self.reporter.finalize(
            best_path=best_path,
            best_metrics=best_metrics,
            selected_augmentations=selected_augmentations,
            analysis=analysis,
        )
        assert isinstance(result, BNNRRunResult)
        return result

    def to_json(self) -> str:
        state = {
            "current_iteration": self.current_iteration,
            "best_augmentation": self.best_augmentation,
            "active_augmentations": [aug.name for aug in self._active_augmentations],
        }
        return json.dumps(state)
