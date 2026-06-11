"""Pydantic configuration model for BNNR training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BNNRConfig(BaseModel):
    """Immutable runtime configuration for a full BNNR training run.

    Defines training budget, metrics/selection policy, reporting paths,
    XAI behavior, and task-specific options for classification, detection,
    and multilabel workflows.
    """
    # extra="forbid" turns YAML typos (e.g. "m_epoch: 50") into an immediate
    # ValidationError naming the unknown key instead of silently training with
    # the default value.
    model_config = ConfigDict(frozen=True, extra="forbid")

    m_epochs: int = Field(default=5, ge=1)
    # max_iterations=0 is valid: it runs the baseline phase only (no search).
    max_iterations: int = Field(default=10, ge=0)
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
    device: str = "auto"
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
