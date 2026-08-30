"""Pydantic configuration model for BNNR training runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from bnnr.analysis.diagnosis import DiagnosisThresholds

#: Selectors that cannot run without calibrated thresholds. A diagnosis-driven
#: search policy (#413) joins this set rather than growing a second check.
_DIAGNOSIS_DRIVEN_SELECTORS = frozenset({"diagnosis"})


class DiagnosisConfig(BaseModel):
    """Cut points for the attention diagnosis. Every one starts unset.

    There is deliberately no numeric default anywhere in this model. Shipping
    one would repeat exactly the mistake that produced ``xai_selection_weight``
    and its preset values of 0.1 and 0.15: a number nobody measured, driving
    selection for every user. Calibration is a separate pre-registered study,
    and until it reports, requesting the ``diagnosis`` selector fails at config
    construction rather than at some point mid-run.

    **Shadow mode needs none of these.** It records the raw saliency statistics
    rather than a regime, so it collects calibration samples from runs that
    were going to happen anyway, at no extra GPU cost, before any threshold
    exists.

    ``hard_quantile_q`` is deliberately *not* here. It lives on
    :class:`BNNRConfig` because the metrics it produces, ``hard_quantile_acc``
    and ``robustness_gap``, are worth watching with no diagnosis configured at
    all. The calibration study sweeps it alongside these, which does not make
    it a diagnosis field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    concentration_lo: Optional[float] = None  # noqa: UP045
    concentration_hi: Optional[float] = None  # noqa: UP045
    border_mass_hi: Optional[float] = None  # noqa: UP045
    perturbation_shift_hi: Optional[float] = None  # noqa: UP045
    robustness_gap_hi: Optional[float] = None  # noqa: UP045
    min_confidence: Optional[float] = None  # noqa: UP045

    def to_thresholds(self) -> DiagnosisThresholds:
        """Convert to the :class:`~bnnr.analysis.diagnosis.DiagnosisThresholds`
        the rule consumes. Imported lazily to keep config import-light."""
        from bnnr.analysis.diagnosis import DiagnosisThresholds

        return DiagnosisThresholds(
            concentration_lo=self.concentration_lo,
            concentration_hi=self.concentration_hi,
            border_mass_hi=self.border_mass_hi,
            perturbation_shift_hi=self.perturbation_shift_hi,
            robustness_gap_hi=self.robustness_gap_hi,
            min_confidence=self.min_confidence,
        )

    def missing(self) -> tuple[str, ...]:
        """Required thresholds still unset, in declaration order."""
        return self.to_thresholds().missing()

    @field_validator("min_confidence")
    @classmethod
    def validate_min_confidence(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        return value


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
    # Which rule picks the winning candidate. "metric_argmax" is what BNNR has
    # always done and stays the default; see bnnr.training.selection.SELECTORS.
    selector: str = "metric_argmax"
    #: Cut points for the ``diagnosis`` selector. Unset by design; see
    #: DiagnosisConfig and docs/diagnosis.md.
    diagnosis: DiagnosisConfig = Field(default_factory=DiagnosisConfig)

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
    # Disk cap for the on-disk XAI cache, in megabytes. After precompute the
    # cache is trimmed LRU-by-mtime to stay under this. 0 disables the cap.
    xai_cache_max_mb: int = Field(default=2048, ge=0)
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

    # ── Hard-quantile robustness proxy (FIX-1-3) ──
    # Fraction of the validation set treated as "hard", by loss. The diagnosis
    # reads robustness_gap; q is swept by the threshold calibration study, so it
    # is a knob rather than a constant.
    hard_quantile_q: float = 0.2

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

    @field_validator("hard_quantile_q")
    @classmethod
    def validate_hard_quantile_q(cls, value: float) -> float:
        if value <= 0.0 or value > 1.0:
            raise ValueError("hard_quantile_q must be in (0, 1]")
        return value

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

    @model_validator(mode="after")
    def validate_diagnosis_is_calibrated(self) -> BNNRConfig:
        """Refuse a diagnosis-driven run whose thresholds were never measured.

        This fires at config construction, so the failure lands before any GPU
        time is spent rather than at the first selection round. A run that got
        several epochs in before discovering it cannot decide anything is the
        worst version of this error.
        """
        if self.selector in _DIAGNOSIS_DRIVEN_SELECTORS:
            absent = self.diagnosis.missing()
            if absent:
                raise ValueError(
                    f"selector={self.selector!r} needs calibrated diagnosis thresholds; "
                    f"{', '.join(absent)} {'is' if len(absent) == 1 else 'are'} unset. "
                    f"There is deliberately no default: an uncalibrated cut point driving "
                    f"selection is the defect this replaces. Supply them under the "
                    f"'diagnosis:' key, or load a profile with "
                    f"bnnr.config.load_diagnosis_profile(). See docs/diagnosis.md."
                )
        return self

    @field_validator("selector")
    @classmethod
    def validate_selector(cls, value: str) -> str:
        # Imported here rather than at module scope: selection imports the
        # config for typing only, but a top-level import would still be a cycle
        # waiting for the first runtime import either side adds.
        from bnnr.training.selection import SELECTORS

        if value not in SELECTORS:
            raise ValueError(f"selector must be one of {sorted(SELECTORS)}, got {value!r}")
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
