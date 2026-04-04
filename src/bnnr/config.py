"""Configuration loading helpers and XAI preset utilities for BNNR configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from bnnr.core import BNNRConfig
from bnnr.utils import _parse_fbeta


def load_config(config_path: Path) -> BNNRConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {exc}") from exc
    try:
        return BNNRConfig(**(data or {}))
    except ValidationError as exc:
        raise ValueError(f"Invalid BNNRConfig in {config_path}: {exc}") from exc


def save_config(config: BNNRConfig, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


def validate_config(config: BNNRConfig) -> list[str]:
    warnings: list[str] = []
    if config.m_epochs <= 0:
        warnings.append("m_epochs should be > 0")
    if config.max_iterations <= 0:
        warnings.append("max_iterations should be > 0")
    if config.selection_metric not in config.metrics:
        warnings.append("selection_metric is not present in metrics")
    if config.selection_mode not in {"max", "min"}:
        warnings.append("selection_mode should be 'max' or 'min'")
    valid_xai_methods = {"opticam", "gradcam", "craft", "nmf", "nmf_concepts", "real_craft"}
    if config.xai_method not in valid_xai_methods:
        warnings.append(f"xai_method should be one of {sorted(valid_xai_methods)}")
    if config.device not in {"cuda", "cpu", "auto"}:
        warnings.append("device should be cuda/cpu/auto")
    if config.early_stopping_patience < 0:
        warnings.append("early_stopping_patience should be >= 0")
    if config.report_preview_size <= 0:
        warnings.append("report_preview_size should be > 0")
    if config.report_xai_size <= 0:
        warnings.append("report_xai_size should be > 0")
    if config.candidate_pruning_relative_threshold <= 0 or config.candidate_pruning_relative_threshold > 1:
        warnings.append("candidate_pruning_relative_threshold should be in (0, 1]")
    if config.candidate_pruning_warmup_epochs <= 0:
        warnings.append("candidate_pruning_warmup_epochs should be > 0")
    if config.report_probe_images_per_class <= 0:
        warnings.append("report_probe_images_per_class should be > 0")
    if config.report_probe_max_classes <= 0:
        warnings.append("report_probe_max_classes should be > 0")
    if config.event_sample_every_epochs <= 0:
        warnings.append("event_sample_every_epochs should be > 0")
    if config.event_xai_every_epochs <= 0:
        warnings.append("event_xai_every_epochs should be > 0")
    if config.event_min_interval_seconds < 0:
        warnings.append("event_min_interval_seconds should be >= 0")
    # Classification-specific validation
    if config.task == "classification":
        cls_metrics = {
            "accuracy", "f1_macro", "f1_micro", "f1_weighted", "loss",
            "precision", "precision_macro", "precision_micro", "precision_weighted",
            "recall", "recall_macro", "recall_micro", "recall_weighted",
            "cohen_kappa", "mcc", "balanced_accuracy", "hamming",
            "jaccard_macro", "jaccard_micro", "jaccard_weighted",
            "zero_one_loss",
        }
        sm = config.selection_metric
        if sm not in cls_metrics and _parse_fbeta(sm) is None:
            warnings.append(
                f"For task='classification', selection_metric should be one of "
                f"{sorted(cls_metrics)} or 'fbeta_<beta>' (e.g. fbeta_0.5, fbeta_2), "
                f"got '{sm}'"
            )
    # Multilabel-specific validation
    if config.task == "multilabel":
        ml_metrics = {
            "f1_samples", "f1_macro", "f1_micro", "f1_weighted", "accuracy", "loss",
            "hamming", "precision", "precision_macro", "precision_micro", "precision_weighted",
            "recall", "recall_macro", "recall_micro", "recall_weighted",
            "jaccard_samples", "jaccard_macro", "jaccard_micro", "jaccard_weighted",
            "zero_one_loss",
        }
        sm = config.selection_metric
        if sm not in ml_metrics and _parse_fbeta(sm) is None:
            warnings.append(
                f"For task='multilabel', selection_metric should be one of "
                f"{sorted(ml_metrics)} or 'fbeta_<beta>' (e.g. fbeta_0.5, fbeta_2), "
                f"got '{sm}'"
            )
    return warnings


def merge_configs(base_config: BNNRConfig, overrides: dict[str, Any]) -> BNNRConfig:
    merged = base_config.model_dump()
    merged.update(overrides)
    return BNNRConfig(**merged)


# ---------------------------------------------------------------------------
#  XAI-Aware Config Presets
# ---------------------------------------------------------------------------

_XAI_PRESETS: dict[str, dict[str, Any]] = {
    "xai_light": {
        "xai_enabled": True,
        "xai_method": "opticam",
        "dual_xai_report": False,
        "xai_selection_weight": 0.0,
        "xai_pruning_threshold": 0.0,
        "adaptive_icd_threshold": False,
    },
    "xai_full": {
        "xai_enabled": True,
        "xai_method": "opticam",
        "dual_xai_report": True,
        "xai_selection_weight": 0.1,
        "xai_pruning_threshold": 0.15,
        "adaptive_icd_threshold": True,
    },
    "xai_adaptive": {
        "xai_enabled": True,
        "xai_method": "opticam",
        "dual_xai_report": False,
        "xai_selection_weight": 0.15,
        "xai_pruning_threshold": 0.2,
        "adaptive_icd_threshold": True,
    },
}


def list_xai_presets() -> list[str]:
    """Return the names of all available XAI presets."""
    return sorted(_XAI_PRESETS.keys())


def get_xai_preset(name: str) -> dict[str, Any]:
    """Return the override dict for a named XAI preset.

    Raises ``KeyError`` if the preset name is not recognized.
    """
    if name not in _XAI_PRESETS:
        raise KeyError(
            f"Unknown XAI preset '{name}'. "
            f"Available presets: {list_xai_presets()}"
        )
    return dict(_XAI_PRESETS[name])


def apply_xai_preset(config: BNNRConfig, preset: str) -> BNNRConfig:
    """Apply a named XAI preset to *config* and return a new ``BNNRConfig``.

    The preset overrides only XAI-related fields; all other config
    values are preserved.  User overrides supplied after this call
    (via ``merge_configs``) take precedence.

    Available presets:

    * ``"xai_light"`` – XAI enabled with defaults (no influence on
      training decisions).  Good for dashboards and reports.
    * ``"xai_full"`` – All XAI features activated: composite selection
      (10 % weight), XAI-based pruning, adaptive ICD thresholds, and
      dual XAI report.
    * ``"xai_adaptive"`` – XAI actively guides training: higher
      composite selection weight (15 %), stricter pruning, and adaptive
      ICD thresholds.

    Parameters
    ----------
    config : BNNRConfig
        Base configuration to apply the preset on top of.
    preset : str
        Preset name (one of ``list_xai_presets()``).

    Returns
    -------
    BNNRConfig
        New config with XAI fields overridden by the preset.
    """
    overrides = get_xai_preset(preset)
    return merge_configs(config, overrides)
