"""Extended tests for bnnr.config — validation branches, save_config,
YAML error handling, detection/multilabel validation paths.

Note: BNNRConfig uses pydantic field_validators that prevent invalid
values from being set. ``validate_config()`` is a secondary validation
layer that catches semantic issues (e.g. metric not in metrics list)
that pydantic validators don't cover.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from bnnr.config import (
    load_config,
    merge_configs,
    save_config,
    validate_config,
)
from bnnr.core import BNNRConfig

# ---------------------------------------------------------------------------
# save_config / load_config roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadConfig:
    def test_roundtrip(self, tmp_path):
        cfg = BNNRConfig(m_epochs=7, seed=123, device="cpu")
        path = tmp_path / "cfg.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.m_epochs == 7
        assert loaded.seed == 123

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = BNNRConfig()
        path = tmp_path / "deep" / "nested" / "cfg.yaml"
        save_config(cfg, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("not: valid: yaml: {{{{")
        with pytest.raises(yaml.YAMLError):
            load_config(path)

    def test_load_invalid_config_raises(self, tmp_path):
        path = tmp_path / "invalid.yaml"
        path.write_text("task: invalid_task\n")
        with pytest.raises(ValueError, match="Invalid BNNRConfig"):
            load_config(path)

    def test_load_empty_yaml(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_config(path)
        assert isinstance(cfg, BNNRConfig)


# ---------------------------------------------------------------------------
# validate_config — exercise branches that pydantic doesn't catch
# ---------------------------------------------------------------------------


class TestValidateConfigBranches:
    def test_valid_default_config_no_warnings(self):
        cfg = BNNRConfig()
        warnings = validate_config(cfg)
        assert warnings == []

    def test_selection_metric_not_in_metrics(self):
        cfg = BNNRConfig(selection_metric="f1_macro", metrics=["accuracy"])
        warnings = validate_config(cfg)
        assert any("selection_metric is not present" in w for w in warnings)

    def test_classification_bad_metric(self):
        cfg = BNNRConfig(
            task="classification",
            selection_metric="map_50",
            metrics=["map_50"],
        )
        warnings = validate_config(cfg)
        assert any("classification" in w.lower() for w in warnings)

    def test_classification_fbeta_metric_valid(self):
        cfg = BNNRConfig(
            task="classification",
            selection_metric="fbeta_0.5",
            metrics=["fbeta_0.5"],
        )
        warnings = validate_config(cfg)
        assert not any("classification" in w.lower() for w in warnings)

    def test_classification_precision_valid(self):
        cfg = BNNRConfig(
            task="classification",
            selection_metric="precision_macro",
            metrics=["precision_macro"],
        )
        warnings = validate_config(cfg)
        assert not any("classification" in w.lower() for w in warnings)

    def test_multilabel_bad_metric(self):
        cfg = BNNRConfig(
            task="multilabel",
            selection_metric="map_50",
            metrics=["map_50"],
        )
        warnings = validate_config(cfg)
        assert any("multilabel" in w.lower() for w in warnings)

    def test_multilabel_valid_metric(self):
        cfg = BNNRConfig(
            task="multilabel",
            selection_metric="f1_samples",
            metrics=["f1_samples"],
        )
        warnings = validate_config(cfg)
        assert not any("multilabel" in w.lower() for w in warnings)

    def test_multilabel_fbeta_valid(self):
        cfg = BNNRConfig(
            task="multilabel",
            selection_metric="fbeta_2",
            metrics=["fbeta_2"],
        )
        warnings = validate_config(cfg)
        assert not any("multilabel" in w.lower() for w in warnings)

    def test_detection_valid_metric(self):
        cfg = BNNRConfig(
            task="detection",
            selection_metric="map_50",
            metrics=["map_50"],
        )
        warnings = validate_config(cfg)
        detection_metric_warnings = [w for w in warnings if "detection" in w.lower() and "selection_metric" in w]
        assert len(detection_metric_warnings) == 0


# ---------------------------------------------------------------------------
# Pydantic validators — these SHOULD raise ValidationError
# ---------------------------------------------------------------------------


class TestPydanticFieldValidators:
    """Test that BNNRConfig field validators reject invalid inputs."""

    def test_invalid_task(self):
        with pytest.raises(ValidationError):
            BNNRConfig(task="segmentation")

    def test_invalid_selection_mode(self):
        with pytest.raises(ValidationError):
            BNNRConfig(selection_mode="median")

    def test_invalid_device(self):
        with pytest.raises(ValidationError):
            BNNRConfig(device="tpu")

    def test_invalid_detection_bbox_format(self):
        with pytest.raises(ValidationError):
            BNNRConfig(detection_bbox_format="pascal_voc")

    def test_invalid_detection_targets_mode(self):
        with pytest.raises(ValidationError):
            BNNRConfig(detection_targets_mode="never")

    def test_invalid_report_preview_size(self):
        with pytest.raises(ValidationError):
            BNNRConfig(report_preview_size=0)

    def test_invalid_report_xai_size(self):
        with pytest.raises(ValidationError):
            BNNRConfig(report_xai_size=-5)

    def test_invalid_candidate_pruning_threshold_low(self):
        with pytest.raises(ValidationError):
            BNNRConfig(candidate_pruning_relative_threshold=0.0)

    def test_invalid_candidate_pruning_threshold_high(self):
        with pytest.raises(ValidationError):
            BNNRConfig(candidate_pruning_relative_threshold=1.5)

    def test_invalid_candidate_pruning_warmup(self):
        with pytest.raises(ValidationError):
            BNNRConfig(candidate_pruning_warmup_epochs=0)

    def test_invalid_event_sample_every_epochs(self):
        with pytest.raises(ValidationError):
            BNNRConfig(event_sample_every_epochs=0)

    def test_invalid_event_xai_every_epochs(self):
        with pytest.raises(ValidationError):
            BNNRConfig(event_xai_every_epochs=0)

    def test_invalid_event_min_interval_seconds(self):
        with pytest.raises(ValidationError):
            BNNRConfig(event_min_interval_seconds=-1.0)

    def test_invalid_xai_selection_weight(self):
        with pytest.raises(ValidationError):
            BNNRConfig(xai_selection_weight=1.5)

    def test_invalid_xai_pruning_threshold(self):
        with pytest.raises(ValidationError):
            BNNRConfig(xai_pruning_threshold=-0.1)

    def test_invalid_multilabel_threshold_low(self):
        with pytest.raises(ValidationError):
            BNNRConfig(multilabel_threshold=0.0)

    def test_invalid_multilabel_threshold_high(self):
        with pytest.raises(ValidationError):
            BNNRConfig(multilabel_threshold=1.0)

    def test_invalid_detection_score_threshold(self):
        with pytest.raises(ValidationError):
            BNNRConfig(detection_score_threshold=1.5)

    def test_invalid_detection_nms_threshold(self):
        with pytest.raises(ValidationError):
            BNNRConfig(detection_nms_threshold=-0.1)

    def test_invalid_detection_min_box_area(self):
        with pytest.raises(ValidationError):
            BNNRConfig(detection_min_box_area=-1.0)

    def test_invalid_report_probe_controls(self):
        with pytest.raises(ValidationError):
            BNNRConfig(report_probe_images_per_class=0)
        with pytest.raises(ValidationError):
            BNNRConfig(report_probe_max_classes=0)

    def test_invalid_detection_xai_grid_size(self):
        with pytest.raises(ValidationError):
            BNNRConfig(detection_xai_grid_size=0)


# ---------------------------------------------------------------------------
# merge_configs
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_override_single_field(self):
        base = BNNRConfig(m_epochs=5)
        merged = merge_configs(base, {"m_epochs": 10})
        assert merged.m_epochs == 10

    def test_base_preserved(self):
        base = BNNRConfig(seed=999, m_epochs=5)
        merged = merge_configs(base, {"m_epochs": 10})
        assert merged.seed == 999
