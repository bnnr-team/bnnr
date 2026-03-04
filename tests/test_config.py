"""Tests for configuration parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bnnr.config import (
    apply_xai_preset,
    get_xai_preset,
    list_xai_presets,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)
from bnnr.core import BNNRConfig


def test_config_roundtrip_yaml(temp_dir: Path) -> None:
    cfg = BNNRConfig(m_epochs=2, max_iterations=3, device="cpu", xai_enabled=False)
    path = temp_dir / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.m_epochs == 2
    assert loaded.max_iterations == 3


def test_validate_config_warnings() -> None:
    cfg = BNNRConfig(m_epochs=1, max_iterations=1, device="cpu", xai_enabled=False)
    warnings = validate_config(cfg)
    assert isinstance(warnings, list)


def test_merge_configs_overrides() -> None:
    cfg = BNNRConfig(device="cpu", xai_enabled=False)
    merged = merge_configs(cfg, {"m_epochs": 9})
    assert merged.m_epochs == 9


def test_load_config_missing_file_raises(temp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = load_config(temp_dir / "missing.yaml")


def test_event_config_defaults() -> None:
    cfg = BNNRConfig()
    assert cfg.event_log_enabled is True
    assert cfg.event_sample_every_epochs == 1
    assert cfg.event_xai_every_epochs == 1
    assert cfg.event_min_interval_seconds == 0.0


def test_event_config_validation() -> None:
    with pytest.raises(ValueError):
        _ = BNNRConfig(event_sample_every_epochs=0)
    with pytest.raises(ValueError):
        _ = BNNRConfig(event_xai_every_epochs=0)
    with pytest.raises(ValueError):
        _ = BNNRConfig(event_min_interval_seconds=-0.1)


# ---------------------------------------------------------------------------
#  XAI Config Presets
# ---------------------------------------------------------------------------


def test_list_xai_presets() -> None:
    presets = list_xai_presets()
    assert isinstance(presets, list)
    assert len(presets) >= 3
    assert "xai_light" in presets
    assert "xai_full" in presets
    assert "xai_adaptive" in presets


def test_get_xai_preset_returns_dict() -> None:
    for name in list_xai_presets():
        preset = get_xai_preset(name)
        assert isinstance(preset, dict)
        assert "xai_enabled" in preset
        assert preset["xai_enabled"] is True


def test_get_xai_preset_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown XAI preset"):
        get_xai_preset("nonexistent_preset")


def test_apply_xai_preset_light() -> None:
    cfg = BNNRConfig(device="cpu", xai_enabled=False)
    result = apply_xai_preset(cfg, "xai_light")
    assert result.xai_enabled is True
    assert result.xai_method == "opticam"
    assert result.xai_selection_weight == 0.0
    assert result.adaptive_icd_threshold is False
    # Non-XAI fields preserved
    assert result.device == "cpu"


def test_apply_xai_preset_full() -> None:
    cfg = BNNRConfig(device="cpu", m_epochs=20)
    result = apply_xai_preset(cfg, "xai_full")
    assert result.xai_enabled is True
    assert result.dual_xai_report is True
    assert result.xai_selection_weight == 0.1
    assert result.xai_pruning_threshold == 0.15
    assert result.adaptive_icd_threshold is True
    # Non-XAI fields preserved
    assert result.device == "cpu"
    assert result.m_epochs == 20


def test_apply_xai_preset_adaptive() -> None:
    cfg = BNNRConfig(device="cpu")
    result = apply_xai_preset(cfg, "xai_adaptive")
    assert result.xai_enabled is True
    assert result.xai_selection_weight == 0.15
    assert result.xai_pruning_threshold == 0.2
    assert result.adaptive_icd_threshold is True


def test_apply_xai_preset_preserves_user_fields() -> None:
    """Applying a preset should not override non-XAI fields."""
    cfg = BNNRConfig(
        device="cpu",
        m_epochs=50,
        max_iterations=20,
        seed=123,
        selection_metric="f1_macro",
    )
    result = apply_xai_preset(cfg, "xai_full")
    assert result.m_epochs == 50
    assert result.max_iterations == 20
    assert result.seed == 123
    assert result.selection_metric == "f1_macro"


def test_apply_xai_preset_can_be_overridden() -> None:
    """User can further override preset values via merge_configs."""
    cfg = BNNRConfig(device="cpu")
    preset_cfg = apply_xai_preset(cfg, "xai_full")
    final = merge_configs(preset_cfg, {"xai_selection_weight": 0.5})
    assert final.xai_selection_weight == 0.5
    # Rest of preset still applied
    assert final.adaptive_icd_threshold is True


def test_apply_xai_preset_returns_new_config() -> None:
    """apply_xai_preset should return a new BNNRConfig, not mutate original."""
    cfg = BNNRConfig(device="cpu", xai_enabled=False)
    result = apply_xai_preset(cfg, "xai_full")
    assert cfg.xai_enabled is False  # original unchanged
    assert result.xai_enabled is True
