"""Tests for diagnosis thresholds as config (FIX-1-4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bnnr.config import load_diagnosis_profile
from bnnr.config_model import BNNRConfig, DiagnosisConfig

CALIBRATED = {
    "concentration_lo": 0.31,
    "concentration_hi": 0.58,
    "border_mass_hi": 0.34,
    "perturbation_shift_hi": 0.47,
    "robustness_gap_hi": 0.12,
    "min_confidence": 0.75,
}


class TestNoDefaults:
    @pytest.mark.parametrize("field", sorted(DiagnosisConfig.model_fields))
    def test_every_threshold_starts_unset(self, field: str) -> None:
        """The whole discipline of the exercise: no number nobody measured."""
        assert getattr(DiagnosisConfig(), field) is None

    def test_a_blank_config_reports_every_required_field_missing(self) -> None:
        missing = DiagnosisConfig().missing()
        assert set(missing) == {
            "concentration_lo",
            "concentration_hi",
            "border_mass_hi",
            "perturbation_shift_hi",
            "robustness_gap_hi",
        }

    def test_min_confidence_is_not_counted_as_missing(self) -> None:
        """A caller may want the regime without a policy for acting on it."""
        partial = DiagnosisConfig(**{k: v for k, v in CALIBRATED.items() if k != "min_confidence"})
        assert partial.missing() == ()

    def test_default_config_carries_an_empty_diagnosis_block(self) -> None:
        assert BNNRConfig().diagnosis == DiagnosisConfig()


class TestTheGate:
    def test_diagnosis_selector_without_thresholds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="calibrated diagnosis thresholds"):
            BNNRConfig(selector="diagnosis")

    def test_the_error_names_the_missing_fields(self) -> None:
        partial = {"concentration_lo": 0.3, "concentration_hi": 0.6}
        with pytest.raises(ValueError) as excinfo:
            BNNRConfig(selector="diagnosis", diagnosis=partial)
        message = str(excinfo.value)
        assert "border_mass_hi" in message
        assert "robustness_gap_hi" in message
        assert "concentration_lo" not in message.split("unset")[0]

    def test_the_error_points_at_the_doc(self) -> None:
        with pytest.raises(ValueError, match="docs/diagnosis.md"):
            BNNRConfig(selector="diagnosis")

    def test_the_error_names_the_profile_loader(self) -> None:
        with pytest.raises(ValueError, match="load_diagnosis_profile"):
            BNNRConfig(selector="diagnosis")

    def test_fully_calibrated_thresholds_are_accepted(self) -> None:
        config = BNNRConfig(selector="diagnosis", diagnosis=CALIBRATED)
        assert config.selector == "diagnosis"
        assert config.diagnosis.concentration_lo == pytest.approx(0.31)

    def test_other_selectors_do_not_need_thresholds(self) -> None:
        """Only a diagnosis-driven run is gated."""
        for selector in ("metric_argmax", "random"):
            assert BNNRConfig(selector=selector).selector == selector

    def test_thresholds_without_the_selector_are_harmless(self) -> None:
        """Shadow mode records statistics with the selector left alone."""
        config = BNNRConfig(diagnosis=CALIBRATED)
        assert config.selector == "metric_argmax"
        assert config.diagnosis.missing() == ()


class TestConversion:
    def test_to_thresholds_round_trips_every_field(self) -> None:
        thresholds = DiagnosisConfig(**CALIBRATED).to_thresholds()
        for name, value in CALIBRATED.items():
            assert getattr(thresholds, name) == pytest.approx(value)

    def test_converted_thresholds_satisfy_the_rule(self) -> None:
        DiagnosisConfig(**CALIBRATED).to_thresholds().require()  # must not raise

    def test_min_confidence_range_is_validated(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            DiagnosisConfig(min_confidence=1.5)


class TestProfileLoading:
    def _write(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    def test_loads_a_named_profile(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"imagewoof_resnet50": CALIBRATED, "other": CALIBRATED})
        loaded = load_diagnosis_profile(path, "imagewoof_resnet50")
        assert loaded.concentration_lo == pytest.approx(0.31)

    def test_a_single_profile_needs_no_name(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"only_one": CALIBRATED})
        assert load_diagnosis_profile(path).min_confidence == pytest.approx(0.75)

    def test_several_profiles_require_an_explicit_name(self, tmp_path: Path) -> None:
        """Picking one silently would be the same mistake as a default."""
        path = self._write(tmp_path, {"a": CALIBRATED, "b": CALIBRATED})
        with pytest.raises(ValueError, match="name the one you want"):
            load_diagnosis_profile(path)

    def test_unknown_name_lists_what_is_available(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"a": CALIBRATED, "b": CALIBRATED})
        with pytest.raises(KeyError, match="Available"):
            load_diagnosis_profile(path, "c")

    def test_missing_file_is_reported_as_such(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Diagnosis profile not found"):
            load_diagnosis_profile(tmp_path / "nope.yaml")

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="no named threshold profiles"):
            load_diagnosis_profile(path)

    def test_non_mapping_profile_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not a mapping"):
            load_diagnosis_profile(path, "a")

    def test_unknown_key_in_a_profile_rejected(self, tmp_path: Path) -> None:
        """extra="forbid" turns a typo into an error naming the key."""
        path = self._write(tmp_path, {"a": {**CALIBRATED, "concentraton_lo": 0.3}})
        with pytest.raises(ValueError, match="Invalid diagnosis profile"):
            load_diagnosis_profile(path, "a")

    def test_a_partial_profile_loads(self, tmp_path: Path) -> None:
        """Loading is not the completeness check; using it for selection is."""
        path = self._write(tmp_path, {"a": {"concentration_lo": 0.3}})
        loaded = load_diagnosis_profile(path, "a")
        assert loaded.concentration_lo == pytest.approx(0.3)
        assert loaded.missing()

    def test_a_loaded_profile_satisfies_the_gate(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"a": CALIBRATED})
        config = BNNRConfig(selector="diagnosis", diagnosis=load_diagnosis_profile(path, "a"))
        assert config.selector == "diagnosis"


class TestYamlRoundTrip:
    def test_diagnosis_survives_save_and_load(self, tmp_path: Path) -> None:
        from bnnr.config import load_config, save_config

        original = BNNRConfig(selector="diagnosis", diagnosis=CALIBRATED)
        path = tmp_path / "config.yaml"
        save_config(original, path)
        assert load_config(path).diagnosis == original.diagnosis

    def test_a_yaml_config_asking_for_diagnosis_without_thresholds_fails(
        self, tmp_path: Path
    ) -> None:
        from bnnr.config import load_config

        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"selector": "diagnosis"}))
        with pytest.raises(ValueError, match="calibrated diagnosis thresholds"):
            load_config(path)
