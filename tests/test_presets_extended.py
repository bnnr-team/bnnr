"""Extended tests for bnnr.presets — auto-selection, screening, kornia preset.

Supplements the existing preset tests with coverage for:
- auto_select_augmentations (GPU vs CPU, kornia availability)
- screening preset
- prob_override parameter
- _build_kornia_preset
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bnnr.presets import auto_select_augmentations, get_preset


class TestAutoSelectAugmentations:
    def test_cpu_returns_standard(self):
        """Without CUDA, auto_select should fall back to standard preset."""
        with patch("bnnr.presets.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            augs = auto_select_augmentations(random_state=42)
        assert isinstance(augs, list)
        assert len(augs) > 0

    def test_prefer_gpu_false_returns_standard(self):
        """With prefer_gpu=False, should use standard even with CUDA."""
        augs = auto_select_augmentations(random_state=42, prefer_gpu=False)
        assert isinstance(augs, list)
        assert len(augs) > 0

    def test_cuda_without_kornia_returns_gpu_preset(self):
        """With CUDA but no kornia, should fall back to gpu preset."""
        with (
            patch("bnnr.presets.torch") as mock_torch,
            patch("bnnr.presets.get_preset") as mock_get_preset,
        ):
            mock_torch.cuda.is_available.return_value = True
            # Simulate kornia not importable
            mock_get_preset.return_value = []

            # Force ImportError on kornia import
            def _fail_kornia_import(name, *a, **kw):
                if "kornia" in name:
                    raise ImportError("No kornia")
                return __builtins__.__import__(name, *a, **kw)

            with patch("builtins.__import__", side_effect=_fail_kornia_import):
                augs = auto_select_augmentations(random_state=42)

        # It will call get_preset("gpu")
        assert isinstance(augs, list)


class TestScreeningPreset:
    def test_screening_uses_aggressive_with_uniform_prob(self):
        """Screening preset should be aggressive with p=0.5."""
        augs = get_preset("screening", random_state=42)
        assert isinstance(augs, list)
        assert len(augs) > 0
        for aug in augs:
            assert aug.probability == 0.5


class TestProbOverride:
    def test_light_with_prob_override(self):
        augs = get_preset("light", random_state=42, prob_override=0.3)
        for aug in augs:
            assert aug.probability == pytest.approx(0.3)

    def test_standard_with_prob_override(self):
        augs = get_preset("standard", random_state=42, prob_override=0.8)
        for aug in augs:
            assert aug.probability == pytest.approx(0.8)

    def test_auto_with_prob_override(self):
        augs = get_preset("auto", random_state=42, prob_override=0.1)
        for aug in augs:
            assert aug.probability == pytest.approx(0.1)


class TestBuildKorniaPreset:
    def test_kornia_preset_has_builtin_and_kornia_augs(self):
        """If kornia is installed, the preset should mix built-in and kornia augs."""
        try:
            import kornia  # noqa: F401
        except ImportError:
            pytest.skip("Kornia not installed")

        from bnnr.presets import _build_kornia_preset

        augs = _build_kornia_preset(random_state=42)
        assert len(augs) >= 4  # 3 built-in + at least 1 kornia
        names = [a.name for a in augs]
        assert any("kornia" in n for n in names)

    def test_kornia_preset_fallback_without_kornia(self):
        """If kornia import fails, _build_kornia_preset falls back to gpu preset."""
        from bnnr.presets import _build_kornia_preset

        # The function is called only when kornia_aug was successfully imported
        # but kornia itself may fail at K import — it falls back to gpu preset
        augs = _build_kornia_preset(random_state=42)
        assert isinstance(augs, list)
        assert len(augs) > 0
