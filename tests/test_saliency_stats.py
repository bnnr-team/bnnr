"""Tests for bnnr.analysis.saliency_stats (FIX-1-1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from bnnr.analysis.saliency_stats import (
    DEFAULT_RESOLUTION,
    PERTURBATION_FILL,
    SaliencyStats,
    aggregate_saliency_stats,
    perturbation_shift,
    saliency_stats_from_map,
)


def _delta_map(size: int = 14) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.float32)
    m[size // 2, size // 2] = 1.0
    return m


def _uniform_map(size: int = 14) -> np.ndarray:
    return np.full((size, size), 0.5, dtype=np.float32)


def _border_map(size: int = 14) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.float32)
    strip = max(1, int(size * 0.15))
    m[:] = 0.0
    m[:strip, :] = 1.0
    m[-strip:, :] = 1.0
    m[:, :strip] = 1.0
    m[:, -strip:] = 1.0
    return m


class TestPureStatistics:
    def test_delta_map_is_maximally_concentrated(self) -> None:
        stats = saliency_stats_from_map(_delta_map())
        assert stats.concentration > 0.95
        assert stats.gini > 0.95

    def test_uniform_map_is_unconcentrated(self) -> None:
        stats = saliency_stats_from_map(_uniform_map())
        assert stats.concentration < 0.05
        assert stats.gini < 0.05

    def test_border_map_has_high_border_mass(self) -> None:
        stats = saliency_stats_from_map(_border_map())
        assert stats.border_mass > 0.9

    def test_centre_blob_has_low_border_mass(self) -> None:
        stats = saliency_stats_from_map(_delta_map())
        assert stats.border_mass == pytest.approx(0.0, abs=1e-6)

    def test_all_zero_map_does_not_divide_by_zero(self) -> None:
        stats = saliency_stats_from_map(np.zeros((14, 14), dtype=np.float32))
        assert stats.concentration == 0.0
        assert stats.gini == 0.0
        assert stats.border_mass == 0.0

    def test_non_2d_map_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            saliency_stats_from_map(np.zeros((3, 14, 14), dtype=np.float32))


class TestResolutionIsRecordedAndApplied:
    def test_resolution_recorded_on_result(self) -> None:
        stats = saliency_stats_from_map(_delta_map(7), resolution=(14, 14))
        assert stats.resolution == (14, 14)

    def test_default_resolution_used_when_unspecified(self) -> None:
        stats = saliency_stats_from_map(_delta_map(7))
        assert stats.resolution == DEFAULT_RESOLUTION

    def test_native_and_upsampled_maps_land_on_the_same_grid(self) -> None:
        """The whole point of declaring a resolution: 7x7 and its 224x224
        upsampling must not produce two different concentrations."""
        native = _delta_map(7)
        upsampled = np.asarray(
            np.kron(native, np.ones((32, 32), dtype=np.float32)), dtype=np.float32
        )
        a = saliency_stats_from_map(native, resolution=(7, 7))
        b = saliency_stats_from_map(upsampled, resolution=(7, 7))
        assert a.concentration == pytest.approx(b.concentration, abs=0.02)

    def test_to_dict_carries_resolution(self) -> None:
        d = saliency_stats_from_map(_delta_map()).to_dict()
        assert d["resolution"] == DEFAULT_RESOLUTION
        assert d["perturbation_fill"] is None


class TestAggregation:
    def test_median_reduction_ignores_one_broken_map(self) -> None:
        good = [saliency_stats_from_map(_delta_map()) for _ in range(5)]
        broken = saliency_stats_from_map(_uniform_map())
        agg = aggregate_saliency_stats([*good, broken])
        assert agg.concentration > 0.95
        assert agg.n_maps == 6

    def test_mixed_resolutions_rejected(self) -> None:
        a = saliency_stats_from_map(_delta_map(), resolution=(7, 7))
        b = saliency_stats_from_map(_delta_map(), resolution=(14, 14))
        with pytest.raises(ValueError, match="different resolutions"):
            aggregate_saliency_stats([a, b])

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aggregate_saliency_stats([])

    def test_shifts_recorded_with_their_fill(self) -> None:
        stats = [saliency_stats_from_map(_delta_map()) for _ in range(3)]
        agg = aggregate_saliency_stats(stats, shifts=[0.1, 0.2, 0.9])
        assert agg.perturbation_shift == pytest.approx(0.2)
        assert agg.perturbation_fill == PERTURBATION_FILL

    def test_no_shifts_leaves_both_fields_none(self) -> None:
        agg = aggregate_saliency_stats([saliency_stats_from_map(_delta_map())])
        assert agg.perturbation_shift is None
        assert agg.perturbation_fill is None


class _StaticExplainer:
    """Explainer that returns a fixed map regardless of the input."""

    name = "static"

    def __init__(self, maps: np.ndarray) -> None:
        self.maps = maps
        self.calls = 0

    def explain(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
    ) -> np.ndarray:
        self.calls += 1
        return self.maps


class _FlippingExplainer(_StaticExplainer):
    """Explainer whose map reverses on the second call."""

    def explain(
        self,
        model: nn.Module,
        images: Tensor,
        labels: Tensor,
        target_layers: list[nn.Module],
    ) -> np.ndarray:
        self.calls += 1
        if self.calls == 1:
            return self.maps
        return self.maps[:, ::-1, ::-1].copy()


def _gradient_maps(batch: int = 2, size: int = 14) -> np.ndarray:
    single = np.linspace(0.0, 1.0, size * size, dtype=np.float32).reshape(size, size)
    return np.stack([single] * batch)


class TestPerturbationShift:
    def _inputs(self, batch: int = 2) -> tuple[nn.Module, Tensor, Tensor, list[nn.Module]]:
        model = nn.Identity()
        images = torch.rand(batch, 3, 28, 28)
        labels = torch.zeros(batch, dtype=torch.long)
        return model, images, labels, []

    def test_stable_explanation_gives_near_zero_shift(self) -> None:
        model, images, labels, layers = self._inputs()
        explainer = _StaticExplainer(_gradient_maps())
        shifts, maps = perturbation_shift(model, explainer, images, labels, layers)
        assert len(shifts) == 2
        assert all(s == pytest.approx(0.0, abs=1e-6) for s in shifts)
        assert maps.shape == (2, 14, 14)

    def test_reversed_explanation_gives_shift_near_two(self) -> None:
        model, images, labels, layers = self._inputs()
        explainer = _FlippingExplainer(_gradient_maps())
        shifts, _ = perturbation_shift(model, explainer, images, labels, layers)
        assert all(s > 1.9 for s in shifts)

    def test_constant_map_reports_full_shift(self) -> None:
        """rho is undefined on a constant map; the convention is shift 1.0."""
        model, images, labels, layers = self._inputs()
        flat = np.ones((2, 14, 14), dtype=np.float32)
        shifts, _ = perturbation_shift(model, _StaticExplainer(flat), images, labels, layers)
        assert all(s == pytest.approx(1.0) for s in shifts)

    def test_explainer_called_exactly_twice(self) -> None:
        model, images, labels, layers = self._inputs()
        explainer = _StaticExplainer(_gradient_maps())
        perturbation_shift(model, explainer, images, labels, layers)
        assert explainer.calls == 2

    def test_perturbation_keeps_top_k_pixels_untouched(self) -> None:
        model, images, labels, layers = self._inputs(batch=1)
        # Strictly decreasing over the flat index, so "the top 10%" is exactly
        # the first 78 pixels with no tie to break arbitrarily.
        maps = np.linspace(1.0, 0.0, 28 * 28, dtype=np.float32).reshape(1, 28, 28)

        captured: list[Tensor] = []

        class _Capture(_StaticExplainer):
            def explain(
                self,
                model: nn.Module,
                images: Tensor,
                labels: Tensor,
                target_layers: list[nn.Module],
            ) -> np.ndarray:
                captured.append(images.clone())
                return self.maps

        perturbation_shift(model, _Capture(maps), images, labels, layers, top_k=0.1)
        original, perturbed = captured
        # Kept region is bit-identical, the rest was blurred.
        assert torch.equal(original[:, :, :2, :], perturbed[:, :, :2, :])
        assert not torch.equal(original[:, :, 10:, :], perturbed[:, :, 10:, :])

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_top_k_rejected(self, bad: float) -> None:
        model, images, labels, layers = self._inputs()
        with pytest.raises(ValueError, match="top_k"):
            perturbation_shift(
                model, _StaticExplainer(_gradient_maps()), images, labels, layers, top_k=bad
            )


class TestDataclass:
    def test_stats_are_frozen(self) -> None:
        stats = saliency_stats_from_map(_delta_map())
        with pytest.raises(Exception):
            stats.concentration = 0.0  # type: ignore[misc]

    def test_single_map_counts_as_one(self) -> None:
        assert saliency_stats_from_map(_delta_map()).n_maps == 1

    def test_is_a_saliency_stats(self) -> None:
        assert isinstance(saliency_stats_from_map(_delta_map()), SaliencyStats)
