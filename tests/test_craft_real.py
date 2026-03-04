"""Tests for the Real CRAFT implementation (gradient-based sensitivity)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from bnnr.craft import NMFConceptExplainer, RealCRAFTExplainer, RecursiveCRAFTExplainer


class _SimpleCNN(nn.Module):
    """Minimal CNN for testing CRAFT."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@pytest.fixture
def model() -> _SimpleCNN:
    m = _SimpleCNN(num_classes=5)
    m.eval()
    return m


@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.rand(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 3])
    return images, labels


class TestRealCRAFTExplainer:
    def test_explain_output_shape(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        craft = RealCRAFTExplainer(n_concepts=5, sensitivity_method="gradient")
        target_layers = [model.features[-2]]  # second conv layer
        saliency = craft.explain(model, images, labels, target_layers)
        # Shape should be [B, H_act, W_act]
        assert saliency.ndim == 3
        assert saliency.shape[0] == 4

    def test_explain_output_range(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        craft = RealCRAFTExplainer(n_concepts=5)
        target_layers = [model.features[-2]]
        saliency = craft.explain(model, images, labels, target_layers)
        assert float(saliency.min()) >= 0.0
        assert float(saliency.max()) <= 1.0 + 1e-6

    def test_concept_importance_keys(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        craft = RealCRAFTExplainer(n_concepts=5)
        target_layers = [model.features[-2]]
        importance = craft.get_concept_importance(model, images, labels, target_layers)
        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert all(isinstance(k, int) for k in importance)
        assert all(isinstance(v, float) for v in importance.values())

    def test_gradient_sensitivity_nonzero(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        craft = RealCRAFTExplainer(n_concepts=5, sensitivity_method="gradient")
        target_layers = [model.features[-2]]
        craft.explain(model, images, labels, target_layers)
        sensitivity = craft.get_concept_sensitivity()
        assert sensitivity is not None
        # At least some concepts should have non-zero sensitivity
        assert float(sensitivity.sum()) > 0

    def test_deletion_sensitivity(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        craft = RealCRAFTExplainer(n_concepts=5, sensitivity_method="deletion")
        target_layers = [model.features[-2]]
        saliency = craft.explain(model, images, labels, target_layers)
        assert saliency.ndim == 3
        assert saliency.shape[0] == 4

    def test_concept_basis_shape(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        craft = RealCRAFTExplainer(n_concepts=5)
        target_layers = [model.features[-2]]
        craft.explain(model, images, labels, target_layers)
        basis = craft.get_concept_basis()
        assert basis is not None
        assert basis.shape[0] == 5  # n_concepts
        assert basis.shape[1] == 16  # channels from conv layer

    def test_reproducibility(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        target_layers = [model.features[-2]]

        craft1 = RealCRAFTExplainer(n_concepts=5, sensitivity_method="gradient")
        saliency1 = craft1.explain(model, images, labels, target_layers)

        craft2 = RealCRAFTExplainer(n_concepts=5, sensitivity_method="gradient")
        saliency2 = craft2.explain(model, images, labels, target_layers)

        # NMF has random_state=42 internally, so results should be identical
        np.testing.assert_array_almost_equal(saliency1, saliency2, decimal=4)

    def test_invalid_sensitivity_method(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        craft = RealCRAFTExplainer(n_concepts=5, sensitivity_method="invalid")
        target_layers = [model.features[-2]]
        with pytest.raises(ValueError, match="Unknown sensitivity method"):
            craft.explain(model, images, labels, target_layers)


class TestRecursiveCRAFTExplainer:
    def test_recursive_multi_layer(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        recursive = RecursiveCRAFTExplainer(n_concepts=4, layer_names=["conv1", "conv2"])
        # Two layer groups
        layers_list = [[model.features[0]], [model.features[2]]]
        saliency_maps = recursive.explain_recursive(model, images, labels, layers_list)
        assert len(saliency_maps) == 2
        for sal in saliency_maps:
            assert sal.ndim == 3
            assert sal.shape[0] == 4

    def test_layer_results_metadata(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        recursive = RecursiveCRAFTExplainer(n_concepts=4, layer_names=["conv1", "conv2"])
        layers_list = [[model.features[0]], [model.features[2]]]
        recursive.explain_recursive(model, images, labels, layers_list)
        results = recursive.get_layer_results()
        assert len(results) == 2
        assert results[0]["layer_name"] == "conv1"
        assert results[1]["layer_name"] == "conv2"
        for r in results:
            assert "importance" in r
            assert "basis" in r
            assert "sensitivity" in r


class TestNMFConceptExplainer:
    def test_nmf_explain(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        explainer = NMFConceptExplainer(n_concepts=5)
        target_layers = [model.features[-2]]
        saliency = explainer.explain(model, images, labels, target_layers)
        assert saliency.ndim == 3
        assert saliency.shape[0] == 4

    def test_nmf_concept_importance(self, model: _SimpleCNN, sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        images, labels = sample_data
        explainer = NMFConceptExplainer(n_concepts=5)
        target_layers = [model.features[-2]]
        importance = explainer.get_concept_importance(model, images, labels, target_layers)
        assert len(importance) == 5
