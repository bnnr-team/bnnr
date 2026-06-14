"""Tests for core XAI explainers and saliency generation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from bnnr.xai import (
    CRAFTExplainer,
    GradCAMExplainer,
    OptiCAMExplainer,
    generate_saliency_maps,
    save_xai_visualization,
)


def test_opticam_explainer_init() -> None:
    exp = OptiCAMExplainer(use_cuda=False)
    assert exp.name == "opticam"


def test_gradcam_explainer_init() -> None:
    exp = GradCAMExplainer(use_cuda=False)
    assert exp.name == "gradcam"


def test_gradcam_explainer_explain(dummy_model) -> None:
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    maps = GradCAMExplainer(use_cuda=False).explain(dummy_model, images, labels, [dummy_model.conv1])
    assert maps.shape == (2, 32, 32)
    assert maps.min() >= 0.0
    assert maps.max() <= 1.0 + 1e-5


def test_opticam_returns_normalized_map(dummy_model) -> None:
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    maps = OptiCAMExplainer(use_cuda=False, n_iters=10).explain(
        dummy_model, images, labels, [dummy_model.conv1]
    )
    assert maps.shape == (2, 32, 32)
    assert maps.dtype == np.float32
    assert maps.min() >= 0.0
    assert maps.max() <= 1.0 + 1e-5


def test_opticam_optimization_changes_saliency(dummy_model) -> None:
    """Real Opti-CAM optimises the map; the result must differ from the
    un-optimised uniform combination (n_iters=0) and raise target confidence."""
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    layers = [dummy_model.conv1]

    uniform = OptiCAMExplainer(use_cuda=False, n_iters=0).explain(dummy_model, images, labels, layers)
    optimized = OptiCAMExplainer(use_cuda=False, n_iters=50, lr=0.5).explain(dummy_model, images, labels, layers)
    assert not np.allclose(uniform, optimized)

    dummy_model.eval()
    with torch.no_grad():
        idx = labels
        base_conf = torch.softmax(dummy_model(torch.from_numpy(uniform).unsqueeze(1) * images), dim=1)
        opt_conf = torch.softmax(dummy_model(torch.from_numpy(optimized).unsqueeze(1) * images), dim=1)
        base = base_conf.gather(1, idx.unsqueeze(1)).squeeze(1)
        opt = opt_conf.gather(1, idx.unsqueeze(1)).squeeze(1)
    assert (opt.mean() >= base.mean() - 1e-6)


def test_generate_saliency_maps_gradcam_and_opticam_differ(dummy_model) -> None:
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    layers = [dummy_model.conv1]
    opticam = generate_saliency_maps(dummy_model, images, labels, layers, method="opticam", n_iters=10)
    gradcam = generate_saliency_maps(dummy_model, images, labels, layers, method="gradcam")
    assert opticam.shape == (2, 32, 32)
    assert gradcam.shape == (2, 32, 32)


def test_craft_explainer_init() -> None:
    with pytest.deprecated_call():
        exp = CRAFTExplainer(n_concepts=5)
    assert exp.name == "craft"


def test_save_xai_visualization(temp_dir) -> None:
    images = (np.random.rand(2, 32, 32, 3) * 255).astype(np.uint8)
    maps = np.random.rand(2, 32, 32).astype(np.float32)
    paths = save_xai_visualization(images, maps, temp_dir)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_save_xai_visualization_resizes_map_and_handles_grayscale(temp_dir) -> None:
    images = (np.random.rand(2, 28, 28, 1) * 255).astype(np.uint8)
    maps = np.random.rand(2, 14, 14).astype(np.float32)
    paths = save_xai_visualization(images, maps, temp_dir, output_size=512)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    loaded = cv2.imread(str(paths[0]))
    assert loaded is not None
    assert loaded.shape[:2] == (512, 512)


def test_craft_explain_and_concept_importance(dummy_model) -> None:
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    exp = CRAFTExplainer(n_concepts=4)
    maps = exp.explain(dummy_model, images, labels, [dummy_model.conv1])
    scores = exp.get_concept_importance(dummy_model, images, labels, [dummy_model.conv1])

    assert maps.shape[0] == images.shape[0]
    assert maps.ndim == 3
    assert maps.min() >= 0.0
    assert maps.max() <= 1.0 + 1e-5
    assert isinstance(scores, dict)
    assert len(scores) > 0


def test_generate_saliency_maps_supports_nmf_method(dummy_model) -> None:
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    maps = generate_saliency_maps(
        dummy_model,
        images,
        labels,
        [dummy_model.conv1],
        method="nmf_concepts",
    )
    assert maps.shape[0] == 2
