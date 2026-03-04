"""Tests for core XAI explainers and saliency generation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from bnnr.xai import (
    CRAFTExplainer,
    OptiCAMExplainer,
    generate_saliency_maps,
    save_xai_visualization,
)


def test_opticam_explainer_init() -> None:
    exp = OptiCAMExplainer(use_cuda=False)
    assert exp.name == "opticam"


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
