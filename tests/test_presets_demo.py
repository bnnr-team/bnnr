"""Tests for the onboarding ``demo`` augmentation preset."""

from __future__ import annotations

import torch
import torch.nn as nn

from bnnr.icd import AICD, ICD
from bnnr.presets import (
    build_demo_augmentations,
    build_icd_augmentations,
    get_preset,
    list_presets,
)


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


def test_demo_listed_in_presets() -> None:
    presets = list_presets()
    assert "demo" in presets
    assert "ICD" in presets["demo"] or "saliency" in presets["demo"].lower()


def test_build_demo_augmentations_includes_icd() -> None:
    model = _TinyCNN()
    target_layers = [model.conv2]
    augs = build_demo_augmentations(model, target_layers, random_state=42)
    assert len(augs) == 2
    assert any(isinstance(a, ICD) for a in augs)


def test_get_preset_demo_with_model() -> None:
    model = _TinyCNN()
    augs = get_preset("demo", random_state=7, model=model, target_layers=[model.conv2])
    assert any(isinstance(a, ICD) for a in augs)


def test_icd_listed_in_presets() -> None:
    presets = list_presets()
    assert "icd" in presets
    assert "ICD" in presets["icd"] or "saliency" in presets["icd"].lower()


def test_build_icd_augmentations_has_icd_and_aicd() -> None:
    model = _TinyCNN()
    augs = build_icd_augmentations(model, [model.conv2], random_state=42)
    assert any(isinstance(a, ICD) for a in augs)
    assert any(isinstance(a, AICD) for a in augs)


def test_get_preset_icd_requires_model() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires model"):
        get_preset("icd")


def test_get_preset_icd_with_model() -> None:
    model = _TinyCNN()
    augs = get_preset("icd", random_state=7, model=model, target_layers=[model.conv2])
    assert any(isinstance(a, ICD) for a in augs)
    assert any(isinstance(a, AICD) for a in augs)
