"""Augmentation presets and auto-selection for BNNR.

Provides ready-to-use augmentation sets for common use cases, with
automatic GPU/CPU detection and kornia integration when available.

Usage::

    from bnnr.presets import auto_select_augmentations, get_preset

    # Auto-select best augmentations for current hardware
    augs = auto_select_augmentations()

    # Use a named preset
    augs = get_preset("standard")

    # List available presets
    from bnnr.presets import list_presets
    print(list_presets())
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from bnnr.augmentations import (
    BaseAugmentation,
    BasicAugmentation,
    ChurchNoise,
    DifPresets,
    Drust,
    LuxferGlass,
    ProCAM,
    Smugs,
    TeaStains,
)

logger = logging.getLogger("bnnr.presets")

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "description": "Lightweight augmentations for fast iteration. Best for quick experiments.",
        "augmentations": [
            {"cls": ChurchNoise, "kwargs": {"probability": 0.5, "intensity": 0.5, "noise_strength_range": (3.0, 8.0)}},
            {"cls": ProCAM, "kwargs": {"probability": 0.5}},
        ],
    },
    "standard": {
        "description": "Balanced augmentation set for general-purpose training.",
        "augmentations": [
            {"cls": ChurchNoise, "kwargs": {"probability": 0.5, "intensity": 0.5}},
            {"cls": BasicAugmentation, "kwargs": {"probability": 0.5, "intensity": 0.5}},
            {"cls": ProCAM, "kwargs": {"probability": 0.5}},
            {"cls": DifPresets, "kwargs": {"probability": 0.5, "intensity": 0.7}},
        ],
    },
    "aggressive": {
        "description": "Heavy augmentations for robust model training. May slow down convergence.",
        "augmentations": [
            {"cls": ChurchNoise, "kwargs": {"probability": 0.5, "intensity": 0.5}},
            {"cls": BasicAugmentation, "kwargs": {"probability": 0.5, "intensity": 0.5}},
            {"cls": DifPresets, "kwargs": {"probability": 0.5, "intensity": 0.7}},
            {"cls": Drust, "kwargs": {"probability": 0.5, "intensity": 0.5}},
            {"cls": LuxferGlass, "kwargs": {"probability": 0.5, "intensity": 0.5}},
            {"cls": ProCAM, "kwargs": {"probability": 0.5}},
            {"cls": Smugs, "kwargs": {"probability": 0.5, "intensity": 1.5}},
            {"cls": TeaStains, "kwargs": {"probability": 0.5, "intensity": 0.5}},
        ],
    },
    "gpu": {
        "description": "GPU-native augmentations only. Fastest throughput, requires CUDA.",
        "augmentations": [
            {"cls": ChurchNoise, "kwargs": {"probability": 0.5, "intensity": 0.5}},
            {"cls": ProCAM, "kwargs": {"probability": 0.5}},
            {"cls": DifPresets, "kwargs": {"probability": 0.5, "intensity": 0.7}},
        ],
    },
}


def list_presets() -> dict[str, str]:
    """Return a dict of preset_name → description."""
    return {name: info["description"] for name, info in _PRESETS.items()}


def get_preset(
    name: str,
    random_state: int | None = 42,
    prob_override: float | None = None,
) -> list[BaseAugmentation]:
    """Get augmentations for a named preset.

    Parameters
    ----------
    name:
        Preset name. One of: ``auto``, ``light``, ``standard``, ``aggressive``,
        ``gpu``, ``screening``.
        If ``auto``, calls :func:`auto_select_augmentations`.
        If ``screening``, returns the ``aggressive`` preset with uniform ``p=0.5``
        — useful for the subset-proxy screening phase where uniform probability
        isolates augmentation *type* from *dosage*.
    random_state:
        Seed for reproducibility.
    prob_override:
        If set, override **all** augmentation probabilities to this value.
        Useful for screening phases where uniform probability is desired.

    Returns
    -------
    list[BaseAugmentation]
    """
    if name == "auto":
        augs = auto_select_augmentations(random_state=random_state)
        if prob_override is not None:
            for aug in augs:
                aug.probability = prob_override
        return augs

    if name == "screening":
        return get_preset("aggressive", random_state=random_state, prob_override=0.5)

    if name not in _PRESETS:
        available = ", ".join(sorted(list(_PRESETS.keys()) + ["auto", "screening"]))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    preset = _PRESETS[name]
    augmentations: list[BaseAugmentation] = []
    for aug_def in preset["augmentations"]:
        kwargs = dict(aug_def["kwargs"])
        kwargs["random_state"] = random_state
        if prob_override is not None:
            kwargs["probability"] = prob_override
        augmentations.append(aug_def["cls"](**kwargs))

    return augmentations


def auto_select_augmentations(
    random_state: int | None = 42,
    prefer_gpu: bool = True,
) -> list[BaseAugmentation]:
    """Auto-select the best augmentation set for the current environment.

    Strategy:
    1. If CUDA is available and ``prefer_gpu=True``, use GPU-native augmentations.
    2. If Kornia is installed, prefer Kornia GPU augmentations.
    3. Otherwise, fall back to the ``standard`` preset.

    Parameters
    ----------
    random_state:
        Seed for reproducibility.
    prefer_gpu:
        If True, prefer GPU-native augmentations when CUDA is available.

    Returns
    -------
    list[BaseAugmentation]
    """
    cuda_available = torch.cuda.is_available()

    if cuda_available and prefer_gpu:
        # Try kornia first
        try:
            from bnnr.kornia_aug import kornia_available

            if kornia_available():
                logger.info("Auto-select: Using Kornia GPU augmentations (CUDA + kornia available)")
                return _build_kornia_preset(random_state)
        except ImportError:
            pass

        # Fall back to built-in GPU-native augmentations
        logger.info("Auto-select: Using GPU-native built-in augmentations (CUDA available)")
        return get_preset("gpu", random_state=random_state)

    # CPU fallback
    logger.info("Auto-select: Using standard CPU augmentations (no CUDA)")
    return get_preset("standard", random_state=random_state)


def _build_kornia_preset(random_state: int | None = 42) -> list[BaseAugmentation]:
    """Build a GPU-native preset using Kornia augmentations + built-in GPU-native ones."""
    from bnnr.kornia_aug import KorniaAugmentation

    try:
        import kornia.augmentation as K  # noqa: N812
    except ImportError:
        return get_preset("gpu", random_state=random_state)

    augmentations: list[BaseAugmentation] = [
        # Built-in GPU-native augmentations
        ChurchNoise(probability=0.5, intensity=0.5, random_state=random_state),
        ProCAM(probability=0.5, random_state=random_state),
        DifPresets(probability=0.5, intensity=0.7, random_state=random_state),
        # Kornia augmentations
        KorniaAugmentation(
            kornia_transform=K.RandomHorizontalFlip(p=1.0),
            probability=0.5,
            name_override="kornia_hflip",
            random_state=random_state,
        ),
        KorniaAugmentation(
            kornia_transform=K.ColorJitter(0.2, 0.2, 0.2, 0.05, p=1.0),
            probability=0.5,
            name_override="kornia_color_jitter",
            random_state=random_state,
        ),
        KorniaAugmentation(
            kornia_transform=K.RandomRotation(degrees=15.0, p=1.0),
            probability=0.5,
            name_override="kornia_rotation",
            random_state=random_state,
        ),
    ]
    return augmentations


__all__ = [
    "auto_select_augmentations",
    "get_preset",
    "list_presets",
]
