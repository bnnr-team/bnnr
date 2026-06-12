"""Tests for the weights_only-first torch.load helper (FINDING-21)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from bnnr.utils import numpy_rng_safe_globals, safe_torch_load


class _ArbitraryObject:
    """A module-level (picklable) stand-in for arbitrary checkpoint content."""

    def __init__(self) -> None:
        self.payload = "arbitrary"


def _bnnr_like_payload() -> dict:
    """A payload shaped like a BNNR checkpoint: tensors + numpy RNG state."""
    return {
        "model_state": {"w": torch.zeros(3)},
        "iteration": 1,
        "augmentation_name": "icd",
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": [],
        },
    }


def test_bnnr_checkpoint_roundtrips_weights_only(
    temp_dir: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """BNNR checkpoints load via the safe weights_only=True path, no warning."""
    path = temp_dir / "ckpt.pt"
    torch.save(_bnnr_like_payload(), path)
    loaded = safe_torch_load(path, extra_safe_globals=numpy_rng_safe_globals())
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    assert torch.equal(loaded["model_state"]["w"], torch.zeros(3))
    assert isinstance(loaded["rng_state"]["numpy"], tuple)


def test_arbitrary_pickle_falls_back_with_warning(temp_dir: Path) -> None:
    """A checkpoint pickling an arbitrary object falls back, warning loudly."""
    path = temp_dir / "unsafe.pt"
    torch.save({"obj": _ArbitraryObject()}, path)
    with pytest.warns(RuntimeWarning, match=str(path.name)):
        loaded = safe_torch_load(path)
    assert loaded["obj"].payload == "arbitrary"


def test_pure_tensor_checkpoint_no_warning(
    temp_dir: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """A plain state_dict loads via weights_only=True with no fallback warning."""
    path = temp_dir / "weights.pt"
    torch.save({"w": torch.ones(4), "b": torch.zeros(2)}, path)
    loaded = safe_torch_load(path)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    assert torch.equal(loaded["w"], torch.ones(4))
