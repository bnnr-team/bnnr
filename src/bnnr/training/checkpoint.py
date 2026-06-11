"""State-dict helpers, checkpoint I/O, and internal trainer state dataclasses."""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from bnnr.utils import numpy_rng_safe_globals, safe_torch_load

if TYPE_CHECKING:
    from bnnr.trainer import BNNRTrainer


@dataclass
class _TrainerState:
    current_iteration: int
    active_augmentations: list[str]
    baseline_metrics: dict[str, float]


@dataclass
class _RuntimeState:
    """Mutable runtime flags that should NOT live on the frozen BNNRConfig."""

    xai_disabled: bool = False


def clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Create a detached clone of a state dict (initial allocation).

    Handles nested dicts (e.g. adapter state with ``model``, ``optimizer``
    sub-dicts) by recursing into them and cloning tensor leaves.
    """
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = clone_state_dict(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def copy_state_dict_inplace(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Copy *src* state dict values into *dst* buffers **in-place**.

    Avoids the overhead of ``copy.deepcopy`` by reusing pre-allocated
    tensors instead of creating new ones each time.  Handles nested
    dicts by recursing, copying tensor leaves in-place.  Non-tensor
    leaves are replaced directly (e.g. step counters).
    """
    for k, v in src.items():
        if isinstance(v, Tensor) and isinstance(dst.get(k), Tensor):
            dst[k].copy_(v)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            copy_state_dict_inplace(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def _is_ultralytics_tasks_backbone(model: Any) -> bool:
    """Ultralytics task modules use a BCHW tensor forward, not a list of CHW tensors like torchvision detection."""
    mod = type(model).__module__
    name = type(model).__name__
    return mod.startswith("ultralytics.nn.tasks") and name in {
        "DetectionModel",
        "OBBModel",
        "SegmentationModel",
        "PoseModel",
        "YOLOEModel",
    }


def save_checkpoint(
    trainer: BNNRTrainer,
    iteration: int,
    augmentation_name: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None = None,
    completed_candidates: list[str] | None = None,
    current_best_metric: float | None = None,
    iteration_results: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Persist trainer state to a checkpoint file under ``config.checkpoint_dir``."""
    trainer.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", augmentation_name)
    checkpoint_path = trainer.config.checkpoint_dir / f"iter_{iteration}_{safe_name}.pt"
    payload = {
        "model_state": trainer.model.state_dict(),
        "iteration": iteration,
        "augmentation_name": augmentation_name,
        "metrics": metrics,
        "active_augmentations": [aug.name for aug in trainer._active_augmentations],
        "baseline_metrics": baseline_metrics or metrics,
        "completed_candidates": completed_candidates or [],
        "current_best_metric": current_best_metric,
        "iteration_results": iteration_results or {},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
        },
        "config_snapshot": trainer.config.model_dump(mode="json"),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(trainer: BNNRTrainer, checkpoint_path: Path) -> dict[str, Any]:
    """Load a BNNR checkpoint and restore model + RNG state on *trainer*."""
    # BNNR checkpoints hold tensors + plain metadata + numpy RNG state, so the
    # safe weights_only=True path works once the numpy RNG globals are allowed.
    state = cast(
        dict[str, Any],
        safe_torch_load(
            checkpoint_path,
            map_location="cpu",
            extra_safe_globals=numpy_rng_safe_globals(),
        ),
    )
    expected_checkpoint_keys = {"model_state", "iteration", "augmentation_name"}
    if not isinstance(state, dict) or not expected_checkpoint_keys.issubset(state.keys()):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not appear to be a valid BNNR checkpoint. "
            f"Expected keys {expected_checkpoint_keys}, got "
            f"{set(state.keys()) if isinstance(state, dict) else type(state).__name__}."
        )
    trainer.model.load_state_dict(state["model_state"])
    rng_state = state.get("rng_state")
    if rng_state is not None:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch_cpu" in rng_state:
            torch.random.set_rng_state(rng_state["torch_cpu"])
        if "torch_cuda" in rng_state and torch.cuda.is_available():
            cuda_states = rng_state["torch_cuda"]
            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)
    return state
