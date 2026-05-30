"""Backward-compatible re-exports for the BNNR training core.

``BNNRTrainer`` lives in :mod:`bnnr.trainer`; this module preserves
``from bnnr.core import BNNRTrainer`` and related import paths.
"""

from __future__ import annotations

from bnnr.adapter import (  # noqa: F401 — re-exported for backward compat
    ModelAdapter,
    SimpleTorchAdapter,
    XAICapableModel,
)
from bnnr.config_model import BNNRConfig  # noqa: F401 — re-exported for backward compat
from bnnr.trainer import BNNRTrainer
from bnnr.training.checkpoint import (  # noqa: F401 — re-exported for backward compat
    clone_state_dict,
    copy_state_dict_inplace,
)

__all__ = [
    "BNNRConfig",
    "BNNRTrainer",
    "ModelAdapter",
    "SimpleTorchAdapter",
    "XAICapableModel",
    "clone_state_dict",
    "copy_state_dict_inplace",
]
