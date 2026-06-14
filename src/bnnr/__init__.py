"""Public package exports for the BNNR library."""

from __future__ import annotations

import importlib
import warnings

# ── Layer 1: Always-available, promoted to __all__ ──────────────────────
from bnnr.adapter import ModelAdapter, SimpleTorchAdapter
from bnnr.analyze import AnalysisReport, analyze_model
from bnnr.augmentations import BaseAugmentation, BasicAugmentation, ChurchNoise
from bnnr.core import BNNRConfig, BNNRTrainer
from bnnr.dashboard.serve import start_dashboard
from bnnr.icd import AICD, ICD
from bnnr.presets import auto_select_augmentations, get_preset, list_presets
from bnnr.quick_run import quick_run
from bnnr.reporting import BNNRRunResult, Reporter, compare_runs, load_report
from bnnr.version import __version__
from bnnr.xai import GradCAMExplainer, OptiCAMExplainer, generate_saliency_maps

# Full API available via subpackage imports (bnnr.xai, bnnr.augmentations,
# bnnr.detection_adapter, bnnr.events, bnnr.kornia_aug, etc.).
# Only the most common entry-points are promoted to __all__.
__all__ = [
    # Core
    "BNNRConfig",
    "BNNRTrainer",
    "quick_run",
    # Adapters
    "ModelAdapter",
    "SimpleTorchAdapter",
    # Reporting
    "BNNRRunResult",
    "Reporter",
    "load_report",
    "compare_runs",
    # Augmentations (essentials)
    "BaseAugmentation",
    "BasicAugmentation",
    "ChurchNoise",
    "ICD",
    "AICD",
    "auto_select_augmentations",
    "get_preset",
    "list_presets",
    # XAI (essentials)
    "OptiCAMExplainer",
    "GradCAMExplainer",
    "generate_saliency_maps",
    # Dashboard
    "start_dashboard",
    # Analysis
    "analyze_model",
    "AnalysisReport",
    # Version
    "__version__",
]


# ── Layer 2: Deprecated top-level imports ───────────────────────────────
# These names are still importable from `bnnr` but emit a DeprecationWarning.
# Users should switch to ``from bnnr.<subpackage> import <name>``.

_DEPRECATED_EXPORTS: dict[str, tuple[str, str]] = {
    # name -> (subpackage, attribute_name)
    # Detection
    "DetectionAdapter": ("bnnr.detection_adapter", "DetectionAdapter"),
    "BboxAwareAugmentation": ("bnnr.detection_augmentations", "BboxAwareAugmentation"),
    "AlbumentationsBboxAugmentation": ("bnnr.detection_augmentations", "AlbumentationsBboxAugmentation"),
    "DetectionHorizontalFlip": ("bnnr.detection_augmentations", "DetectionHorizontalFlip"),
    "DetectionVerticalFlip": ("bnnr.detection_augmentations", "DetectionVerticalFlip"),
    "DetectionRandomRotate90": ("bnnr.detection_augmentations", "DetectionRandomRotate90"),
    "DetectionRandomScale": ("bnnr.detection_augmentations", "DetectionRandomScale"),
    "MosaicAugmentation": ("bnnr.detection_augmentations", "MosaicAugmentation"),
    "DetectionMixUp": ("bnnr.detection_augmentations", "DetectionMixUp"),
    "get_detection_preset": ("bnnr.detection_augmentations", "get_detection_preset"),
    "DetectionICD": ("bnnr.detection_icd", "DetectionICD"),
    "DetectionAICD": ("bnnr.detection_icd", "DetectionAICD"),
    "detection_collate_fn": ("bnnr.detection_collate", "detection_collate_fn"),
    "detection_collate_fn_with_index": ("bnnr.detection_collate", "detection_collate_fn_with_index"),
    "calculate_detection_metrics": ("bnnr.detection_metrics", "calculate_detection_metrics"),
    # Kornia / Albumentations
    "KorniaAugmentation": ("bnnr.kornia_aug", "KorniaAugmentation"),
    "create_kornia_pipeline": ("bnnr.kornia_aug", "create_kornia_pipeline"),
    "kornia_available": ("bnnr.kornia_aug", "kornia_available"),
    "AlbumentationsAugmentation": ("bnnr.albumentations_aug", "AlbumentationsAugmentation"),
    "albumentations_available": ("bnnr.albumentations_aug", "albumentations_available"),
    # Events
    "JsonlEventSink": ("bnnr.events", "JsonlEventSink"),
    "EVENT_SCHEMA_VERSION": ("bnnr.events", "EVENT_SCHEMA_VERSION"),
    "replay_events": ("bnnr.events", "replay_events"),
    # Augmentation extras
    "AugmentationRunner": ("bnnr.augmentation_runner", "AugmentationRunner"),
    "AugmentationRegistry": ("bnnr.augmentations", "AugmentationRegistry"),
    "DifPresets": ("bnnr.augmentations", "DifPresets"),
    "Drust": ("bnnr.augmentations", "Drust"),
    "LuxferGlass": ("bnnr.augmentations", "LuxferGlass"),
    "ProCAM": ("bnnr.augmentations", "ProCAM"),
    "Smugs": ("bnnr.augmentations", "Smugs"),
    "TeaStains": ("bnnr.augmentations", "TeaStains"),
    "TorchvisionAugmentation": ("bnnr.augmentations", "TorchvisionAugmentation"),
    # XAI extras
    "XAICapableModel": ("bnnr.adapter", "XAICapableModel"),
    "XAICache": ("bnnr.xai_cache", "XAICache"),
    "BaseExplainer": ("bnnr.xai", "BaseExplainer"),
    "NMFConceptExplainer": ("bnnr.xai", "NMFConceptExplainer"),
    "CRAFTExplainer": ("bnnr.xai", "CRAFTExplainer"),
    "RealCRAFTExplainer": ("bnnr.xai", "RealCRAFTExplainer"),
    "RecursiveCRAFTExplainer": ("bnnr.xai", "RecursiveCRAFTExplainer"),
    "generate_craft_concepts": ("bnnr.xai", "generate_craft_concepts"),
    "generate_nmf_concepts": ("bnnr.xai", "generate_nmf_concepts"),
    "save_xai_visualization": ("bnnr.xai", "save_xai_visualization"),
    "analyze_xai_batch": ("bnnr.xai_analysis", "analyze_xai_batch"),
    "analyze_xai_batch_rich": ("bnnr.xai_analysis", "analyze_xai_batch_rich"),
    "compute_xai_quality_score": ("bnnr.xai_analysis", "compute_xai_quality_score"),
    "generate_class_diagnosis": ("bnnr.xai_analysis", "generate_class_diagnosis"),
    "generate_class_insight": ("bnnr.xai_analysis", "generate_class_insight"),
    "generate_epoch_summary": ("bnnr.xai_analysis", "generate_epoch_summary"),
    "generate_rich_epoch_summary": ("bnnr.xai_analysis", "generate_rich_epoch_summary"),
    # Reporting extras
    "CheckpointInfo": ("bnnr.reporting", "CheckpointInfo"),
    # Config helpers
    "load_config": ("bnnr.config", "load_config"),
    "save_config": ("bnnr.config", "save_config"),
    "validate_config": ("bnnr.config", "validate_config"),
    "merge_configs": ("bnnr.config", "merge_configs"),
    "apply_xai_preset": ("bnnr.config", "apply_xai_preset"),
    "get_xai_preset": ("bnnr.config", "get_xai_preset"),
    "list_xai_presets": ("bnnr.config", "list_xai_presets"),
}


def __getattr__(name: str) -> object:
    if name in _DEPRECATED_EXPORTS:
        module_path, attr_name = _DEPRECATED_EXPORTS[name]
        warnings.warn(
            f"Importing {name} from bnnr is deprecated. "
            f"Use 'from {module_path} import {attr_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'bnnr' has no attribute {name!r}")
