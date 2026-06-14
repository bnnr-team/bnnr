"""Backward-compatibility safety net for the BNNR public API.

These tests lock down every public import path so that refactoring
internal module boundaries never breaks existing user code.
"""

from __future__ import annotations

import bnnr

# ---------------------------------------------------------------------------
#  Original 80 names that were importable from `bnnr` before the refactor.
#  Even though __all__ has been reduced, every one of these MUST remain
#  importable via ``from bnnr import <name>``.
# ---------------------------------------------------------------------------

_ALL_IMPORTABLE_NAMES = sorted([
    "AugmentationRunner",
    "BNNRTrainer",
    "BNNRConfig",
    "ModelAdapter",
    "XAICapableModel",
    "SimpleTorchAdapter",
    "BNNRRunResult",
    "BaseAugmentation",
    "AugmentationRegistry",
    "ChurchNoise",
    "BasicAugmentation",
    "DifPresets",
    "Drust",
    "LuxferGlass",
    "ProCAM",
    "Smugs",
    "TeaStains",
    "TorchvisionAugmentation",
    "ICD",
    "AICD",
    "XAICache",
    "BaseExplainer",
    "OptiCAMExplainer",
    "GradCAMExplainer",
    "NMFConceptExplainer",
    "CRAFTExplainer",
    "RealCRAFTExplainer",
    "RecursiveCRAFTExplainer",
    "generate_saliency_maps",
    "generate_craft_concepts",
    "generate_nmf_concepts",
    "save_xai_visualization",
    "analyze_xai_batch",
    "analyze_xai_batch_rich",
    "compute_xai_quality_score",
    "generate_class_diagnosis",
    "generate_class_insight",
    "generate_epoch_summary",
    "generate_rich_epoch_summary",
    "Reporter",
    "JsonlEventSink",
    "EVENT_SCHEMA_VERSION",
    "replay_events",
    "CheckpointInfo",
    "load_report",
    "compare_runs",
    "load_config",
    "save_config",
    "validate_config",
    "merge_configs",
    "apply_xai_preset",
    "get_xai_preset",
    "list_xai_presets",
    "quick_run",
    "KorniaAugmentation",
    "create_kornia_pipeline",
    "kornia_available",
    "AlbumentationsAugmentation",
    "albumentations_available",
    "auto_select_augmentations",
    "get_preset",
    "list_presets",
    "start_dashboard",
    "DetectionAdapter",
    "BboxAwareAugmentation",
    "AlbumentationsBboxAugmentation",
    "DetectionHorizontalFlip",
    "DetectionVerticalFlip",
    "DetectionRandomRotate90",
    "DetectionRandomScale",
    "MosaicAugmentation",
    "DetectionMixUp",
    "get_detection_preset",
    "DetectionICD",
    "DetectionAICD",
    "detection_collate_fn",
    "detection_collate_fn_with_index",
    "calculate_detection_metrics",
    "analyze_model",
    "AnalysisReport",
])

_REDUCED_ALL = sorted([
    "BNNRConfig",
    "BNNRTrainer",
    "quick_run",
    "ModelAdapter",
    "SimpleTorchAdapter",
    "BNNRRunResult",
    "Reporter",
    "load_report",
    "compare_runs",
    "BaseAugmentation",
    "BasicAugmentation",
    "ChurchNoise",
    "ICD",
    "AICD",
    "auto_select_augmentations",
    "get_preset",
    "list_presets",
    "OptiCAMExplainer",
    "GradCAMExplainer",
    "generate_saliency_maps",
    "start_dashboard",
    "analyze_model",
    "AnalysisReport",
    "__version__",
])


# ---------------------------------------------------------------------------
#  1. Every previously importable name is STILL importable from `bnnr`
# ---------------------------------------------------------------------------

def test_all_previously_importable_names_still_work() -> None:
    for name in _ALL_IMPORTABLE_NAMES:
        obj = getattr(bnnr, name, None)
        assert obj is not None, f"bnnr.{name} is no longer importable (backward compat broken)"


def test_all_list_matches_reduced_snapshot() -> None:
    assert sorted(bnnr.__all__) == _REDUCED_ALL


def test_all_list_is_subset_of_importable() -> None:
    for name in bnnr.__all__:
        if name == "__version__":
            continue
        assert name in _ALL_IMPORTABLE_NAMES, f"{name} in __all__ but not in importable set"


# ---------------------------------------------------------------------------
#  2. Canonical import paths from bnnr.core
# ---------------------------------------------------------------------------

def test_core_exports_bnnrconfig() -> None:
    from bnnr.core import BNNRConfig  # noqa: F811
    assert BNNRConfig is bnnr.BNNRConfig


def test_core_exports_bnnrtrainer() -> None:
    from bnnr.core import BNNRTrainer  # noqa: F811
    assert BNNRTrainer is bnnr.BNNRTrainer


def test_core_exports_simpletorchadapter() -> None:
    from bnnr.core import SimpleTorchAdapter  # noqa: F811
    assert SimpleTorchAdapter is bnnr.SimpleTorchAdapter


def test_core_state_dict_helpers() -> None:
    from bnnr.core import clone_state_dict, copy_state_dict_inplace  # noqa: F401


# ---------------------------------------------------------------------------
#  3. Submodule import paths that users may rely on
# ---------------------------------------------------------------------------

def test_config_module_imports() -> None:
    from bnnr.config import (  # noqa: F401
        apply_xai_preset,
        get_xai_preset,
        list_xai_presets,
        load_config,
        merge_configs,
        save_config,
        validate_config,
    )


def test_adapter_module_imports() -> None:
    from bnnr.adapter import (  # noqa: F401
        ModelAdapter,
        SimpleTorchAdapter,
        XAICapableModel,
    )


def test_xai_module_imports() -> None:
    from bnnr.xai import (  # noqa: F401
        BaseExplainer,
        OptiCAMExplainer,
        generate_saliency_maps,
    )


def test_augmentations_module_imports() -> None:
    from bnnr.augmentations import (  # noqa: F401
        AugmentationRegistry,
        BaseAugmentation,
        BasicAugmentation,
        ChurchNoise,
    )


def test_detection_module_imports() -> None:
    from bnnr.detection_adapter import DetectionAdapter  # noqa: F401
    from bnnr.detection_augmentations import get_detection_preset  # noqa: F401
    from bnnr.detection_collate import detection_collate_fn  # noqa: F401
    from bnnr.detection_icd import DetectionAICD, DetectionICD  # noqa: F401
    from bnnr.detection_metrics import calculate_detection_metrics  # noqa: F401


def test_reporting_module_imports() -> None:
    from bnnr.reporting import (  # noqa: F401
        BNNRRunResult,
        CheckpointInfo,
        Reporter,
        compare_runs,
        load_report,
    )


def test_quick_run_module_import() -> None:
    from bnnr.quick_run import quick_run  # noqa: F401


def test_events_module_imports() -> None:
    from bnnr.events import (  # noqa: F401
        EVENT_SCHEMA_VERSION,
        JsonlEventSink,
        replay_events,
    )


# ---------------------------------------------------------------------------
#  4. Identity checks — same object regardless of import path
# ---------------------------------------------------------------------------

def test_bnnrconfig_identity() -> None:
    from bnnr import BNNRConfig as PkgConfig
    from bnnr.core import BNNRConfig as CoreConfig

    assert PkgConfig is CoreConfig


def test_bnnrtrainer_identity() -> None:
    from bnnr import BNNRTrainer as PkgTrainer
    from bnnr.core import BNNRTrainer as CoreTrainer

    assert PkgTrainer is CoreTrainer


# ---------------------------------------------------------------------------
#  5. Version string
# ---------------------------------------------------------------------------

def test_version_exists() -> None:
    assert hasattr(bnnr, "__version__")
    assert isinstance(bnnr.__version__, str)


# ---------------------------------------------------------------------------
#  6. New import path for config_model
# ---------------------------------------------------------------------------

def test_config_model_import() -> None:
    from bnnr.config_model import BNNRConfig  # noqa: F401, F811


# ---------------------------------------------------------------------------
#  7. Training subpackage imports
# ---------------------------------------------------------------------------

def test_training_checkpoint_imports() -> None:
    from bnnr.training.checkpoint import clone_state_dict, copy_state_dict_inplace  # noqa: F401


def test_training_image_utils_imports() -> None:
    from bnnr.training.image_utils import tensor_to_uint8, uint8_to_tensor  # noqa: F401


def test_training_branching_imports() -> None:
    from bnnr.training.branching import select_best_path, should_prune_candidate  # noqa: F401


def test_training_callbacks_imports() -> None:
    from bnnr.training.callbacks import saliency_recommendations, xai_mean_quality  # noqa: F401


def test_training_metrics_imports() -> None:
    from bnnr.training.metrics import average_metrics  # noqa: F401


def test_training_dataset_profile_imports() -> None:
    from bnnr.training.dataset_profile import count_labels  # noqa: F401


# ---------------------------------------------------------------------------
#  8. Deprecation warnings for moved imports
# ---------------------------------------------------------------------------

def test_deprecated_detection_adapter_warns() -> None:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = getattr(bnnr, "DetectionAdapter")
        assert obj is not None
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "deprecated" in str(dep_warnings[0].message).lower()


def test_deprecated_kornia_warns() -> None:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = getattr(bnnr, "KorniaAugmentation")
        assert obj is not None
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1


def test_deprecated_events_warns() -> None:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = getattr(bnnr, "JsonlEventSink")
        assert obj is not None
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1


def test_nonexistent_attr_raises() -> None:
    import pytest
    with pytest.raises(AttributeError):
        getattr(bnnr, "NonExistentThing12345")
