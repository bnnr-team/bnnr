"""Public package exports for the BNNR library."""

from bnnr.adapter import ModelAdapter, SimpleTorchAdapter, XAICapableModel
from bnnr.albumentations_aug import AlbumentationsAugmentation, albumentations_available
from bnnr.augmentation_runner import AugmentationRunner
from bnnr.augmentations import (
    AugmentationRegistry,
    BaseAugmentation,
    BasicAugmentation,
    ChurchNoise,
    DifPresets,
    Drust,
    LuxferGlass,
    ProCAM,
    Smugs,
    TeaStains,
    TorchvisionAugmentation,
)
from bnnr.config import (
    apply_xai_preset,
    get_xai_preset,
    list_xai_presets,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)
from bnnr.core import BNNRConfig, BNNRTrainer

# Dashboard — public helper
from bnnr.dashboard.serve import start_dashboard
from bnnr.detection_adapter import DetectionAdapter
from bnnr.detection_augmentations import (
    AlbumentationsBboxAugmentation,
    BboxAwareAugmentation,
    DetectionHorizontalFlip,
    DetectionMixUp,
    DetectionRandomRotate90,
    DetectionRandomScale,
    DetectionVerticalFlip,
    MosaicAugmentation,
    get_detection_preset,
)
from bnnr.detection_collate import detection_collate_fn, detection_collate_fn_with_index
from bnnr.detection_icd import DetectionAICD, DetectionICD
from bnnr.detection_metrics import calculate_detection_metrics
from bnnr.events import EVENT_SCHEMA_VERSION, JsonlEventSink, replay_events
from bnnr.icd import AICD, ICD
from bnnr.kornia_aug import KorniaAugmentation, create_kornia_pipeline, kornia_available
from bnnr.presets import auto_select_augmentations, get_preset, list_presets
from bnnr.quick_run import quick_run
from bnnr.reporting import BNNRRunResult, CheckpointInfo, Reporter, compare_runs, load_report
from bnnr.xai import (
    BaseExplainer,
    CRAFTExplainer,
    NMFConceptExplainer,
    OptiCAMExplainer,
    RealCRAFTExplainer,
    RecursiveCRAFTExplainer,
    generate_craft_concepts,
    generate_nmf_concepts,
    generate_saliency_maps,
    save_xai_visualization,
)
from bnnr.xai_analysis import (
    analyze_xai_batch,
    analyze_xai_batch_rich,
    compute_xai_quality_score,
    generate_class_diagnosis,
    generate_class_insight,
    generate_epoch_summary,
    generate_rich_epoch_summary,
)
from bnnr.xai_cache import XAICache

__version__ = "0.1.0.post1"

__all__ = [
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
]
