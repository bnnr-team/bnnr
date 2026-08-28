"""BNNR training subpackage — modularized training loop components."""

from bnnr.training.branching import (
    get_current_best_metric,
    select_best_path,
    should_prune_candidate,
    top_k_candidate_names,
)
from bnnr.training.callbacks import (
    adapt_icd_thresholds,
    build_xai_insights,
    build_xai_summary,
    saliency_recommendations,
    xai_mean_quality,
)
from bnnr.training.checkpoint import (
    clone_state_dict,
    copy_state_dict_inplace,
)
from bnnr.training.dataset_profile import compute_dataset_profile, count_labels
from bnnr.training.image_utils import (
    BatchScale,
    NormalisedInputError,
    det_uint8_batch_to_float01,
    detect_batch_scale,
    resize_saliency_batch,
    tensor_batch_to_preview_uint8,
    tensor_to_uint8,
    uint8_to_tensor,
)
from bnnr.training.loop import evaluate, run, run_single_iteration, train_epoch
from bnnr.training.metrics import average_metrics
from bnnr.training.probe import initialize_report_probe_samples
from bnnr.training.xai_runner import (
    generate_augmentation_previews,
    generate_xai,
    generate_xai_lightweight,
    precompute_xai_cache,
)

__all__ = [
    "BatchScale",
    "NormalisedInputError",
    "adapt_icd_thresholds",
    "average_metrics",
    "build_xai_insights",
    "build_xai_summary",
    "clone_state_dict",
    "compute_dataset_profile",
    "copy_state_dict_inplace",
    "count_labels",
    "det_uint8_batch_to_float01",
    "detect_batch_scale",
    "evaluate",
    "generate_augmentation_previews",
    "generate_xai",
    "generate_xai_lightweight",
    "precompute_xai_cache",
    "run",
    "run_single_iteration",
    "get_current_best_metric",
    "initialize_report_probe_samples",
    "resize_saliency_batch",
    "saliency_recommendations",
    "select_best_path",
    "should_prune_candidate",
    "tensor_batch_to_preview_uint8",
    "tensor_to_uint8",
    "top_k_candidate_names",
    "train_epoch",
    "uint8_to_tensor",
    "xai_mean_quality",
]
