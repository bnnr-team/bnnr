# Python API Reference

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## What you will find here
User-facing Python API for integrating BNNR with your own model and dataloaders.

## When to use this page
Use this when CLI presets are not enough and you need full control.

## Source of truth

Implemented in `src/bnnr/`. The package exposes two import tiers:

1. **Stable API** — names in `bnnr.__all__` (recommended for new code).
2. **Backward-compatible imports** — additional names importable as `from bnnr import …` but emit `DeprecationWarning`; prefer `from bnnr.<subpackage> import …` (see [tests/test_backward_compat.py](../tests/test_backward_compat.py)).

Detection, events, config I/O, and extended XAI symbols are **not** in `__all__`; import them from the submodules listed below.

## Stable API (`bnnr.__all__`)

| Symbol | Role |
|--------|------|
| `__version__` | Package version string |
| `BNNRConfig`, `BNNRTrainer` | Config model and training loop |
| `quick_run` | One-call classification training |
| `ModelAdapter`, `SimpleTorchAdapter` | Model integration protocols |
| `BNNRRunResult`, `Reporter`, `load_report`, `compare_runs` | Run outputs and comparison |
| `BaseAugmentation`, `BasicAugmentation`, `ChurchNoise` | Core augmentation types |
| `ICD`, `AICD` | Saliency-guided augmentations |
| `auto_select_augmentations`, `get_preset`, `list_presets` | Preset helpers |
| `OptiCAMExplainer`, `GradCAMExplainer`, `generate_saliency_maps` | Classification XAI |
| `start_dashboard` | Live dashboard helper |
| `analyze_model`, `AnalysisReport` | Standalone model diagnostics |

## Quickstart API (recommended)

- **`quick_run(model, train_loader, val_loader, ...)`** — one-call classification training with `default_train_config()` defaults (`m_epochs=3`, `max_iterations=2`, `device=auto`, XAI on). Infers `target_layers` from the last `Conv2d` when omitted. Set `dashboard=True` to start the live dashboard before `run()` (non-blocking afterward). Override via kwargs, e.g. `m_epochs=1`. Detection and multi-label: see [golden_path.md](golden_path.md).

```python
import bnnr

result = bnnr.quick_run(model, train_loader, val_loader)
print(result.best_metrics)
```

Returns `BNNRRunResult`.

## Core training API (low-level)

Stable: `BNNRConfig`, `BNNRTrainer`, `BNNRRunResult`.

Backward-compatible: `CheckpointInfo` — `from bnnr.reporting import CheckpointInfo`.

## Model adapter API

Stable: `ModelAdapter`, `SimpleTorchAdapter`.

Backward-compatible: `XAICapableModel` — `from bnnr.adapter import XAICapableModel`.

## Analysis API (standalone model diagnostics)

- `analyze_model(adapter, val_loader, *, task, output_dir, run_data_quality, max_worst, xai_enabled, xai_method, xai_samples, cv_folds, data_quality_max_samples)` — run full analysis (metrics, XAI, data quality, failure analysis, patterns, recommendations) on an adapter and validation loader; returns `AnalysisReport`.
- `AnalysisReport` — dataclass; core attributes: `metrics`, `per_class_accuracy`, `confusion`, `xai_insights`, `xai_diagnoses`, `xai_quality_summary`, `data_quality_result`, `failure_patterns`, `recommendations`; extended attributes (v0.2+): `schema_version`, `executive_summary`, `findings`, `recommendations_structured`, `class_diagnostics`, `true_distribution`, `pred_distribution`, `distribution_summary`, `failure_patterns_extended`, `xai_quality_per_class`, `xai_examples_per_class`, `data_quality_summary`, `cv_results`, `calibration_summary`, `confusion_pair_xai`, `best_worst_examples`, `analysis_scope`; methods: `save(output_dir)`, `to_html(path)`, `failure_patterns_list()`.

See `analyze.md` for usage and CLI.

## Reporting and events API

Stable: `Reporter`, `load_report`, `compare_runs`.

Backward-compatible (prefer submodule imports):

- `JsonlEventSink`, `EVENT_SCHEMA_VERSION` (`"2.1"`), `replay_events` — `from bnnr.events import …`

## Config helpers (backward-compatible)

Import from `bnnr.config` (top-level `from bnnr import …` is deprecated):

- `load_config`, `save_config`, `validate_config`, `merge_configs`
- `apply_xai_preset`, `get_xai_preset`, `list_xai_presets` — presets `xai_light`, `xai_full`, `xai_adaptive`

CLI defaults without YAML: `default_train_config()` (`m_epochs=3`, `max_iterations=2`, `device="auto"`), `default_demo_config()` (`m_epochs=1`, `max_iterations=1`).

## Augmentation API

Stable: `BaseAugmentation`, `BasicAugmentation`, `ChurchNoise`, `ICD`, `AICD`, `auto_select_augmentations`, `get_preset`, `list_presets`.

Backward-compatible — `from bnnr.augmentations import …` / `from bnnr.augmentation_runner import …`:

- `AugmentationRegistry`, `AugmentationRunner`, `TorchvisionAugmentation`
- `DifPresets`, `Drust`, `LuxferGlass`, `ProCAM`, `Smugs`, `TeaStains`

Optional backends — `from bnnr.kornia_aug import …`, `from bnnr.albumentations_aug import …`:

- `KorniaAugmentation`, `create_kornia_pipeline`, `kornia_available`
- `AlbumentationsAugmentation`, `albumentations_available`

## XAI API (classification)

Stable: `OptiCAMExplainer`, `GradCAMExplainer`, `generate_saliency_maps`, `ICD`, `AICD`.

Backward-compatible — `from bnnr.xai import …`, `from bnnr.xai_cache import …`, `from bnnr.xai_analysis import …`:

- Explainers: `BaseExplainer`, `NMFConceptExplainer`, `CRAFTExplainer`, `RealCRAFTExplainer`, `RecursiveCRAFTExplainer`
- `generate_craft_concepts`, `generate_nmf_concepts`, `save_xai_visualization`
- `analyze_xai_batch`, `analyze_xai_batch_rich`, `compute_xai_quality_score`
- `generate_class_diagnosis`, `generate_class_insight`, `generate_epoch_summary`, `generate_rich_epoch_summary`
- `XAICache`

## Dashboard helper

- `start_dashboard` — returns the LAN URL when the background server starts, or `None` if optional dashboard dependencies are missing (no misleading localhost URL).

## Minimal classification integration

```python
import torch
import torch.nn as nn
from bnnr import BNNRConfig, BNNRTrainer, SimpleTorchAdapter, auto_select_augmentations

model = ...
train_loader = ...  # (image, label, index)
val_loader = ...

adapter = SimpleTorchAdapter(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    target_layers=[...],
    device="auto",
)

config = BNNRConfig(m_epochs=3, max_iterations=2, device="auto")
trainer = BNNRTrainer(adapter, train_loader, val_loader, auto_select_augmentations(), config)
result = trainer.run()
print(result.best_metrics)
```

## Detection

Import detection symbols from submodules (top-level `from bnnr import DetectionAdapter` is deprecated).

### Model adapters

`from bnnr.detection_adapter import DetectionAdapter, UltralyticsDetectionAdapter`

- `DetectionAdapter(model, optimizer, target_layers=None, device="cuda", scheduler=None, use_amp=False, score_threshold=0.05)` — wraps torchvision-style detectors (Faster R-CNN, RetinaNet, SSD, FCOS). In train mode calls `model(images, targets)` for losses; in eval mode calls `model(images)` for prediction dicts. `device` accepts `cuda`, `cpu`, or `auto`.
- `UltralyticsDetectionAdapter(model_name="yolov8n.pt", device="cuda", score_threshold=0.05, num_classes=None, lr=1e-3, optimizer=None, use_amp=False)` — wraps Ultralytics YOLO. Runs inference via `YOLO.predict` with 0–255 scaling, and exposes `predict_detection_dicts(batch_bchw)` for XAI and probe snapshots.

Both adapters implement `train_step`, `eval_step`, `epoch_end_eval`, `epoch_end`, `state_dict`, `load_state_dict`, `get_target_layers`, and `get_model`.

For raw `ultralytics.nn.tasks` modules without this adapter, detection XAI stays disabled; see [troubleshooting.md §17](troubleshooting.md).

### Collate functions

`from bnnr.detection_collate import detection_collate_fn, detection_collate_fn_with_index`

- `detection_collate_fn(batch)` → `(Tensor[B,C,H,W], list[dict])` — stack images, keep targets as list
- `detection_collate_fn_with_index(batch)` → `(Tensor[B,C,H,W], list[dict], Tensor[B])` — same, with sample indices

### Detection augmentations

`from bnnr.detection_augmentations import …` — bbox-aware transforms (subclass `BboxAwareAugmentation`):

- `DetectionHorizontalFlip` — horizontal flip with bbox mirroring
- `DetectionVerticalFlip` — vertical flip with bbox mirroring
- `DetectionRandomRotate90` — 90° rotation with bbox transform
- `DetectionRandomScale(scale_range=(0.8, 1.2))` — random resize with bbox scaling
- `MosaicAugmentation(output_size=(640, 640))` — 4-image mosaic composition (requires `set_pool`)
- `DetectionMixUp(alpha_range=(0.3, 0.7))` — alpha-blend two images with merged targets (requires `set_pool`)
- `AlbumentationsBboxAugmentation(transform)` — Albumentations wrapper with bbox support

XAI-driven augmentations:

- `DetectionICD(threshold_percentile=70.0, tile_size=8, fill_strategy="gaussian_blur")` — masks high-saliency (object) tiles
- `DetectionAICD(threshold_percentile=70.0, tile_size=8, fill_strategy="gaussian_blur")` — masks low-saliency (background) tiles

Presets: `from bnnr.detection_augmentations import get_detection_preset` — `name` ∈ `{"light", "standard", "aggressive"}`.

### Detection metrics

`from bnnr.detection_metrics import …`

- `calculate_detection_metrics(predictions, targets, iou_thresholds=None, score_threshold=0.0)` → `{"map_50": float, "map_50_95": float}`
- `calculate_per_class_ap(predictions, targets, iou_threshold=0.5, class_names=None)` → per-class AP dict
- `calculate_detection_confusion_matrix(predictions, targets, num_classes=None, iou_threshold=0.5)` → `{"labels", "matrix"}`

### Detection XAI

`from bnnr.detection_xai import …`

- `generate_detection_saliency(model, images, target_layers, device="cpu", forward_layout="torchvision_list"|"ultralytics_bchw")` — backbone activation–based class-agnostic saliency
- `compute_detection_box_saliency_occlusion(...)` — per-box occlusion grid saliency
- `draw_boxes_on_image(...)` — draw xyxy boxes with optional labels, scores, and class names
- `overlay_saliency_heatmap(...)` — blend normalized saliency with colormap
- `save_detection_xai_panels(...)` — writes `{stem}_gt.png`, `{stem}_sal.png`, `{stem}_pred.png` triptych

See [detection.md](detection.md) for the full detection guide with examples.
