# Python API Reference

## What you will find here
User-facing Python API for integrating BNNR with your own model and dataloaders.

## When to use this page
Use this when CLI presets are not enough and you need full control.

## Source of truth

This page documents only symbols exported publicly from `src/bnnr/__init__.py`.

## Core training API

- `BNNRConfig`
- `BNNRTrainer`
- `quick_run`
- `BNNRRunResult`
- `CheckpointInfo`

## Model adapter API

- `ModelAdapter`
- `XAICapableModel`
- `SimpleTorchAdapter`

## Reporting and events API

- `Reporter`
- `load_report`
- `compare_runs`
- `JsonlEventSink`
- `EVENT_SCHEMA_VERSION`
- `replay_events`

## Config helpers

- `load_config`
- `save_config`
- `validate_config`
- `merge_configs`
- `apply_xai_preset`
- `get_xai_preset`
- `list_xai_presets`

## Augmentation API

- `BaseAugmentation`
- `AugmentationRegistry`
- `AugmentationRunner`
- `TorchvisionAugmentation`
- `KorniaAugmentation`
- `AlbumentationsAugmentation`
- `create_kornia_pipeline`
- `kornia_available`
- `albumentations_available`

Built-in classification augmentations:

- `ChurchNoise`
- `BasicAugmentation`
- `DifPresets`
- `Drust`
- `LuxferGlass`
- `ProCAM`
- `Smugs`
- `TeaStains`

Preset helpers:

- `auto_select_augmentations`
- `get_preset`
- `list_presets`

## XAI API (classification)

Explainers and generation:

- `BaseExplainer`
- `OptiCAMExplainer`
- `NMFConceptExplainer`
- `CRAFTExplainer`
- `RealCRAFTExplainer`
- `RecursiveCRAFTExplainer`
- `generate_saliency_maps`
- `generate_craft_concepts`
- `generate_nmf_concepts`
- `save_xai_visualization`

Analysis and scoring:

- `analyze_xai_batch`
- `analyze_xai_batch_rich`
- `compute_xai_quality_score`
- `generate_class_diagnosis`
- `generate_class_insight`
- `generate_epoch_summary`
- `generate_rich_epoch_summary`

Cache:

- `XAICache`

ICD variants:

- `ICD`
- `AICD`

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

## `quick_run()` helper

`quick_run()` builds `SimpleTorchAdapter` internally.

```python
from bnnr import quick_run

result = quick_run(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
)
```

Useful arguments include `augmentations`, `config`/overrides, `criterion`, `optimizer`, `target_layers`, and `eval_metrics`.

## Detection

### Model adapters

- `DetectionAdapter(model, optimizer, target_layers=None, device="cuda", scheduler=None, use_amp=False, score_threshold=0.05)` — wraps torchvision-style detectors (Faster R-CNN, RetinaNet, SSD, FCOS). In train mode calls `model(images, targets)` for losses; in eval mode calls `model(images)` for prediction dicts.
- `UltralyticsDetectionAdapter(model_name="yolov8n.pt", device="cuda", score_threshold=0.05, num_classes=None, lr=1e-3, optimizer=None, use_amp=False)` — wraps Ultralytics YOLO. Runs inference via `YOLO.predict` with 0–255 scaling, and exposes `predict_detection_dicts(batch_bchw)` for XAI and probe snapshots.

Both adapters implement `train_step`, `eval_step`, `epoch_end_eval`, `epoch_end`, `state_dict`, `load_state_dict`, `get_target_layers`, and `get_model`.

For raw `ultralytics.nn.tasks` modules without this adapter, detection XAI stays disabled; see [troubleshooting.md §17](troubleshooting.md).

### Collate functions

- `detection_collate_fn(batch)` → `(Tensor[B,C,H,W], list[dict])` — stack images, keep targets as list
- `detection_collate_fn_with_index(batch)` → `(Tensor[B,C,H,W], list[dict], Tensor[B])` — same, with sample indices

### Detection augmentations

Bbox-aware transforms (subclass `BboxAwareAugmentation`):

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

Presets: `get_detection_preset(name)` with `name` ∈ `{"light", "standard", "aggressive"}`.

### Detection metrics

- `calculate_detection_metrics(predictions, targets, iou_thresholds=None, score_threshold=0.0)` → `{"map_50": float, "map_50_95": float}`
- `calculate_per_class_ap(predictions, targets, iou_threshold=0.5, class_names=None)` → per-class AP dict
- `calculate_detection_confusion_matrix(predictions, targets, num_classes=None, iou_threshold=0.5)` → `{"labels", "matrix"}`

### Detection XAI

- `generate_detection_saliency(model, images, target_layers, device="cpu", forward_layout="torchvision_list"|"ultralytics_bchw")` — backbone activation–based class-agnostic saliency
- `compute_detection_box_saliency_occlusion(...)` — per-box occlusion grid saliency
- `draw_boxes_on_image(...)` — draw xyxy boxes with optional labels, scores, and class names
- `overlay_saliency_heatmap(...)` — blend normalized saliency with colormap
- `save_detection_xai_panels(...)` — writes `{stem}_gt.png`, `{stem}_sal.png`, `{stem}_pred.png` triptych

See [detection.md](detection.md) for the full detection guide with examples.
