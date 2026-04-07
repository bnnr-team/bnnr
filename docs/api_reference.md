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

- `start_dashboard`

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

## Detection and Ultralytics YOLO

- Torchvision-style detectors (Faster R-CNN, etc.): use `DetectionAdapter` from `bnnr.detection_adapter`.
- Ultralytics YOLO: use `UltralyticsDetectionAdapter`. It runs inference via `YOLO.predict` with the same 0–255 scaling as `eval_step`, and exposes `predict_detection_dicts(batch_bchw)` for detection XAI and probe snapshots inside `BNNRTrainer`.

For raw `ultralytics.nn.tasks` modules without this adapter, detection XAI stays disabled; see [troubleshooting.md §17](troubleshooting.md).
