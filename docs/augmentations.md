# Augmentations and Presets

## What you will find here
Implemented presets and augmentation classes currently available in code, with constraints by task/backend.

## When to use this page
Use this when selecting augmentation candidates for classification, detection, or multi-label runs.

## Presets (`src/bnnr/presets.py`)

Available names:

- `auto`
- `light`
- `standard`
- `aggressive`
- `gpu`
- `screening` (API-level helper that maps to aggressive with uniform probability)

Examples:

```python
from bnnr import auto_select_augmentations, get_preset

augs_auto = auto_select_augmentations(random_state=42)
augs_std = get_preset("standard", random_state=42)
```

CLI `--preset` supports: `auto`, `light`, `standard`, `aggressive`, `gpu`.

## Built-in classification augmentations

Main classes used by presets:

- `ChurchNoise`
- `BasicAugmentation`
- `DifPresets`
- `Drust`
- `LuxferGlass`
- `ProCAM`
- `Smugs`
- `TeaStains`

## Detection augmentations

From `src/bnnr/detection_augmentations.py` and `src/bnnr/detection_icd.py`:

- `DetectionHorizontalFlip`
- `DetectionVerticalFlip`
- `DetectionRandomRotate90`
- `DetectionRandomScale`
- `MosaicAugmentation`
- `DetectionMixUp`
- `AlbumentationsBboxAugmentation`
- `DetectionICD`
- `DetectionAICD`

Detection augmentations are bbox-aware and must preserve valid box/label structure.

## Multi-label note

Multi-label task uses the same augmentation pipeline interface as classification.
Selection defaults differ (`f1_samples`), but preset mechanics stay the same.

## Optional integrations

### Kornia (`.[gpu]`)

```bash
python3 -m pip install -e ".[gpu]"
```

Used for GPU-native augmentation paths when available.

### Albumentations (`.[albumentations]`)

```bash
python3 -m pip install -e ".[albumentations]"
```

Used by bbox-aware wrapper `AlbumentationsBboxAugmentation`.

## Practical constraints

- Grayscale datasets (e.g. MNIST) can expose edge cases for some aggressive augmentations.
- For quick smoke tests, start with `light` on RGB datasets.
- For throughput-focused GPU runs, start with `gpu` or `auto`.

## Custom augmentation registration

Register subclasses of `BaseAugmentation` in `AugmentationRegistry` and keep deterministic behavior via `random_state` where relevant.
