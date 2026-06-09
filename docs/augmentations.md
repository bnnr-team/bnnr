# Augmentations and Presets

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## What you will find here
Implemented presets and augmentation classes currently available in code, with constraints by task/backend.

## When to use this page
Use this when selecting augmentation candidates for classification or multi-label runs.

## Presets (`src/bnnr/presets.py`)

Available names:

- `auto` — virtual; hardware-aware selection via `auto_select_augmentations`
- `light`
- `standard`
- `aggressive`
- `gpu`
- `demo` — ICD + ChurchNoise (used by `python -m bnnr demo` and `get_preset("demo")`; not shown by `bnnr list-presets`)
- `screening` — virtual; maps to aggressive with uniform probability (`get_preset` / API only)
- `none` — no augmentations (`python -m bnnr train --preset none`; shown by `bnnr list-presets`)

Examples:

```python
from bnnr import auto_select_augmentations, get_preset

augs_auto = auto_select_augmentations(random_state=42)
augs_std = get_preset("standard", random_state=42)
```

CLI `--preset` / `--augmentation-preset` on `train` supports: `auto`, `light`, `standard`, `aggressive`, `gpu`, `none` (unknown names fall back to `auto` with a warning). The `demo` command always uses preset `demo`.

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

## Detection augmentations

Detection augmentations are bbox-aware — they transform both images and bounding boxes consistently. They subclass `BboxAwareAugmentation` and implement `apply_with_targets(image, target) -> (image, target)`.

### Built-in detection transforms

| Class | Description | Key parameters |
|-------|-------------|----------------|
| `DetectionHorizontalFlip` | Horizontal flip with bbox mirroring | `probability` |
| `DetectionVerticalFlip` | Vertical flip with bbox mirroring | `probability` |
| `DetectionRandomRotate90` | 90° rotation with bbox transform | `probability` |
| `DetectionRandomScale` | Random resize with bbox scaling | `probability`, `scale_range` |
| `MosaicAugmentation` | 4-image mosaic | `probability`, `output_size` |
| `DetectionMixUp` | Alpha-blend two images | `probability`, `alpha_range` |
| `AlbumentationsBboxAugmentation` | Albumentations wrapper | `transform`, `probability` |

### Detection ICD / AICD

- `DetectionICD` — masks high-saliency tiles (forces context learning)
- `DetectionAICD` — masks low-saliency tiles (sharpens object focus)

Both accept `threshold_percentile`, `tile_size`, `fill_strategy`, and `probability`.

Method description and citation: [ICD/AICD method paper](citation.md#icd-aicd-method-paper) (DOI [10.5281/zenodo.20581077](https://doi.org/10.5281/zenodo.20581077)). Plug-in guide: [plugin_icd.md](plugin_icd.md).

### Detection presets

```python
from bnnr.detection_augmentations import get_detection_preset

augmentations = get_detection_preset("standard")  # "light", "standard", "aggressive"
```

See [detection.md](detection.md) for detailed parameter tables and usage examples.

## Custom augmentation registration

Register subclasses of `BaseAugmentation` in `AugmentationRegistry` and keep deterministic behavior via `random_state` where relevant.
