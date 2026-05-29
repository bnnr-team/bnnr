# Integrations — pytorch-grad-cam, Ultralytics YOLO, and reference adapters

BNNR integrates with popular PyTorch vision stacks as an **augmentation and analysis layer**, not as a replacement training framework.

## Overview

| Integration | Status | Install | Entry point | Example |
|-------------|--------|---------|-------------|---------|
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | **Supported** (core dependency) | `pip install bnnr` | ICD/AICD in your training loop | [plugin_icd.md](plugin_icd.md), [gradcam_to_icd_loop.py](../examples/integrations/gradcam_to_icd_loop.py) |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | **Supported** (adapter) | `pip install "bnnr[ultralytics]"` | `UltralyticsDetectionAdapter` | [detection.md](detection.md), [ultralytics_yolo_quickstart.py](../examples/integrations/ultralytics_yolo_quickstart.py) |
| PyTorch Lightning | Reference | `pip install pytorch-lightning` | `LightningAdapter` | [lightning_adapter.py](../examples/classification/lightning_adapter.py) |
| Hugging Face Accelerate | Reference | `pip install accelerate` | `AccelerateAdapter` | [lightning_adapter.py](../examples/classification/lightning_adapter.py) |

Runnable index: [examples/integrations/README.md](../examples/integrations/README.md).

## Positioning

**pytorch-grad-cam** answers *where the model looks*. BNNR adds **ICD/AICD** — augmentations that mask salient or background regions using those maps — plus optional `BNNRTrainer` branch search and `bnnr analyze` HTML reports. BNNR is not a second Grad-CAM library; it builds on `grad-cam` (see [plugin_icd.md](plugin_icd.md)).

**Ultralytics YOLO** provides training and deployment via `yolo train` / `YOLO.predict`. BNNR wraps YOLO with [`UltralyticsDetectionAdapter`](../src/bnnr/detection_adapter.py) for **bbox-aware augmentation search**, detection XAI, and structured reports — alongside your YOLO workflow, not instead of it.

## Stable URLs for maintainers

Use these links in upstream docs or issues (replace `main` with a release tag when pinning versions):

| Asset | URL |
|-------|-----|
| ICD plug-in guide | https://github.com/bnnr-team/bnnr/blob/main/docs/plugin_icd.md |
| Grad-CAM → ICD example | https://github.com/bnnr-team/bnnr/blob/main/examples/integrations/gradcam_to_icd_loop.py |
| Ultralytics quickstart | https://github.com/bnnr-team/bnnr/blob/main/examples/integrations/ultralytics_yolo_quickstart.py |
| This hub | https://github.com/bnnr-team/bnnr/blob/main/docs/integrations.md |
| Detection guide | https://github.com/bnnr-team/bnnr/blob/main/docs/detection.md |

## License note

BNNR is **MIT**. The [Ultralytics](https://github.com/ultralytics/ultralytics) repository is **AGPL-3.0**; using `pip install ultralytics` in your project is separate from forking Ultralytics. This documentation describes an optional adapter pattern only.
