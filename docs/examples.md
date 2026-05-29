# Examples Guide (Production Usage)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## What you will find here
A practical guide for Python scripts under `examples/` (for example `examples/classification/`, `examples/multilabel/`) with:
- what each script demonstrates,
- exact run commands,
- which dashboard flow to use,
- smoke commands for fast verification.

These paths assume a **cloned repository** with installs from source (`pip install -e ".[…]"`). They are not shipped inside the PyPI wheel; see the root [README.md](../README.md).

## 1) Classification showcase

Script:
- `examples/classification/showcase_stl10.py`

What it demonstrates:
- iterative augmentation selection,
- XAI-driven candidates (ICD/AICD),
- full live dashboard flow.

### Full showcase

```bash
PYTHONPATH=src python3 examples/classification/showcase_stl10.py --with-dashboard
```

### Fast smoke (CI/dev machine)

```bash
PYTHONPATH=src python3 examples/classification/showcase_stl10.py \
  --without-dashboard --no-dashboard-auto-open \
  --max-train-samples 32 --max-val-samples 16 --batch-size 16 \
  --m-epochs 1 --decisions 1
```

**First run:** STL-10 is downloaded automatically (needs network). For a shorter interactive run, the script supports `--quick`; see the module docstring for timings and GPU/CPU notes.

## 2) Multi-label showcase

Script:
- `examples/multilabel/multilabel_demo.py`

What it demonstrates:
- multi-label pipeline (`task="multilabel"`),
- F1-samples oriented selection,
- dashboard-compatible events and artifacts.

### Full demo

```bash
PYTHONPATH=src python3 examples/multilabel/multilabel_demo.py --with-dashboard
```

### Fast smoke

```bash
PYTHONPATH=src python3 examples/multilabel/multilabel_demo.py \
  --without-dashboard --no-dashboard-auto-open \
  --n-train 64 --n-val 32 --batch-size 16 --m-epochs 1 --decisions 1
```

## 3) Classification: full pipeline demo (synthetic)

Script:
- `examples/classification/demo_full_pipeline.py`

What it demonstrates:
- all built-in BNNR augmentations and optional torchvision / Kornia / Albumentations wrappers (when those extras are installed),
- ICD/AICD, XAI cache, OptiCAM, and a short branch-selection loop on a tiny synthetic dataset (no dataset download).

Run from repository root (no live dashboard; typically well under a minute on CPU):

```bash
PYTHONPATH=src python3 examples/classification/demo_full_pipeline.py
```

Optional: install `albumentations` and/or `kornia` extras (`pip install -e ".[albumentations]"`, `pip install -e ".[gpu]"`) so the corresponding wrapper branches execute.

## 4) Lightning / Accelerate adapters (reference module)

File:
- `examples/classification/lightning_adapter.py`

This file is **reference code**, not a `main` entrypoint: it defines `LightningAdapter` and `AccelerateAdapter` for use with `BNNRTrainer`. PyTorch Lightning and Hugging Face Accelerate are optional dependencies:

```bash
pip install pytorch-lightning accelerate
```

See the module docstring and inline comments for wiring; also [api_reference.md](api_reference.md).

## 5) Detection showcases

### 5a) Ultralytics YOLO (SDK)

Scripts:
- `examples/integrations/ultralytics_yolo_quickstart.py` — copy-paste CLI for maintainers
- `examples/detection/bnnr_detection_demo.ipynb` — interactive COCO128 + YOLOv8n

What they demonstrate:
- `UltralyticsDetectionAdapter` (not `yolo train` CLI),
- COCO128 auto-download via `bnnr.example_data.ensure_coco128_yolo`,
- bbox-aware augmentations + DetectionICD/AICD,
- mAP metrics and optional dashboard.

```bash
pip install "bnnr[ultralytics]"
PYTHONPATH=src python examples/integrations/ultralytics_yolo_quickstart.py --quick
```

See also [integrations.md](integrations.md).

### 5b) YOLO-format data + torchvision detector

Script:
- `examples/detection/showcase_yolo_coco128.py`

What it demonstrates:
- **YOLO dataset layout** (COCO128 `data.yaml`) with `build_pipeline("yolo")` → **torchvision Faster R-CNN** (not Ultralytics SDK),
- detection-specific bbox augmentations + DetectionICD/AICD,
- mAP tracking and dashboard.

```bash
PYTHONPATH=src python3 examples/detection/showcase_yolo_coco128.py --with-dashboard
```

Fast smoke:

```bash
PYTHONPATH=src python3 examples/detection/showcase_yolo_coco128.py --quick --without-dashboard
```

### Pascal VOC 2007

Script:
- `examples/detection/showcase_voc.py`

What it demonstrates:
- torchvision-style detector with `DetectionAdapter`,
- full augmentation suite (all 9 augmentation families),
- XAI saliency generation,
- long training runs with dashboard.

```bash
PYTHONPATH=src python3 examples/detection/showcase_voc.py --with-dashboard
```

### Detection notebook

- `examples/detection/bnnr_detection_demo.ipynb` — interactive Jupyter notebook covering adapter setup, augmentations, training, and XAI visualization.

## 6) Dashboard workflow for examples

For any example with `--with-dashboard`:
1. Start script.
2. Open Local URL on desktop.
3. Scan QR for mobile view.
4. Validate branch tree, KPI cards, samples/XAI sections.
5. Stop server with `Ctrl+C` after checks.

For offline sharing:

```bash
python3 -m bnnr dashboard export --run-dir <run_dir> --out exported_dashboard
```

## 7) Example artifacts you should always verify

After each example run, verify:
- `report.json` exists,
- `events.jsonl` exists,
- metrics are present for task type,
- dashboard replay works (`bnnr dashboard serve --run-dir ...`).

## 8) Related docs

- `dashboard.md`
- `artifacts.md`
- `troubleshooting.md`
- `notebooks.md`
- `detection.md`
