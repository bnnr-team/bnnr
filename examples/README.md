<p align="center">
  <img src="../docs/assets/logo.png" alt="BNNR Logo" width="160">
</p>

# BNNR Examples

Practical entrypoint for runnable examples and notebooks.

Use this with:
- `docs/examples.md` for script-by-script commands,
- `docs/notebooks.md` for notebook execution,
- `docs/dashboard.md` for live/mobile/QR workflow.

## Python examples (`.py`)

### Classification

- `classification/showcase_stl10.py`

```bash
# Full showcase + dashboard
PYTHONPATH=src python3 examples/classification/showcase_stl10.py --with-dashboard

# Fast smoke
PYTHONPATH=src python3 examples/classification/showcase_stl10.py \
  --without-dashboard --no-dashboard-auto-open \
  --max-train-samples 32 --max-val-samples 16 --batch-size 16 \
  --m-epochs 1 --decisions 1
```

### Multi-label

- `multilabel/multilabel_demo.py`

```bash
# Full demo + dashboard
PYTHONPATH=src python3 examples/multilabel/multilabel_demo.py --with-dashboard

# Fast smoke
PYTHONPATH=src python3 examples/multilabel/multilabel_demo.py \
  --without-dashboard --no-dashboard-auto-open \
  --n-train 64 --n-val 32 --batch-size 16 --m-epochs 1 --decisions 1
```

### Detection

- `detection/showcase_voc.py`
- `detection/showcase_yolo_coco128.py`

```bash
# VOC showcase + dashboard
PYTHONPATH=src python3 examples/detection/showcase_voc.py --with-dashboard

# VOC tiny smoke (baseline-only completion)
PYTHONPATH=src python3 examples/detection/showcase_voc.py \
  --without-dashboard --no-dashboard-auto-open \
  --max-train-samples 4 --max-val-samples 2 --batch-size 1 \
  --m-epochs 1 --decisions 0 --target-size 128

# YOLO showcase + dashboard (COCO128 auto-downloads if data/coco128/data.yaml is missing)
PYTHONPATH=src python3 examples/detection/showcase_yolo_coco128.py \
  --data-path data/coco128/data.yaml --with-dashboard
```

## Notebooks (`.ipynb`)

- `bnnr_augmentations_guide.ipynb`
- `bnnr_custom_data.ipynb`
- `classification/bnnr_classification_demo.ipynb`
- `detection/bnnr_detection_demo.ipynb`
- `multilabel/bnnr_multilabel_demo.ipynb`

See `docs/notebooks.md` for local and CI-style execution (`jupyter nbconvert --execute`).

## Dashboard-first validation loop

For each example run:
1. Confirm `report.json` and `events.jsonl` exist.
2. Open dashboard (local URL).
3. Check mobile via QR code (same network).
4. Export static dashboard and open `index.html`.

```bash
python3 -m bnnr dashboard export --run-dir <run_dir> --out exported_dashboard
```
