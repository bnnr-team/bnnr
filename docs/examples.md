# Examples Guide (Production Usage)

## What you will find here
A practical guide for all `examples/*.py` scripts with:
- what each script demonstrates,
- exact run commands,
- which dashboard flow to use,
- smoke commands for fast verification.

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

## 3) Detection VOC showcase

Script:
- `examples/detection/showcase_voc.py`

What it demonstrates:
- detection branch-selection loop,
- bbox-aware augmentations,
- detection metrics (`map_50`, `map_50_95`),
- detection dashboard integration.

### Full showcase

```bash
PYTHONPATH=src python3 examples/detection/showcase_voc.py --with-dashboard
```

### Fast smoke (baseline-only completion)

```bash
PYTHONPATH=src python3 examples/detection/showcase_voc.py \
  --without-dashboard --no-dashboard-auto-open \
  --max-train-samples 4 --max-val-samples 2 --batch-size 1 \
  --m-epochs 1 --decisions 0 --target-size 128
```

## 4) Detection YOLO showcase

Script:
- `examples/detection/showcase_yolo_coco128.py`

What it demonstrates:
- YOLO `data.yaml` ingestion,
- detection augmentations + ICD/AICD,
- dashboard-compatible detection run artifacts.

Important:
- `--data-path` must point to YOLO `data.yaml` (not BNNR config YAML).

### Full showcase

```bash
PYTHONPATH=src python3 examples/detection/showcase_yolo_coco128.py \
  --data-path data/coco128/data.yaml --with-dashboard
```

### Fast smoke

```bash
PYTHONPATH=src python3 examples/detection/showcase_yolo_coco128.py \
  --without-dashboard --no-dashboard-auto-open \
  --data-path /tmp/b11_tiny_yolo/data.yaml \
  --batch-size 2 --max-train-samples 8 --max-val-samples 4 --quick
```

## 5) Dashboard workflow for examples

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

## 6) Example artifacts you should always verify

After each example run, verify:
- `report.json` exists,
- `events.jsonl` exists,
- metrics are present for task type,
- dashboard replay works (`bnnr dashboard serve --run-dir ...`).

## 7) Related docs

- `dashboard.md`
- `artifacts.md`
- `troubleshooting.md`
- `notebooks.md`

## 8) Phase C Benchmark Commands (Colab / production)

Single run (resume-safe):

```bash
python -m benchmarks.phase_c.run_phase_c \
  --dataset eurosat \
  --variant bnnr_auto \
  --seed 42 \
  --config benchmarks/configs/phase_c/default.yaml \
  --resume
```

Full matrix per dataset:

```bash
python -m benchmarks.bench_eurosat --config benchmarks/configs/phase_c/default.yaml --resume
python -m benchmarks.bench_isic2019 --config benchmarks/configs/phase_c/default.yaml --resume
python -m benchmarks.bench_voc_detection --config benchmarks/configs/phase_c/default.yaml --resume
```

Dry-run smoke (no dataset download):

```bash
python -m benchmarks.phase_c.run_phase_c \
  --dataset eurosat \
  --variant no_aug \
  --seed 42 \
  --config benchmarks/configs/phase_c/quick.yaml \
  --resume \
  --dry-run
```
