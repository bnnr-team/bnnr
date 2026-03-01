# Notebooks Guide (Production + Colab)

## Scope
This guide covers all user notebooks in `examples/` and how to run them in a way that is reproducible, useful, and dashboard-first.

## Notebook Catalog

| Notebook | Goal | Dashboard usage | Typical runtime |
|---|---|---|---|
| `examples/bnnr_augmentations_guide.ipynb` | Visualize all built-in augmentations and ICD/AICD behavior | Optional (focus is augmentation visuals) | Short |
| `examples/classification/bnnr_classification_demo.ipynb` | End-to-end STL-10 classification with XAI and branch selection | Required (live tracking) | Medium |
| `examples/multilabel/bnnr_multilabel_demo.ipynb` | Multi-label training flow (`task="multilabel"`) | Required (live tracking) | Medium |
| `examples/detection/bnnr_detection_demo.ipynb` | VOC detection flow with bbox-aware augmentations + detection XAI | Required (live tracking) | Long |
| `examples/bnnr_custom_data.ipynb` | Bring-your-own classification/detection data patterns | Recommended | Medium |

## Phase C Benchmark Notebooks (`benchmarks/notebooks/phase_c/`)

| Notebook | Goal | Runtime usage | Typical runtime |
|---|---|---|---|
| `00_setup_private_wheel.ipynb` | Mount Drive, install private wheel, unpack benchmark bundle, verify files | Required first step | Short |
| `01_eurosat.ipynb` | Run EuroSAT benchmark matrix (4 variants x seeds) with resume | Benchmark execution | Medium |
| `02_isic2019.ipynb` | Run ISIC benchmark matrix with Kaggle-first ingest + ZIP fallback | Benchmark execution | Long |
| `03_voc2007_yolo.ipynb` | Run VOC2007 detection benchmark (YOLOv8-only) with resume | Benchmark execution | Long |
| `04_aggregate_results.ipynb` | Validate produced JSON/MD summaries for README/paper tables | Post-processing | Short |

## Recommended order
1. `bnnr_augmentations_guide.ipynb`
2. `classification/bnnr_classification_demo.ipynb`
3. `multilabel/bnnr_multilabel_demo.ipynb`
4. `detection/bnnr_detection_demo.ipynb`
5. `bnnr_custom_data.ipynb`

This order gives fastest understanding: augmentations -> core training loop -> task variants -> custom integration.

## Local setup

```bash
python3 -m venv /tmp/bnnr-nb-venv
source /tmp/bnnr-nb-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dashboard]"
python -m pip install jupyter nbconvert
```

Run:

```bash
jupyter lab
```

## Colab setup (recommended for first run)
1. Open notebook via the “Open in Colab” badge.
2. Runtime -> Change runtime type -> GPU (recommended for detection).
3. Run installation cell first (`%pip install -q "bnnr[dashboard]" ...`).
4. Run cells top-to-bottom without skipping.

## Dashboard-first workflow (desktop + mobile)
For classification/multilabel/detection notebooks:
1. Run dashboard section **before** training.
2. Confirm local dashboard URL appears.
3. On Colab, confirm iframe is rendered.
4. Optional mobile/public tracking from Colab:
   - set `NGROK_AUTHTOKEN`,
   - use the provided `pyngrok` URL,
   - open it on phone.

What to verify while training:
- branch tree updates,
- KPI trend updates,
- samples/XAI section renders,
- `events.jsonl` grows in `report_dir`.

## Validation checklist (per notebook)
After each run verify:
- no traceback in output cells,
- `report.json` exists,
- `events.jsonl` exists,
- task metrics are present:
  - classification: `accuracy`, `f1_*`,
  - multilabel: `f1_samples`, `fbeta_*`,
  - detection: `map_50`, `map_50_95`.

Optional replay/export checks:

```bash
python3 -m bnnr dashboard serve --run-dir <run_dir_parent> --port 8080
python3 -m bnnr dashboard export --run-dir <run_dir> --out /tmp/exported_dashboard
```

## Quality notes from notebook audit
During B11 notebook hardening, the following were normalized:
- valid Colab links to `mateuszwalo/BNNR`,
- modern CLI examples (`python3 -m bnnr ...`, no legacy `bnnr.cli`),
- kernelspec metadata set to `python3`,
- dashboard cells aligned for local + Colab + optional mobile flow,
- artifact preview paths made compatible with current output layout.

## Common pitfalls
- Missing `jupyter`/`nbconvert` locally.
- Expecting full detection runs to finish quickly on CPU.
- Confusing YOLO `data.yaml` with BNNR config YAML.
- Forgetting to keep `events.jsonl` (required for replay/export).

For concrete fixes, see `troubleshooting.md`.
