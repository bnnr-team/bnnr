# Notebooks Guide (Production + Colab)

## Scope
This guide covers all user notebooks in `examples/` and how to run them in a way that is reproducible, useful, and dashboard-first.

## Notebook Catalog

| Notebook | Goal | Dashboard usage | Typical runtime |
|---|---|---|---|
| `examples/bnnr_augmentations_guide.ipynb` | Visualize all built-in augmentations and ICD/AICD behavior | Optional (focus is augmentation visuals) | Short |
| `examples/classification/bnnr_classification_demo.ipynb` | End-to-end STL-10 classification with XAI and branch selection | Required (live tracking) | Medium |
| `examples/multilabel/bnnr_multilabel_demo.ipynb` | Multi-label training flow (`task="multilabel"`) | Required (live tracking) | Medium |
| `examples/bnnr_custom_data.ipynb` | Bring-your-own classification data patterns | Recommended | Medium |
| `examples/detection/bnnr_detection_demo.ipynb` | Object detection training with bbox augmentations, mAP metrics, and XAI | Required (live tracking) | Medium |

## Recommended order
1. `bnnr_augmentations_guide.ipynb`
2. `classification/bnnr_classification_demo.ipynb`
3. `multilabel/bnnr_multilabel_demo.ipynb`
4. `detection/bnnr_detection_demo.ipynb`
5. `bnnr_custom_data.ipynb`

This order gives fastest understanding: augmentations -> core training loop -> task variants (classification, multilabel, detection) -> custom integration.

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
2. Runtime -> Change runtime type -> GPU (recommended for larger runs).
3. Run installation cell first (`%pip install -q "bnnr[dashboard]" ...`).
4. Run cells top-to-bottom without skipping.

## Dashboard-first workflow (desktop + mobile)
For classification/multilabel notebooks:
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
  - detection: `map_50`, `map_50_95`,

Optional replay/export checks:

```bash
python3 -m bnnr dashboard serve --run-dir <run_dir_parent> --port 8080
python3 -m bnnr dashboard export --run-dir <run_dir> --out /tmp/exported_dashboard
```

## CI (GitHub Actions)

The workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) job **notebooks-smoke** runs on every push and pull request. It does **not** execute training; it runs [scripts/validate_user_notebooks.py](../scripts/validate_user_notebooks.py), which checks that each catalog notebook exists, is valid nbformat v4 JSON, uses the `python3` kernelspec, and that every code cell **compiles** as Python after stripping line magics (`%pip`, `!`, etc.). That catches broken syntax and drift in the notebook list before users open Colab or Jupyter.

Locally:

```bash
python3 scripts/validate_user_notebooks.py
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
- Forgetting to keep `events.jsonl` (required for replay/export).

For concrete fixes, see `troubleshooting.md`.
