# Changelog

## [0.2.0] — 2026-04-08

### Added — Object Detection

- **Object detection task** (`task: detection`) — full training, evaluation, and XAI pipeline for detection models.
- **Detection adapters**: `DetectionAdapter` (torchvision Faster R-CNN / FCOS / RetinaNet / SSD) and `UltralyticsDetectionAdapter` (YOLO v8).
- **Detection augmentations**: bbox-aware transforms (`DetectionHorizontalFlip`, `DetectionVerticalFlip`, `DetectionRandomRotate90`, `DetectionRandomScale`), `MosaicAugmentation`, `DetectionMixUp`, Albumentations bbox integration, preset system (`get_detection_preset`).
- **Detection ICD/AICD**: iterative dataset curation for detection using bounding-box saliency priors with multiple fill strategies (gaussian blur, noise, global/local mean, solid).
- **Detection metrics**: mAP@0.5, mAP@[.50:.95], per-class AP (`calculate_per_class_ap`), detection confusion matrix; falls back to built-in implementation when torchmetrics is unavailable.
- **Detection XAI**: activation-based saliency maps (`generate_detection_saliency`), occlusion sensitivity (`compute_detection_box_saliency_occlusion`), three-panel visualization (GT overlay, saliency heatmap, prediction overlay).
- **Pipelines**: `yolo` and `coco_mini` built-in dataset pipelines for detection quickstart.
- **Dashboard**: detection triptych XAI panels in HTML standalone export.
- **Events**: detection metric units, split XAI artifact keys (`xai_gt`, `xai_saliency`, `xai_pred`), `detection_details` in sample prediction snapshots.
- Example configs, scripts, and notebook for detection (`examples/detection/`).

### Fixed

- `is_detection` variable missing in `_standalone_report_html` (dashboard exporter crash).
- mypy errors in detection code paths (`core.py` `predict_ultra` narrowing, `forward_layout` Literal type, removed invalid `is_detection` kwarg in `run_data_quality_analysis` call).

## [0.1.2] — 2026-04-04

- Dashboard (npm): `lodash` forced to **^4.18.0** via `package.json` `overrides` (Dependabot GHSA-f23m-r3pf-42rh, GHSA-r5fr-rjxr-66jc); regenerated `package-lock.json`.
- Align `__version__`, FastAPI dashboard `version`, README, and docs with release **0.1.2** (Python **>=3.10** in user-facing docs).

## [0.1.1] — 2026-04-04

- Security: `constraint-dependencies` in `pyproject.toml` (uv) — `filelock>=3.20.3`, `pillow>=12.1.1`, `pygments>=2.20.0` to address Dependabot alerts; regenerated `uv.lock`.
- **Breaking:** minimum Python is now **3.10** (required for patched Pillow 12.x).

## [0.1.0.post4] — 2026-04-01

- CI: run `npm ci` + Vite build **before every** `pip install -e ".[dev,dashboard]"` so `src/bnnr/dashboard/frontend/dist` exists; hatch `force-include` no longer breaks editable installs on lint/test jobs.

## [0.1.0.post3] — 2026-04-01

- Dashboard: production Vite build is **bundled in the PyPI wheel** under `bnnr/dashboard/frontend/dist`, so `pip install "bnnr[dashboard]"` can serve the full React UI without cloning the repo.
- `dashboard_web` build output targets that path; CI runs `npm ci` + `npm run build` before `python -m build` so releases always include the frontend.

## [0.1.0.post2] — 2026-04-01

Post-release: nowe artefakty na PyPI (`bnnr-0.1.0.post2-*`) — unika konfliktu z już wgranymi plikami `0.1.0`. Zawartość funkcjonalna jak 0.1.0.

## [0.1.0] — 2026-04-01

### Initial Release (PyPI)

- BNNR core training loop — automated augmentation evaluation and selection
- Seven custom augmentations: ChurchNoise, Drust, LuxferGlass, ProCAM, Smugs, TeaStains, BasicAugmentation
- External augmentation integration: Albumentations, Kornia, torchvision
- XAI: OptiCAM, GradCAM saliency maps
- Advanced XAI: CRAFT, RealCRAFT, RecursiveCRAFT, NMF concept decomposition
- ICD / AICD — XAI-driven intelligent augmentation
- Live dashboard (React + FastAPI + WebSocket)
- CLI: train, report, dashboard, list-augmentations, list-presets, list-datasets
- Config system (YAML/JSON) with presets: auto, standard, aggressive, light, gpu
- Data quality helpers (near-duplicate hashing, basic image sanity checks)
- **Tasks:** single-label classification, multi-label classification, and **object detection** (this branch)
- Checkpointing and training resume
- Event system (JSONL) for training replay
