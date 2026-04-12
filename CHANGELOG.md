# Changelog

## [0.2.9] — 2026-04-12

### Added

- **CI**: `notebooks-smoke` job and [scripts/validate_user_notebooks.py](scripts/validate_user_notebooks.py) — validates catalog notebooks (nbformat, kernelspec, syntax of code cells after stripping magics) without executing training.

### Changed

- **Reporting**: atomic write for `report.json`; `load_report` validates JSON shape and required keys, with clearer `ValueError` messages and `ValidationError` wrapping for config.
- **Dashboard**: `start_dashboard` returns `None` when optional dashboard dependencies are missing instead of a misleading localhost URL; CLI and examples handle the degraded case.
- **Pipelines**: unknown augmentation preset logs a `UserWarning` in addition to the existing logger warning when falling back to `auto`.
- **Training**: custom metric failures are always logged with `logger.warning(..., exc_info=True)` instead of only when `verbose` is enabled.
- **Dashboard API**: invalid JSONL lines in the paginated events endpoint are logged with `logger.warning`.
- **Docs / CLI**: `docs/cli.md` aligned with supported datasets and presets; related doc updates for `start_dashboard` return value.
- **CI / dev**: comment on lint vs pytest matrix; pre-commit pins updated (ruff, mypy) closer to dev tooling; `build-package` waits on `notebooks-smoke`.

## [0.2.8] — 2026-04-10

### Changed

- **Tests vs examples**: Removed YOLO ``smoke_*.py`` scripts from ``examples/detection/`` (they duplicated CI tests). Training compatibility is covered only by ``tests/test_ultralytics_detection_train_integration.py``. The optional XAI overlay helper lives under ``tests/integration/smoke_yolo_xai_overlay.py``.
- **CI / YOLO26**: On ``GITHUB_ACTIONS``, the ``yolo26_cpu_adapter`` fixture **fails** the job if ``ultralytics`` is older than 8.3 or ``yolo26n.pt`` cannot be loaded, so YOLO26 + BNNR is always exercised on CI (locally, missing weights still **skip**). Tests ``test_yolo26_*`` are marked ``@pytest.mark.yolo26``.

### Added

- ``test_yolo26_multiple_train_steps_finite`` — eight consecutive ``train_step`` calls on ``yolo26n.pt``.

## [0.2.7] — 2026-04-10

### Fixed

- **Ultralytics YOLO26 / ``E2ELoss``** (`UltralyticsDetectionAdapter`): before ``loss()``, the task model and last (detection) module are forced into **train mode** so the head returns a ``one2many`` dict with ``boxes`` (inference mode previously led to ``KeyError`` / wrong layout). **Fused** heads (``cv2`` / ``cv3`` cleared after ``fuse()``) are detected early with ``loss_yolo_fused_head``; ``KeyError`` / ``TypeError`` from the criterion are caught as ``loss_yolo_pred_format_error`` with logs including the installed ``ultralytics`` version and head state.
- **Detection XAI / probes**: ``predict_detection_dicts`` restores the prior ``training`` flag after ``eval`` + ``predict``, so interleaved inference cannot leave the YOLO task stuck in ``eval()`` for the next training step.

### Added

- **Tests**: optional ``yolo26n.pt`` integration step (skipped when Ultralytics is older than 8.3 or weights are unavailable; **tightened in 0.2.8** to fail on CI).

### Changed

- **Dev dependency**: ``ultralytics>=8.3.0,<9`` (YOLO26 / current loss paths).

## [0.2.6] — 2026-04-10

### Fixed

- **Detection XAI (Ultralytics)**: `generate_detection_saliency` now scores every eligible `Conv2d` map by **isotropy** of the channel-mean saliency (downsampled), then re-runs a forward pass to capture the best layer. Taking only the chronologically last `H,W>1` conv could still yield strong vertical/horizontal “barcode” structure after upsampling; the score `min(column-wise std, row-wise std)` prefers maps that actually vary in **both** axes.

## [0.2.5] — 2026-04-10

### Fixed

- **CI**: detection XAI regression test no longer triggers Ruff `F841` (use column-wise std assert for 2D saliency vs vertical-stripe artifacts).

## [0.2.4] — 2026-04-10

### Fixed

- **Detection XAI (Ultralytics)**: `generate_detection_saliency` no longer uses the final `Conv2d` when its feature map has **height or width 1** (common in YOLO heads). Resizing such maps to the image size produced vertical/horizontal “barcode” heatmaps; the code now keeps the **last 4D conv feature map with both spatial sizes > 1**, with a safe fallback to `target_layers` hooks.

## [0.2.3] — 2026-04-10

### Fixed

- **Ultralytics YOLO** (`UltralyticsDetectionAdapter`): detection images stay in **`[0, 1]`** end-to-end (uint8 round-trip in `BNNRTrainer` always `/255`); `predict` passes **float tensors in `[0, 1]`** (Ultralytics does not divide torch tensors by 255). `train_step` uses **fp32 loss**, **plain backward** (no `GradScaler` on this path), **grad clipping**, **cls clamped to `[0, nc-1]`**, degenerate bbox filtering, **`IndexError`** from assigner caught.
- **Imports**: lazy `sklearn` in `calculate_metrics` and lazy NMF in CRAFT so **`import bnnr`** works when Colab’s numpy/scipy/sklearn stack is temporarily broken until `pip install -U` aligns versions.

## [0.2.2] — 2026-04-10

### Fixed

- **Detection training**: `_average_metrics` no longer assumes every batch returns the same metric keys. Skipped batches (non-finite loss in `DetectionAdapter`) returned only `loss` + `loss_non_finite`, which caused `KeyError: 'loss_loss_classifier'` when averaging epoch metrics (e.g. Colab detection notebook).
- **Ultralytics YOLO**: training images are scaled to Ultralytics’ 0–255 float range only when the batch is already in ~0–1 space. Feeding 0–255 tensors no longer multiplies by 255 again (avoids overflow, NaN loss, and the “divide by 255” warning storm). Non-finite loss batches are skipped like torchvision detection.
- **Dashboard (React)**: `task: "detection"` from the run state is honored — KPIs and charts use `map_50` / `map_50_95` instead of treating detection as classification (which led to empty charts, NaNs, or a blank UI when combined with bad loss data).

## [0.2.1] — 2026-04-09

### Fixed

- **PyPI publish**: remove redundant `wheel.force-include` for dashboard frontend dist that caused duplicate ZIP entries rejected by PyPI.
- **Security**: bump vite 5.4 to 6.4.2 (CVE path-traversal in optimized deps .map handling).
- **Testing**: add regression test for detection data quality analysis silent failure.

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
