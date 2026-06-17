# Changelog

## [0.6.1](https://github.com/bnnr-team/bnnr/compare/v0.6.0...v0.6.1) (2026-06-17)


### Bug Fixes

* **deps:** require starlette&gt;=1.3.1 to patch four CVEs ([#323](https://github.com/bnnr-team/bnnr/issues/323)) ([22cbad2](https://github.com/bnnr-team/bnnr/commit/22cbad2c9ee1f998b0c109b0acb1cdfc046aba6d))

## [0.6.0](https://github.com/bnnr-team/bnnr/compare/v0.5.7...v0.6.0) (2026-06-16)


### Features

* add icd preset for saliency-guided training ([#303](https://github.com/bnnr-team/bnnr/issues/303)) ([ba7ba4d](https://github.com/bnnr-team/bnnr/commit/ba7ba4d212c03ca15dea671d7d25949342a49d86))
* analyze progress output and CV cached-pred reuse ([e7d6364](https://github.com/bnnr-team/bnnr/commit/e7d636443b556921eecbe3452c35d670a9ba144f))
* bound the XAI cache and stop unbounded hash-keyed growth ([1be85ac](https://github.com/bnnr-team/bnnr/commit/1be85ac97de2d4e1d0a9e9594096acee598b0b03))
* dashboard port auto-fallback, train --dry-run, config warning echo ([309d1eb](https://github.com/bnnr-team/bnnr/commit/309d1ebbaa69bedb41e0e041a38249d4ffad6597))
* warn once when branch search projects a very long runtime ([#318](https://github.com/bnnr-team/bnnr/issues/318)) ([0231591](https://github.com/bnnr-team/bnnr/commit/02315916aca57ee681bb025b0b98dd65b3a8efd4))


### Bug Fixes

* apply augmentations in list order and pass async sample indices ([76ea8d9](https://github.com/bnnr-team/bnnr/commit/76ea8d9179a1a59a49e722c728ce385552cef352))
* invalidate XAI cache when a different model shares the cache dir ([#317](https://github.com/bnnr-team/bnnr/issues/317)) ([202e258](https://github.com/bnnr-team/bnnr/commit/202e25822bf5b1adc320a1fefabbc7925c7e6e8b))


### Performance Improvements

* lazy-import cv2 so import bnnr no longer pays the OpenCV cost ([85f0689](https://github.com/bnnr-team/bnnr/commit/85f0689db1804685819ff095e661305c99cb1660))

## [0.5.7](https://github.com/bnnr-team/bnnr/compare/v0.5.6...v0.5.7) (2026-06-12)


### Bug Fixes

* select the best baseline epoch, not the last ([#301](https://github.com/bnnr-team/bnnr/issues/301)) ([95b9d8d](https://github.com/bnnr-team/bnnr/commit/95b9d8d24cb511ef802e76d3415eb89a0a2865f9))

## [0.5.6](https://github.com/bnnr-team/bnnr/compare/v0.5.5...v0.5.6) (2026-06-12)


### Bug Fixes

* make analyze sequential for shuffled val loaders ([#299](https://github.com/bnnr-team/bnnr/issues/299)) ([f204f29](https://github.com/bnnr-team/bnnr/commit/f204f29451d699a6dea4cb78b5a8db94b293ed57))

## [0.5.5](https://github.com/bnnr-team/bnnr/compare/v0.5.4...v0.5.5) (2026-06-12)


### Bug Fixes

* forbid unknown config keys and enforce epoch bounds ([#297](https://github.com/bnnr-team/bnnr/issues/297)) ([985dcc8](https://github.com/bnnr-team/bnnr/commit/985dcc87c3ce5dff2ac4c7a2a80091a3e376df5a))

## [0.5.4](https://github.com/bnnr-team/bnnr/compare/v0.5.3...v0.5.4) (2026-06-11)


### Bug Fixes

* load checkpoints with weights_only=True first ([#298](https://github.com/bnnr-team/bnnr/issues/298)) ([233dc36](https://github.com/bnnr-team/bnnr/commit/233dc36cfcc201076ff4eac9b683620dea4221e9))

## [0.5.3](https://github.com/bnnr-team/bnnr/compare/v0.5.2...v0.5.3) (2026-06-11)


### Bug Fixes

* implement real local_mean fill in DetectionICD ([#295](https://github.com/bnnr-team/bnnr/issues/295)) ([8b72635](https://github.com/bnnr-team/bnnr/commit/8b7263583cda364713a796acb6846d615e9a6070)), closes [#255](https://github.com/bnnr-team/bnnr/issues/255)

## [0.5.2](https://github.com/bnnr-team/bnnr/compare/v0.5.1...v0.5.2) (2026-06-11)


### Bug Fixes

* raise torch floor to 2.10.0 for security fixes ([#293](https://github.com/bnnr-team/bnnr/issues/293)) ([76b52bc](https://github.com/bnnr-team/bnnr/commit/76b52bc31425f017fe04dbddb2cd78c013393d12))

## [0.5.1](https://github.com/bnnr-team/bnnr/compare/v0.5.0...v0.5.1) (2026-06-10)


### Bug Fixes

* keep canonical augmentation names instead of indexed aliases ([#289](https://github.com/bnnr-team/bnnr/issues/289)) ([8e285f9](https://github.com/bnnr-team/bnnr/commit/8e285f965585ea45fbb84678cd61c4c077ab4e5c)), closes [#258](https://github.com/bnnr-team/bnnr/issues/258)

## [0.5.0](https://github.com/bnnr-team/bnnr/compare/v0.4.16...v0.5.0) (2026-06-10)


### Features

* implement real Opti-CAM and split out GradCAMExplainer ([#287](https://github.com/bnnr-team/bnnr/issues/287)) ([385aadd](https://github.com/bnnr-team/bnnr/commit/385aaddb717f7a325645cc9a9655fa9581c9c302))

## [0.4.16](https://github.com/bnnr-team/bnnr/compare/v0.4.15...v0.4.16) (2026-06-10)


### Bug Fixes

* make ICD/AICD CPU-bound to remove apply_tensor recursion ([#285](https://github.com/bnnr-team/bnnr/issues/285)) ([69435d6](https://github.com/bnnr-team/bnnr/commit/69435d68c9de17cd3d6bc9f4acbf988f94586881))

## [0.4.15](https://github.com/bnnr-team/bnnr/compare/v0.4.14...v0.4.15) (2026-06-10)


### Bug Fixes

* treat winning baseline_reeval as no improvement instead of crashing ([#279](https://github.com/bnnr-team/bnnr/issues/279)) ([6ae1512](https://github.com/bnnr-team/bnnr/commit/6ae151226989fd19bd4bc9a79c6b292973948b8a))


### Documentation

* add branded ICD/AICD figures to README and align tagline ([#282](https://github.com/bnnr-team/bnnr/issues/282)) ([7f1e3ea](https://github.com/bnnr-team/bnnr/commit/7f1e3ea2405bed36fbf6ddb5b6ebbe9474d4925f))

## [Unreleased]

## [0.4.14] — 2026-06-09

### Fixed

- XAI cache for `ICD`/`AICD` is now precomputed **after** the baseline phase (on the trained baseline model) instead of before it (on random initial weights), so saliency-guided masks are no longer driven by an untrained network in from-scratch runs.
- The cache now defaults to a per-run directory (`<report_dir>/run_<timestamp>/xai_cache`) and carries a `manifest.json` (XAI method, dataset size, image shape); stale maps are dropped on a mismatch. This stops a different model from silently reusing another run's cached saliency under a shared `checkpoint_dir`.

## [0.4.13] — 2026-06-09

### Changed

- CLI flags are now consistent across commands: `bnnr analyze -d` maps to `--device` (previously `--data`); `bnnr train` uses `--token` (previously `--dashboard-token`); `bnnr dashboard export` uses `--output`/`-o` (previously `--out`).
- `bnnr report` and `bnnr analyze` return clean error messages instead of tracebacks for a missing path, an unsupported `--format`, or a mixed-case `--task`.
- Default `device` is now `auto` (was `cuda`) for `BNNRConfig`, `SimpleTorchAdapter`, `DetectionAdapter`, and `UltralyticsDetectionAdapter`, so constructing them on a machine without a GPU no longer raises.
- `bnnr list-presets` lists exactly the presets accepted by `train --preset` (`auto`, `light`, `standard`, `aggressive`, `gpu`, `none`); the `demo` preset is no longer shown there (still available via `bnnr demo`).

### Removed

- `Recommendation.example_command` field from the analyze report schema; it duplicated `literature_note`, which now drives the HTML report citation.

## [0.4.12] — 2026-06-07

### Added

- `examples/integrations/gradcam_to_saliency_augmentation.ipynb` — tutorial from a Grad-CAM heatmap to the ICD/AICD saliency-guided augmentations.
- `docs/getting_started.md`: torchvision analyze golden path promoted with a copy-paste run block.

### Changed

- Benchmark suite: ResNet18 / Imagewoof low-data, from-scratch harness replaces the ResNet50 / CIFAR-100 setup; RandAugment and TrivialAugment baselines.

## [0.4.11] — 2026-05-30

### Added

- `examples/classification/torchvision_analyze_cifar10.py` — ResNet-18 on CIFAR-10 → `analyze_model` HTML report (Python API).
- GitHub issue templates: benchmark contribution, “Who uses BNNR” showcase.
- `docs/getting_started.md`: “Try analyze first” section with live sample HTML link.

### Changed

- `docs/analyze.md`: torchvision checkpoint workflow (example script + CLI note for demo CNN).
- `docs/examples.md`: torchvision analyze example section.
- `docs/assets/analyze-report-sample.html`: footer/header version synced to 0.4.11.

## [0.4.10] — 2026-05-29

### Added

- `docs/plugin_icd.md` and minimal ICD plug-in example (`examples/classification/icd_plugin_minimal.py`) — saliency-guided augmentation without `BNNRTrainer`.
- Integration examples: Grad-CAM → ICD bridge (`examples/integrations/gradcam_to_icd_loop.py`); Ultralytics YOLOv8 quickstart on COCO128 (`examples/integrations/ultralytics_yolo_quickstart.py`).
- `docs/integrations.md` hub; optional extra `pip install "bnnr[ultralytics]"`.

### Changed

- `docs/examples.md`: separate Ultralytics SDK path from YOLO-format + torchvision `showcase_yolo_coco128.py`.

## [0.4.9] — 2026-05-29

### Added

- **Dependabot** (`dependabot.yml`): weekly pip, GitHub Actions, and `dashboard_web` npm groups.
- **Dependency Review** workflow on pull requests to `main`.
- **CODEOWNERS** and expanded **MAINTAINER_SETUP** (rulesets, Dependabot batch policy, review/bypass notes).
- CI security: **pip-audit** (with documented allowlist) and **bandit** in `quality-linux`.

### Changed

- GitHub Actions: `checkout@v6`, `setup-python@v6`, `upload-artifact@v7`, `download-artifact@v8`, `dependency-review-action@v5`.
- Dashboard toolchain: TypeScript 6, recharts 3, Vite 8 + `@vitejs/plugin-react` 6.
- `albumentations` optional extra: `>=1.3.0,<3.0`.
- Consolidated CodeQL quality fixes and closed superseded autofix Dependabot PRs (#59–#103).

## [0.4.8] — 2026-05-28

### Added

- Regression test for `benchmarks/results.json` conditions (#49).
- GitHub Discussions link in README; live githack link in `docs/analyze.md`; benchmark protocol caveat in `docs/benchmarks.md` (#47–#50).

### Changed

- Analyze report `schema_version` aligned with package version (0.4.8); single source in `src/bnnr/version.py`.
- HTML report header and footer show package version, not legacy 0.2.1.
- Type hints in `benchmarks/summarize.py` (#51).

### Fixed

- Version display in analyze HTML reports and regenerated sample artifact.

## [0.4.7] — 2026-05-28

### Fixed

- **Analyze HTML reports** embed all XAI and confusion overlay PNGs as base64 when `artifact_root` is provided — single portable `report.html` (CLI and `report.to_html(..., artifact_root=...)`).

### Changed

- Regenerated [sample analyze HTML report](docs/assets/analyze-report-sample.html) (self-contained, images visible on GitHub).
- Maintainer script: `scripts/generate_analyze_sample_report.py`.

## [0.4.6] — 2026-05-28

### Added

- Sample [analyze HTML report](docs/assets/analyze-report-sample.html) for preview without install.
- **CIFAR-10 benchmark suite** ([`benchmarks/`](benchmarks/)): compares training without BNNR, torchvision RandAugment, and BNNR branch search with ICD/AICD — includes validation metrics and OptiCAM attention overlays on shared validation images.

### Fixed

- **LR scheduler for baseline-only runs** (`max_iterations=0`): `CosineAnnealingLR` no longer uses `T_max=0` in CIFAR/STL pipelines.
- **Version sync:** package, dashboard API, and PyPI metadata aligned to 0.4.6.

## [0.4.5] — 2026-05-22

### Security

- **CodeQL wave 2:** additional path hardening and CI workflow permissions (see PRs #42–#45).

## [0.4.4] — 2026-05-22

### Security

- **CodeQL path hardening**: consolidated fix for *Uncontrolled data used in path expression* alerts in dashboard backend, exporter, and events loader (`path_security` helpers with `ensure_child`, `validate_run_id`, `resolve_events_jsonl`).

## [0.4.3] — 2026-05-22

### Fixed

- **`bnnr demo` post-run messages**: `Your report` and XAI paths print immediately after training (before the dashboard wait loop), using the real artifact location `reports/<run>/artifacts/xai/`.

## [0.4.2] — 2026-05-22

### Fixed

- **`quick_run` tests on Python 3.10+**: mock patches now target the `bnnr.quick_run` submodule directly (the public `bnnr.quick_run` name is the function re-exported from `bnnr.__init__`).

## [0.4.1] — 2026-05-22

### Added

- **CLI `bnnr demo`**: zero-flag onboarding run on CIFAR-10 with a small demo CNN, preset **`demo`** (saliency-guided ICD + ChurchNoise), live dashboard, and post-run paths to the JSON report and `xai/` heatmaps.
- **Preset `demo`**: model-aware augmentation set for the demo command (`build_demo_augmentations` / `get_preset("demo", model=..., target_layers=...)`).
- **`default_demo_config()`** for CLI demo defaults (`m_epochs=1`, `max_iterations=1`, XAI on).
- **`docs/quickstart_api.md`**: one-page `quick_run` reference.

### Changed

- **`quick_run`**: uses `default_train_config()` when `config` is omitted; infers `target_layers` from the last `Conv2d`; `dashboard=True` starts the live dashboard before training (non-blocking after `run()`).
- **Docs**: README, getting started, CLI, API reference, and golden path promote `bnnr demo` / `quick_run` as the primary quickstarts.

## [0.4.0] — 2026-05-22

### Changed

- **Training refactor (6.1 / 6.2)**: split monolithic training code into `bnnr.training` submodules and reduced top-level `bnnr` re-exports; public import paths and behavior remain backward compatible.

## [0.3.1] — 2026-05-21

### Added

- **CLI `bnnr quickstart`**: interactive wizard for a zero-config demo run (built-in defaults, sample limits 128/64, optional live dashboard).
- **Zero-config `bnnr train`**: `--config` is optional; built-in quickstart defaults when omitted (`default_train_config()`).
- **Community infra**: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates, label sync config.

### Changed

- **PyPI readme**: dedicated `README.pypi.md` with absolute asset URLs (logo, dashboard screenshots, demo poster) so images render on [pypi.org](https://pypi.org/project/bnnr/); synced with week-1 audit README.
- **GitHub README**: problem-first structure, dashboard screenshots, ICD/AICD panels, inline demo video with audio (1280p).
- **Classifiers**: Development Status `4 - Beta` (was Alpha on PyPI 0.3.0 metadata).

## [0.3.0] — 2026-05-15

### Added

- **Analyze**: standalone model analysis feature (`bnnr analyze`) — metrics, XAI diagnostics, data quality checks, failure pattern detection, structured recommendations, and HTML report generation.
- **Analysis module** (`src/bnnr/analysis/`): schema, findings engine, recommendations engine, class diagnostics, calibration (ECE), cross-validation, and HTML report renderer.
- **STL-10 pipeline**: built-in STL-10 dataset support with VGG-style and quick CNN architectures.
- **CLI**: `bnnr analyze` command with `--model`, `--data`, `--output`, `--task`, `--cv-folds`, `--xai-samples` options.
- **Evaluation module** (`src/bnnr/evaluation.py`): prediction collection and evaluation utilities for analyze.

### Changed

- **Pipelines**: supported datasets expanded to include `stl10` alongside existing `mnist`, `fashion_mnist`, `cifar10`, `imagefolder`, `coco_mini`, `yolo`.
- **Exports**: `analyze_model` and `AnalysisReport` added to public API (`bnnr.__init__`).

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
