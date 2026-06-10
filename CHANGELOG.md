# Changelog

## [0.5.0](https://github.com/bnnr-team/bnnr/compare/bnnr-v0.4.14...bnnr-v0.5.0) (2026-06-10)


### Features

* **analyze:** merge analyze report feature; bump v0.3.0 ([e320e13](https://github.com/bnnr-team/bnnr/commit/e320e13a7c8ed5a68c08dd436258a591c402e1aa))
* **analyze:** multilabel report path, ECE, CV; CLI classification/multilabel only ([1756b09](https://github.com/bnnr-team/bnnr/commit/1756b096b30c60ae66d5450b3c9424ecfda8e55d))
* **benchmark:** Colab notebook + --drive-base-dir flag for one-click runs ([#205](https://github.com/bnnr-team/bnnr/issues/205)) ([096c6b4](https://github.com/bnnr-team/bnnr/commit/096c6b4a8c288637f13ad3327f5c62a1fa37a3e0))
* **benchmarks:** add reproducible baseline vs BNNR harness ([08d42cf](https://github.com/bnnr-team/bnnr/commit/08d42cfb359dd1e7578948af9c07c7242805e88e))
* **detection:** enable XAI and probe snapshots for Ultralytics YOLO ([b58b749](https://github.com/bnnr-team/bnnr/commit/b58b74989016f8e6090ca3ff267d302bd467ac65))
* **detection:** Ultralytics v8 training adapter and YOLO loader options ([8035bbd](https://github.com/bnnr-team/bnnr/commit/8035bbd9254c4e5bd6bdcb1cc00e40385f50061b))
* Faza 0 integrations (grad-cam, Ultralytics, docs hub, outreach) ([ad7b889](https://github.com/bnnr-team/bnnr/commit/ad7b889017e79e6c94c5515d046041a0acc3d49f))
* merge detection feature into main for v0.2.0 release ([61763c5](https://github.com/bnnr-team/bnnr/commit/61763c590c17fcbb97589a15d93960166fec98e8))
* **notebook:** add butterfly Colab cooking notebook for students ([79be03f](https://github.com/bnnr-team/bnnr/commit/79be03ff95e5055c9751f5431a917e9481f3ae0a))
* **notebook:** add butterfly Colab cooking notebook for students ([3e2def8](https://github.com/bnnr-team/bnnr/commit/3e2def8356659da5ce0e9637c44537f07f8ab037))
* week-1 OSS polish — README, zero-config CLI, community infra ([3641c74](https://github.com/bnnr-team/bnnr/commit/3641c7422fa8a0ee3710eac5175c7144701b7c19))


### Bug Fixes

* all co-authors in CITATION.cff ([2d86310](https://github.com/bnnr-team/bnnr/commit/2d86310de21d484f252bf3cdd1e8aa4bb89f6d93))
* **analyze:** auto-detect STL-10 model architecture from checkpoint ([60d09fd](https://github.com/bnnr-team/bnnr/commit/60d09fdc7b2b179f4292bc0f93aae12f9317a587))
* **analyze:** embed XAI images in portable HTML report; release v0.4.7 ([67ad30d](https://github.com/bnnr-team/bnnr/commit/67ad30d514e8bbc8ae3c634a6af74b895144b96b))
* **analyze:** resolve QA issues in report and docs ([6df0f98](https://github.com/bnnr-team/bnnr/commit/6df0f9809ffa8be2bb8c575f8ce2d6bb873d6516))
* ASCII arrow in print for Windows console encoding ([756029b](https://github.com/bnnr-team/bnnr/commit/756029bd5df913c822b1f4dc01e5fe3949f8da41))
* **benchmark:** artifacts, resume safety, stats, notebook cleanup ([#217](https://github.com/bnnr-team/bnnr/issues/217)) ([841af4f](https://github.com/bnnr-team/bnnr/commit/841af4fd72ef9765217ae3f53717f2a6f7626b4e))
* **benchmark:** make run_resnet50.py actually resume-safe ([#204](https://github.com/bnnr-team/bnnr/issues/204)) ([c0e2a64](https://github.com/bnnr-team/bnnr/commit/c0e2a6410e365e4c66ae1913fcfb10507e0bf50f))
* **ci:** add workflow permissions for CodeQL ([#42](https://github.com/bnnr-team/bnnr/issues/42)) ([39a1599](https://github.com/bnnr-team/bnnr/commit/39a1599218dbae7629b7a3c162a76732526a7d91))
* **ci:** build dashboard before pip install -e; release 0.1.0.post4 ([9b61e56](https://github.com/bnnr-team/bnnr/commit/9b61e5611436811ac18f59182b3b7f5c387931f6))
* **ci:** gradcam smoke test without network downloads ([28f1fe4](https://github.com/bnnr-team/bnnr/commit/28f1fe46495a6862310ce8f59915d3c0d9984a59))
* consolidate code-quality autofixes ([#221](https://github.com/bnnr-team/bnnr/issues/221)–[#228](https://github.com/bnnr-team/bnnr/issues/228)) ([a819276](https://github.com/bnnr-team/bnnr/commit/a8192760e08d0e29f8524e60b0f09a0bb006139c))
* **dashboard:** lodash &gt;=4.18.0 via npm overrides; release 0.1.2, sync version strings ([dfc7731](https://github.com/bnnr-team/bnnr/commit/dfc77311bcf048c2ff55b6ff1eddf7f408ec9e4a))
* declare click as a runtime dependency ([#240](https://github.com/bnnr-team/bnnr/issues/240)) ([8c1e168](https://github.com/bnnr-team/bnnr/commit/8c1e1685baa3cae5ffff1cd859c0165de006bae2))
* detection examples compatibility issues ([4678bd1](https://github.com/bnnr-team/bnnr/commit/4678bd1fc44cbf7fe4475bd2c731ff933bf058b8))
* detection metrics averaging, YOLO input scaling, dashboard detection UI (v0.2.2) ([b1581c3](https://github.com/bnnr-team/bnnr/commit/b1581c39a8fb3d89f95311ec31302cbae73a9a04))
* **detection-xai:** pick Ultralytics conv layer by saliency isotropy ([ddb136b](https://github.com/bnnr-team/bnnr/commit/ddb136b5acd67f6a507aac6916bb43ff1df7999a))
* **detection:** YOLO adapter image scale, rotate90 demo semantics, ICD targets ([d3220f0](https://github.com/bnnr-team/bnnr/commit/d3220f0728b2efb6ddbac8f18dee4fcf495c6da8))
* **detection:** YOLO scale [0,1], Colab notebook install notes ([927f50f](https://github.com/bnnr-team/bnnr/commit/927f50ffc4638ac87dc1dfecf05223475e9ec697))
* final audit — docs consistency, broken refs, and Colab URL ([09ddf9e](https://github.com/bnnr-team/bnnr/commit/09ddf9e66d3a0568afb3a2c48a78ffe878712645))
* **import:** lazy sklearn for Colab numpy/scipy mismatches ([3c61345](https://github.com/bnnr-team/bnnr/commit/3c613457151d79e4d1576969aa89c32dc6e0ece0))
* include predicted classes in analyze CV confusion matrix ([#188](https://github.com/bnnr-team/bnnr/issues/188)) ([4179d97](https://github.com/bnnr-team/bnnr/commit/4179d975ff9f1abe8a085d7d956faec5c6a52334))
* **lint:** rename uppercase variable L to n_labels (N806) ([398c819](https://github.com/bnnr-team/bnnr/commit/398c819a29b084297c4109b1a6709a53f77b036f))
* list all co-authors in CITATION.cff and citation docs ([880be22](https://github.com/bnnr-team/bnnr/commit/880be221183b39617973abb83a07e9a7de99fef6))
* **mypy:** add bnnr.detection_icd to no-any-return overrides ([499fd8d](https://github.com/bnnr-team/bnnr/commit/499fd8d9b7dcc0948b8bf125cdc48c9875597a67))
* **notebook:** remove torch.no_grad around OptiCAM saliency in augmentations guide ([de9b619](https://github.com/bnnr-team/bnnr/commit/de9b619c2830e77f9ccee4f41947b69e1df333ca))
* **notebook:** use saliency maps as numpy from generate_saliency_maps ([9267feb](https://github.com/bnnr-team/bnnr/commit/9267feb2247491b99b5ed7acc65d4f6c8d06de4b))
* precompute XAI cache after baseline and isolate it per run ([#276](https://github.com/bnnr-team/bnnr/issues/276)) ([f0cdc22](https://github.com/bnnr-team/bnnr/commit/f0cdc22709e8d38dcdc50953b52e893499e4fdb4)), closes [#252](https://github.com/bnnr-team/bnnr/issues/252)
* **pypi:** README.pypi.md with absolute URLs; bump 0.3.1 ([12e5979](https://github.com/bnnr-team/bnnr/commit/12e5979a36abc289d7d20e13da3afcc04b876496))
* **readme:** inline video player with audio instead of download link ([4c9f256](https://github.com/bnnr-team/bnnr/commit/4c9f25691adf82d4695df158fcd3e766ff90e138))
* **readme:** pepy downloads badge, analyze intro, clickable hero video ([c8f921f](https://github.com/bnnr-team/bnnr/commit/c8f921f9e177ea7cef7067b05e1d35027bb20d96))
* remove unused pytest import in gradcam smoke test ([ac4e512](https://github.com/bnnr-team/bnnr/commit/ac4e5124a5e0839a247fe1fc5f0dad8d7f6d04d0))
* remove unused pytest import in ICD plugin smoke test ([b9c2d41](https://github.com/bnnr-team/bnnr/commit/b9c2d410206afe438c0fb84b1ef94d4c49bb1897))
* **security:** avoid logging full events path in warnings ([#43](https://github.com/bnnr-team/bnnr/issues/43)) ([e01bc35](https://github.com/bnnr-team/bnnr/commit/e01bc3559610b7807ae851e7de87975b1a06e09d))
* **security:** bump vite 5.4→6.4.2 to resolve CVE path-traversal ([205bc1c](https://github.com/bnnr-team/bnnr/commit/205bc1cbfda496d959c862a7c828be7ac2a41870))
* **security:** harden dashboard export manifest and HTML ([#44](https://github.com/bnnr-team/bnnr/issues/44)) ([a1d49c0](https://github.com/bnnr-team/bnnr/commit/a1d49c0a1c9bcafaa7d88c92b6fb62b62a8867fe))
* **security:** harden path handling for CodeQL alerts (v0.4.4). ([#33](https://github.com/bnnr-team/bnnr/issues/33)) ([93b77cf](https://github.com/bnnr-team/bnnr/commit/93b77cf21110c2c004e787c77da4d07afe1789d1))
* **security:** patch esbuild and rollup vulnerabilities via npm overrides ([f9114f2](https://github.com/bnnr-team/bnnr/commit/f9114f2c291aaf3d0f4f5fbca8572b6816f6f634))
* **security:** patch esbuild and rollup vulnerabilities via npm overrides ([9f04c3a](https://github.com/bnnr-team/bnnr/commit/9f04c3af28f3dbaebad2cd3fccc1c18cfb7325a5))
* **security:** patch esbuild and rollup vulnerabilities via npm overrides ([b07f1b0](https://github.com/bnnr-team/bnnr/commit/b07f1b002fff14cf266ee2bb4c7990c9233dbbea))
* stop pulling fastapi into the core training loop ([#241](https://github.com/bnnr-team/bnnr/issues/241)) ([8927f1e](https://github.com/bnnr-team/bnnr/commit/8927f1e6059c8dd9798b60e649bb792cb9ec9aa7))
* **tests:** adjust list_datasets assertion for 4 datasets (post-detection removal) ([8d0a5ad](https://github.com/bnnr-team/bnnr/commit/8d0a5aded7ec68697d8708d3bdc12bb0f53cac29))
* treat winning baseline_reeval as no improvement instead of crashing ([#279](https://github.com/bnnr-team/bnnr/issues/279)) ([6ae1512](https://github.com/bnnr-team/bnnr/commit/6ae151226989fd19bd4bc9a79c6b292973948b8a))
* use ValueError in dataset __getitem__ for CodeQL ([91f0b01](https://github.com/bnnr-team/bnnr/commit/91f0b01c65587bf2f58f99cd6cca14909caae54d))
* **yolo:** stop non-finite train_step — no GradScaler, clamp cls, grad clip ([e8885a4](https://github.com/bnnr-team/bnnr/commit/e8885a492059bf6a477bcef5b4882edac96fe687))


### Reverts

* **docs:** restore README and README.pypi to pre-launch-edit layout ([fcf88b9](https://github.com/bnnr-team/bnnr/commit/fcf88b960927e81bdad372dddcd37dd71e3f06bc))


### Documentation

* add BNNR citation (CITATION.cff) and integration citing guide ([54f5c2c](https://github.com/bnnr-team/bnnr/commit/54f5c2c68e6cd7a6c983103f22e405c609b14917))
* add BNNR citation (CITATION.cff) and integration citing guide ([65b6f24](https://github.com/bnnr-team/bnnr/commit/65b6f24a5b0e68c9e77407ba41cbf490f090c97a))
* add comprehensive object detection documentation ([ad03045](https://github.com/bnnr-team/bnnr/commit/ad03045a3d8328c7b2dccac8badf55f31e7e988d))
* add externally-managed-environment and venv troubleshooting section ([#201](https://github.com/bnnr-team/bnnr/issues/201)) ([db3bfd0](https://github.com/bnnr-team/bnnr/commit/db3bfd07c5357a11c0309a14b2076598a39b5162)), closes [#182](https://github.com/bnnr-team/bnnr/issues/182)
* add launch story assets and analyze-first README ([ab4830d](https://github.com/bnnr-team/bnnr/commit/ab4830d9144264778658270a71ae274d9c00672c))
* add PyPI downloads badge (pepy.tech) across entrypoints ([7d29989](https://github.com/bnnr-team/bnnr/commit/7d29989b15a0703af07eaf1d1500268d60732f2e))
* add Roadmap link to Links tables ([#166](https://github.com/bnnr-team/bnnr/issues/166)) ([e67ae47](https://github.com/bnnr-team/bnnr/commit/e67ae4700dae22dfbba4ab4d8852618f59678aa7))
* add Zenodo DOIs for software and ICD/AICD paper ([#238](https://github.com/bnnr-team/bnnr/issues/238)) ([ab7f52e](https://github.com/bnnr-team/bnnr/commit/ab7f52e0f6f7b12afc90f268a1600f7de25cfacd))
* add Zenodo DOIs for software and ICD/AICD paper ([#239](https://github.com/bnnr-team/bnnr/issues/239)) ([802f3c8](https://github.com/bnnr-team/bnnr/commit/802f3c8588e3a15b975f74820d3c5a0bb973c4be))
* align public roadmap with analyze-first and Q2–Q4 plan ([#185](https://github.com/bnnr-team/bnnr/issues/185)) ([f36d8bc](https://github.com/bnnr-team/bnnr/commit/f36d8bc02837d49c587ae99cec3addf17d32ee2d))
* align user documentation with code (audit 2026-05-31) ([#186](https://github.com/bnnr-team/bnnr/issues/186)) ([daefea2](https://github.com/bnnr-team/bnnr/commit/daefea24a473573998a5f85451cd6fe7728adb60))
* analyze-first README hero, lead with bnnr analyze ([#199](https://github.com/bnnr-team/bnnr/issues/199)) ([6c0ee1d](https://github.com/bnnr-team/bnnr/commit/6c0ee1d272a522101d49da3192de03aafa62e5c7))
* **analyze:** add --summary flag to analyze.md, expand api_reference.md ([ed7e251](https://github.com/bnnr-team/bnnr/commit/ed7e2518ad3fd814fc0e95160a7492bdd2578e8c))
* **analyze:** supplement classification docs — quick start, stl10, xai-samples, fix API reference ([b525b20](https://github.com/bnnr-team/bnnr/commit/b525b20bd4e63d46cb1080b4d891027913ac1a28))
* clarify YOLO-format vs Ultralytics SDK examples ([55d3e2a](https://github.com/bnnr-team/bnnr/commit/55d3e2afc8787639d222703aa03a32a0c22d3739))
* drop optional label from analyze section heading ([8761702](https://github.com/bnnr-team/bnnr/commit/8761702111c2836e4d3a41d478c4ff74e32b6fef))
* drop release_pypi index entry (doc removed) ([83a5bc9](https://github.com/bnnr-team/bnnr/commit/83a5bc980c5d59e75bc446d7095dfd60dffb0aac))
* drop release_pypi index entry (doc removed) ([0d6b63e](https://github.com/bnnr-team/bnnr/commit/0d6b63ed5b0208c42010fabcd96f99b029f312fb))
* drop release_pypi index entry (doc removed) ([743e72d](https://github.com/bnnr-team/bnnr/commit/743e72d603ec17bc846e55417e816cd49d3db72b))
* **examples:** detection demo uses YOLOv8n and COCO128 ([768c99b](https://github.com/bnnr-team/bnnr/commit/768c99b287fba5b048778d246c24727d753c17b7))
* fix ICD acronym typo (Intelligence -&gt; Intelligent Coarse Dropout) ([#236](https://github.com/bnnr-team/bnnr/issues/236)) ([d5a465f](https://github.com/bnnr-team/bnnr/commit/d5a465f21ed80f8a06d39ed15a4fa7d7f96a3d86))
* hide object detection from public README and changelog ([bf64484](https://github.com/bnnr-team/bnnr/commit/bf644844f87697a479603a1c8aba65e62c3f1c0c))
* ICD plug-in guide + minimal training loop (no BNNRTrainer) ([d2ba461](https://github.com/bnnr-team/bnnr/commit/d2ba461ba949c7afd2173d4a45f5e4943d3d709d))
* ICD plug-in guide + minimal training loop (no BNNRTrainer) ([4335050](https://github.com/bnnr-team/bnnr/commit/43350501fc25f66a5fbc5f51243774f2e2220f59))
* improve README links, tagline, and live sample report preview ([60a00d1](https://github.com/bnnr-team/bnnr/commit/60a00d1fb81ee1517807e593985ef1fdbfb11ac1))
* integrations hub (pytorch-grad-cam + Ultralytics) ([0a3f743](https://github.com/bnnr-team/bnnr/commit/0a3f74335fc25474852b99f7d6c831865ff89ee8))
* move Quickstart with demo/quickstart above the fold in README ([88b9dff](https://github.com/bnnr-team/bnnr/commit/88b9dffeaf14730cbecd8538994c0a828ef4dba1))
* one-click analyze quickstart Colab notebook ([#218](https://github.com/bnnr-team/bnnr/issues/218)) ([42d8cc3](https://github.com/bnnr-team/bnnr/commit/42d8cc31cfd415a42564eb137826624ddfb9324e))
* promote torchvision analyze golden path in getting_started ([#230](https://github.com/bnnr-team/bnnr/issues/230)) ([ff507ce](https://github.com/bnnr-team/bnnr/commit/ff507ce795c9ee7750bb2777f1d578206c9a88f4))
* publish CIFAR-10 benchmark results and branded OptiCAM figure ([6300f13](https://github.com/bnnr-team/bnnr/commit/6300f1327826b581d3361eff535c16a5f2fdcac9))
* put bnnr demo first in README, move analyze below fold ([88fd1ae](https://github.com/bnnr-team/bnnr/commit/88fd1ae6cfb5151a0cf06cce94c835d04e9e9e96))
* PyPI launch polish — wheel vs repo, multilabel/CLI clarity ([610db1d](https://github.com/bnnr-team/bnnr/commit/610db1d9e990244e18be003cb116e8b5cf0e8699))
* README index note for detection advanced API ([a0f8c12](https://github.com/bnnr-team/bnnr/commit/a0f8c1276feae118371b993f8b1467007c7dbd2f))
* **readme:** point to bnnr.dev for full demo video with audio ([798fc65](https://github.com/bnnr-team/bnnr/commit/798fc65474cbca33b1c7206eb3fc4d24b61b92fe))
* torchvision analyze section and contributor issue templates ([#180](https://github.com/bnnr-team/bnnr/issues/180)) ([d9ad723](https://github.com/bnnr-team/bnnr/commit/d9ad7231976a7da488330a582c2f770f5a2c613b))
* Ultralytics YOLO path, YOLO-format loaders, troubleshooting ([fe059aa](https://github.com/bnnr-team/bnnr/commit/fe059aad6bc576cdecb1d8af120df24a22d87145))
* update README version label to v0.2.1 ([c1f1aab](https://github.com/bnnr-team/bnnr/commit/c1f1aabdb84f98c573ffe0407ca595041f591ffc))
* update version references from 0.4.11 to 0.4.12 ([#243](https://github.com/bnnr-team/bnnr/issues/243)) ([b008e68](https://github.com/bnnr-team/bnnr/commit/b008e687807146cd1843ce243fe0b77ca013d807))

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
