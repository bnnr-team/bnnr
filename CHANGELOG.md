# Changelog

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
- **Tasks:** single-label classification and multi-label classification only
- Checkpointing and training resume
- Event system (JSONL) for training replay
