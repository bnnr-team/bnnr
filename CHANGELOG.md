# Changelog

## [0.1.0] — 2026-03-01

### Initial Release

- BNNR Core Training Loop — automated augmentation evaluation and selection
- 7 custom augmentations: ChurchNoise, Drust, LuxferGlass, ProCAM, Smugs, TeaStains, BasicAugmentation
- External augmentation integration: Albumentations, Kornia, torchvision
- XAI: OptiCAM, GradCAM saliency maps
- Advanced XAI: CRAFT, RealCRAFT, RecursiveCRAFT, NMF concept decomposition
- ICD / AICD — XAI-driven intelligent augmentation
- Live dashboard (React + FastAPI + WebSocket)
- CLI: train, report, dashboard, list-augmentations, list-presets, list-datasets
- Config system (YAML/JSON) with presets: auto, standard, aggressive, light, gpu
- Data quality analysis (duplicate detection, anomaly checks)
- Classification, detection (COCO, YOLO), and multi-label support
- Checkpointing and training resume
- Event system (JSONL) for training replay
