# Product roadmap

**Updated:** 2026-06-09 · **Current release:** v0.6.5 <!-- x-release-please-version -->

BNNR is a PyTorch vision toolkit focused on **model diagnostics first** (`bnnr analyze`), then saliency-guided augmentations (ICD/AICD), with optional training and detection adapters.

**Start here:** [analyze.md](analyze.md) · [sample HTML report](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html)

---

## Shipped (baseline)

| Area | What exists today |
|------|-------------------|
| **Analyze** | `bnnr analyze` + portable HTML/JSON for classification and multilabel; [sample report](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html) |
| **Training** | CLI `train` / `demo` / `quickstart`; presets; live dashboard |
| **XAI aug** | ICD, AICD, branch search; [plugin_icd.md](plugin_icd.md) for custom loops |
| **Detection** | Train API + bbox augs; Ultralytics adapter; [detection.md](detection.md) — **no** `analyze` for detection yet |
| **Benchmarks** | CIFAR-10 demo CNN (3 seeds): no aug vs RandAugment vs BNNR — [benchmarks.md](benchmarks.md) |
| **Integrations** | pytorch-grad-cam loop, Ultralytics quickstart — [integrations.md](integrations.md) |
| **Torchvision analyze** | Python example: [torchvision_analyze_cifar10.py](../examples/classification/torchvision_analyze_cifar10.py) |

---

## Q2 2026 (June–August) — credibility + analyze adoption

| Planned | Description |
|---------|-------------|
| **Analyze-first docs** | README and [getting_started.md](getting_started.md): try analyze before train; githack sample links |
| **Augmentation benchmark** | ResNet18 / Imagewoof (fine-grained, low-data, from-scratch): 5 seeds; baselines RandAugment + TrivialAugment; reproducible script `benchmarks/reproduce_imagewoof.sh` (free-T4 friendly) |
| **Torchvision → analyze** | Prominent golden path: pretrained classifier → HTML report; runnable example in README |
| **Repo discoverability** | GitHub description, website link, topics (maintainer ops) |
| **Contributor templates** | Issue forms: benchmark proposal, “Who uses BNNR” showcase |
| **Upstream docs** | Official link in Ultralytics docs (after PR merge); grad-cam ecosystem follow-up |
| **HuggingFace Space** | Minimal Gradio demo (sample report or analyze workflow) |

---

## Q3 2026 (September–November) — analysis depth + ecosystem

| Planned | Description |
|---------|-------------|
| **HuggingFace models** | First `transformers` path for `analyze` (e.g. ViT) — documented example, not full Hub integration yet |
| **Dataset insight** | “What to label next” / outlier buckets for manual review |
| **More XAI methods** | Better stability + more methods; OptiCAM improvements; config docs |
| **Multilabel CLI** | Parity for `bnnr train` / analyze workflows |
| **Colab** | Analyze-only notebook linked from README |
| **Batch reports** | Run `analyze` across many checkpoints/configs (foundation for compare) |

---

## Q4 2026 — depth + scaling

| Planned | Description |
|---------|-------------|
| **Compare runs** | Side-by-side `analyze` for two checkpoints (compliance / audit use cases) |
| **v1.0 criteria** | Public stability criteria; reduce “beta” friction on PyPI when met |
| **GPU speedups** | Faster `analyze` runtime; caching and better defaults |
| **More report templates** | Stakeholder-ready summary pages and comparisons |
| **HF Hub analyze** | Load pretrained weights via Hub URI where feasible |

---

## 2027 H1 (gated)

| Planned | Description |
|---------|-------------|
| **Detection analyze** | Heatmaps + failure buckets for detection — only after classification analyze adoption and clear community demand |
| **CI integration** | Optional GitHub Actions step producing analyze HTML as artifact |

See [detection.md](detection.md) for current train-only detection scope.
