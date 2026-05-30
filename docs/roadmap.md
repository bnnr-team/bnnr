# Product roadmap

**Updated:** 2026-05-30 · **Current release:** v0.4.10

BNNR is a PyTorch vision toolkit: **train → explain → improve → prove** — with `bnnr analyze` (failure + XAI HTML report), ICD/AICD augmentations, and optional detection training via adapters.

---

## Shipped (baseline)

| Area | What exists today |
|------|-------------------|
| **Analyze** | `bnnr analyze` + portable HTML/JSON for classification and multilabel; [sample report](https://raw.githack.com/bnnr-team/bnnr/refs/heads/main/docs/assets/analyze-report-sample.html) |
| **Training** | CLI `train` / `demo` / `quickstart`; presets; live dashboard |
| **XAI aug** | ICD, AICD, branch search; [plugin_icd.md](plugin_icd.md) for custom loops |
| **Detection** | Train API + bbox augs; Ultralytics adapter; [detection.md](detection.md) — **no** `analyze` for detection yet |
| **Benchmarks** | CIFAR-10 (3 seeds): no aug vs RandAugment vs BNNR — [benchmarks.md](benchmarks.md) |
| **Integrations** | pytorch-grad-cam loop, Ultralytics quickstart — [integrations.md](integrations.md) |

---

## Q2 2026 (June–July) — analyze adoption

| Planned | Description |
|---------|-------------|
| **Torchvision → analyze** | Docs + runnable example: pretrained classifier checkpoint → `bnnr analyze` report (no full retrain) |
| **Getting started** | “Try analyze first” + githack sample links in [getting_started.md](getting_started.md) |
| **Upstream docs** | Official link in Ultralytics docs (after PR merge); grad-cam ecosystem follow-up |
| **Contributor templates** | GitHub issue forms: benchmark proposal, “Who uses BNNR” showcase |

---

## Q3 2026 (Aug–Sept) — analysis depth

| Planned | Description |
|---------|-------------|
| **Dataset insight** | “What to label next” / outlier buckets for manual review |
| **More XAI methods** | Better stability + more methods; OptiCAM improvements; config docs |
| **Batch reports** | Run `analyze` across many checkpoints/configs; compare runs |

---

## Q4 2026 — detection analyze + scaling

| Planned | Description |
|---------|-------------|
| **Detection analyze** | Heatmaps + failure buckets for detection tasks |
| **GPU speedups** | Faster `analyze` runtime; caching and better defaults |
| **More report templates** | Stakeholder-ready summary pages and comparisons |

