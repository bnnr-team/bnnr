# Benchmarks

Two reproducible augmentation comparisons:

1. **CIFAR-10 / demo CNN** (below) — fast, illustrative, runs on CPU.
2. **ResNet50 / CIFAR-100** ([jump](#resnet50--cifar-100-benchmark)) — a real architecture and a harder dataset, with RandAugment **and** TrivialAugment baselines over 5 seeds. Infrastructure is in place; numbers land in the table after a GPU run.

---

## CIFAR-10 / demo CNN

Reproducible comparison of **three training setups** on the same demo CNN, dataset split, and epoch budget:

| Condition | What it is |
|-----------|------------|
| `no_bnnr` | Crop + flip only — no BNNR augmentations, no branch search |
| `randaugment` | **torchvision RandAugment** — random policy-based augmentations (external baseline) |
| `bnnr_branch_search` | Full **BNNR branch search** over **ICD**, **AICD**, and ChurchNoise |

## What we compare

1. **BNNR vs no augmentation** — does the branching system (with saliency-guided ICD/AICD) beat plain training?
2. **BNNR vs RandAugment** — does targeted, XAI-aware augmentation beat off-the-shelf random augs?
3. **Attention maps** — after each run, **OptiCAM** overlays on the **same 8 validation images** (`config.yaml` → `xai_val_indices`). Compare `runs/*/xai/attention_*.png` to see where each model looks.

Lower **edge ratio** and more focused **coverage** on the object usually indicate less background reliance.

## Layout

```
benchmarks/
  config.yaml      # shared epochs, metrics, RandAugment params, XAI indices
  lib.py           # conditions, training, attention export
  run.py           # CLI (resume-safe)
  summarize.py     # metrics + attention stats table
  results.json     # aggregated results (commit after review)
  runs/            # per-run logs + xai/ overlays (gitignored)
```

## Run

```bash
python benchmarks/run.py --seeds 42 --device cpu
python benchmarks/summarize.py --markdown
```

Three seeds for publication-ready numbers:

```bash
python benchmarks/run.py --seeds 42,43,44 --device cpu
```

List conditions:

```bash
python benchmarks/run.py --list-conditions
```

## BNNR augmentations in this benchmark

| Name | Role |
|------|------|
| **ICD** | Masks *high-saliency* regions — forces the model to look beyond the easiest cue |
| **AICD** | Masks *low-saliency* background — reduces shortcut learning on context |
| **ChurchNoise** | Lightweight noise aug — non-XAI candidate in the branch pool |

The branch search keeps augmentations that improve validation accuracy; the winning path is recorded in `results.json` → `best_path`.

## Shared fairness rules

- Same demo CNN (`_CifarCNN`), Adam lr=1e-3, batch 64, `m_epochs` from `config.yaml`
- Same random seed per condition within a run
- RandAugment uses `num_ops=2`, `magnitude=9` (torchvision defaults in `_benchmark`)
- Attention maps always use the same validation indices across conditions

**Protocol note:** `no_bnnr` and `randaugment` train for **5 epochs** only. `bnnr_branch_search` runs baseline (5 ep) plus candidate screening (up to 3×5 ep per iteration × 3 iterations) — **much more compute** and a different curriculum (augmentations added after baseline). Compare numbers as *“full BNNR product vs fixed-epoch baselines”*, not equal-budget ablation.

## Results (2026-05-28, seeds 42–44, CPU)

| Condition | Median val acc | Δ vs no BNNR | Per-seed |
|-----------|----------------|--------------|----------|
| Without BNNR (crop + flip) | **75.3%** | — | 75.3, 75.6, 75.3 |
| RandAugment (torchvision 2,9) | **72.5%** | −2.8 pp | 72.2, 72.5, 73.5 |
| BNNR branch search | **81.4%** | +6.1 pp | 81.4, 81.3, 81.6 |

Within-run BNNR gain vs its own baseline phase: +7.3 to +12.1 pp (`gain_vs_within_run_baseline_pp` in `results.json`). Winning paths varied by seed (e.g. ChurchNoise→ICD, or full ICD+AICD+ChurchNoise stack).

**Takeaways (honest):**

1. **BNNR full pipeline** clearly beats both baselines on this demo setup — stable ~81% across seeds.
2. **RandAugment at 5 epochs** underperforms crop+flip here; regularization likely needs longer training or lower `magnitude` — not a bug in integration.
3. **Attention (OptiCAM):** BNNR shows lower mean coverage (~13.5% vs ~18%); edge ratio mixed — qualitative XAI in `runs/*/xai/`, not a single headline metric.
4. **Do not claim SOTA or “beats RandAugment” without citing protocol** — demo CNN, short baselines, unequal epoch budget.

Raw data: [`results.json`](results.json). Regenerate table: `python benchmarks/summarize.py --markdown`.

### README figure

Side-by-side OptiCAM on the same val image (seed 44):

```bash
python scripts/build_benchmark_xai_readme_asset.py
```

Output: `docs/assets/benchmark-xai-comparison.png` (used in root README).

---

## ResNet50 / CIFAR-100 benchmark

A larger, more convincing benchmark than the demo-CNN / CIFAR-10 table above: a real **ResNet50** backbone (ImageNet-pretrained by default) on **CIFAR-100** (100 classes), comparing the BNNR branch search against **two** strong external baselines.

| Condition | What it is |
|-----------|------------|
| `no_bnnr` | Resize + RandomCrop + RandomHorizontalFlip — no BNNR augmentations |
| `randaugment` | + **torchvision RandAugment** (external baseline) |
| `trivialaugment` | + **torchvision TrivialAugmentWide** (parameter-free external baseline) |
| `bnnr_branch_search` | Full **BNNR branch search** over **ICD**, **AICD**, and ChurchNoise |

### Design

- **ResNet50** from `torchvision.models`, ImageNet weights. The classifier head is replaced for 100 classes.
- **In-model normalization:** ImageNet mean/std are applied *inside* the model (registered buffers), so every condition feeds plain `ToTensor()` tensors in `[0, 1]`. Pretrained weights stay valid and BNNR's uint8-range ICD/AICD augmentations remain compatible.
- **Same** backbone, optimizer (SGD, momentum 0.9, weight decay 5e-4), cosine schedule, epochs, and seeds across all conditions — only the augmentation strategy varies.
- **OptiCAM** overlays on fixed CIFAR-100 test indices, exported per run (`runs_resnet50/*/xai/`).

### Run

```bash
# Fast sanity check (CPU-friendly, no pretrained download, tiny subset)
python benchmarks/run_resnet50.py --smoke

# Full benchmark — 5 seeds, GPU (this is the publication run)
bash benchmarks/reproduce_resnet50.sh
# or:
python benchmarks/run_resnet50.py --seeds 42,43,44,45,46 --device cuda

# Summarize
python benchmarks/summarize.py --results benchmarks/results_resnet50.json --markdown
```

`run_resnet50.py` checkpoints `results_resnet50.json` after every (condition, seed) run, so the matrix is resume-safe — a crash or interruption keeps completed runs.

### Layout

```
benchmarks/
  run_resnet50.py            # CLI (resume-safe matrix runner)
  reproduce_resnet50.sh      # one-command full run (5 seeds)
  results_resnet50.json      # aggregated results (commit after review)
  runs_resnet50/             # per-run logs + xai/ overlays (gitignored)
```

### Protocol caveats (read before quoting numbers)

- **Unequal compute by design.** `bnnr_branch_search` runs a baseline phase plus branch search (more epochs of compute than the fixed-epoch baselines). Compare as *"full BNNR product vs fixed-epoch baselines"*, not equal-budget ablation.
- **Not an ImageNet-SOTA claim.** This is a CIFAR-100 transfer setup for comparing augmentation *strategies*, not a leaderboard entry.
- **CIFAR-100 upscaled to 224px** to use the pretrained ResNet50 features. `--img-size` and `--no-pretrained` are available for other regimes.

### Results

_Pending a GPU run._ Run `reproduce_resnet50.sh`, review `results_resnet50.json`, then paste the `summarize.py --markdown` table here and into the root `README.md`. Do not hand-write numbers.
