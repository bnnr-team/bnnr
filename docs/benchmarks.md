# Benchmarks

Reproducible **baseline vs BNNR** comparisons for built-in CLI datasets.

## Protocol

| Setting | Baseline | BNNR |
|---------|----------|------|
| Branching | `max_iterations: 0` | `max_iterations: 3` |
| Augmentations | preset `none` | preset `light` |
| Model | Built-in demo CNN from `bnnr.pipelines` | Same |
| Metric | Validation `accuracy` | Same |

See [benchmarks/README.md](../benchmarks/README.md) for commands and `results.json` format.

## Run locally

```bash
python benchmarks/run_benchmarks.py --pilot --dataset cifar10 --seed 42
python benchmarks/update_readme_table.py
```

## If results are modest

Publish them anyway. BNNR also provides **qualitative** evidence (XAI heatmaps, branch ledger) — see `docs/assets/xai-same-accuracy-diff-behavior.png` in the README.

## Full matrix (optional)

```bash
python benchmarks/run_benchmarks.py \
  --datasets cifar10,fashion_mnist,stl10 \
  --seeds 42,43,44

python benchmarks/update_readme_table.py --aggregate median
```
