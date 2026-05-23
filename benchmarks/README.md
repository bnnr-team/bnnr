# BNNR reproducible benchmarks

Compare **baseline-only** training vs **BNNR branching** on built-in CLI datasets.

## What is measured

- **Model:** built-in demo CNN from `bnnr.pipelines` (not ResNet-18 — honest scope for CLI benchmarks).
- **Metric:** validation `accuracy` (primary), plus `f1_macro` in reports.
- **Baseline:** `max_iterations: 0` — baseline epochs only, no branch search.
- **BNNR:** same config + `max_iterations: 3`, augmentation preset `light`.

## Quick pilot (one evening, one GPU)

```bash
cd /path/to/bnnr
pip install -e ".[dashboard]"   # or your venv

python benchmarks/run_benchmarks.py --pilot --dataset cifar10 --seed 42
python benchmarks/update_readme_table.py
```

Outputs:

- `benchmarks/results.json` — machine-readable
- `benchmarks/runs/<timestamp>_*` — per-run reports (gitignored)

## Full run (optional)

```bash
python benchmarks/run_benchmarks.py \
  --datasets cifar10,fashion_mnist,stl10 \
  --seeds 42,43,44

python benchmarks/update_readme_table.py --aggregate median
```

## Publish results

1. Commit `benchmarks/results.json` and README table update.
2. In posts, cite **pilot** honestly if only one seed: “early results, 1 seed”.
3. If Δacc ≤ 0: still publish + lean on qualitative XAI assets in `docs/assets/`.

## Runtime tips

- Use `--device cuda` if available.
- STL-10 is slower; run it after CIFAR-10 pilot looks sane.
- `--fast` uses `configs/pilot_fast.yaml` (shorter smoke test).
