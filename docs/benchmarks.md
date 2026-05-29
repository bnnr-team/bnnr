# Benchmarks

CIFAR-10 study comparing **no augmentation**, **torchvision RandAugment**, and **BNNR branch search** (ICD + AICD + ChurchNoise), with validation metrics and OptiCAM attention overlays.

See [benchmarks/README.md](../benchmarks/README.md).

```bash
python benchmarks/run.py --seeds 42 --device cpu
python benchmarks/summarize.py --markdown
```

Results: [`benchmarks/results.json`](../benchmarks/results.json) (CIFAR-10 medians: no BNNR **75.3%**, BNNR branch search **81.4%**, RandAugment **72.5%**, 3 seeds). Attention maps: `benchmarks/runs/*/xai/`.

## Protocol caveat (read before comparing numbers)

The public table compares **different training budgets**:

- **`no_bnnr` and `randaugment`:** fixed **5 epochs** on the demo CNN — fast baselines, no branch search.
- **`bnnr_branch_search`:** full BNNR pipeline (baseline phase + screening branches with ICD/AICD) — **more wall-clock and more epochs** on the winning path.

So the Δaccuracy is **illustrative** (product vs simple baselines), not an equal-compute SOTA claim. Hardware, seeds, and full methodology: [`benchmarks/README.md`](../benchmarks/README.md#results-2026-05-28-seeds-4244-cpu).
