# Benchmarks

CIFAR-10 study comparing **no augmentation**, **torchvision RandAugment**, and **BNNR branch search** (ICD + AICD + ChurchNoise), with validation metrics and OptiCAM attention overlays.

See [benchmarks/README.md](../benchmarks/README.md).

```bash
python benchmarks/run.py --seeds 42 --device cpu
python benchmarks/summarize.py --markdown
```

Results: [`benchmarks/results.json`](../benchmarks/results.json) (CIFAR-10 medians: no BNNR **75.3%**, BNNR branch search **81.4%**, RandAugment **72.5%**, 3 seeds). Attention maps: `benchmarks/runs/*/xai/`. See protocol caveats in [`benchmarks/README.md`](../benchmarks/README.md#results-2026-05-28-seeds-4244-cpu).

Protocol caveat: the `no_bnnr` and `randaugment` baselines run for a fixed **5 epochs**, while `bnnr_branch_search` includes the full BNNR pipeline (baseline training plus branch screening), so it uses more compute and a different curriculum. That makes the comparison useful for illustrating the end-to-end workflow, but not a strict apples-to-apples training-budget bakeoff. For the complete methodology and raw caveats, see [`benchmarks/README.md`](../benchmarks/README.md).
