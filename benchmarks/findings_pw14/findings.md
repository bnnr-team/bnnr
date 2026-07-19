# Findings for T06 - resuming interrupted runs works correctly

## Method
- Google Colab runtime: CPU for smoke test, T4 for real grand benchmark
- Python version: 3.12.13
- Date of experiment: 02.07.2026
- Seed: 14 
- Conditions: no_aug, bnnr_xai
- Datasets: imagewoof for smoke test, eurosat for real grand benchmark
- Regime: scratch

## Environment
- BNNR version: 0.6.3
- Git commit: bafa5b4

## CPU Smoke Test
Command:
```bash
python benchmarks/run_grand_benchmark.py --dataset imagewoof --smoke --seed 14
```

Results are available [here](./results_imagewoof_scratch_smoke.json).

## Real Grand Benchmark
Command:
```bash
python benchmarks/run_grand_benchmark.py --dataset eurosat --device cuda --seed 14 --conditions no_aug,bnnr_xai
```
The run got killed with a classic KeyboardInterrupt after 2 epochs of baseline in bnnr_xai condition and then resumed by using the same command once again. 
The run resumed, starting the bnnr_xai condition from epoch 1 of baseline, then it didn't get interrupted until the end and finished the full condition.

Results are available [here](./results_eurosat_scratch.json).



