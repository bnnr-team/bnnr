# Findings

## Experiment plan:

1. Run the CPU smoke that exercises every code path: `python benchmarks/run_grand_benchmark.py --dataset imagewoof --smoke`.
2. On a free Colab or Kaggle T4, run one real (condition, seed), e.g. `no_aug seed 42`.
3. Kill it mid-run and restart to confirm resume-safety (completed work is skipped).

## Experiment environment:

- Machine: Google Colab (T4 GPU)
- Date: 01.07.2026
- BNNR version: 0.6.3
- Python version: 3.12.13

## CPU Smoke test

Command 

```bash
python benchmarks/run_grand_benchmark.py --dataset imagewoof --smoke --device cpu
```

Results:

- CPU Smoke passed as expected.
- [Pull #358](https://github.com/bnnr-team/bnnr/pull/358) solved the problem with running explicitly on cpu while gpu was visible

## Real grand benchmark run

Command 
```bash
python benchmarks/run_grand_benchmark.py --dataset eurosat --device cuda --seeds 67 --conditions no_aug,bnnr_xai
```

After completing `no_aug` and afew epochs of `bnnr_xai`, process was interupted. And ten resumed.

Results:

- Benchmark resumed and copleted as expected.

## Logs:

- commands used: `bnnr/benchmarks/findings_kuba/log.txt`
- grand benchmark result: `bnnr/benchmarks/findings_kuba/results_eurosat_scratch.json`
- cpu smoke result: `bnnr/benchmarks/findings_kuba/results_imagewoof_scratch.json`

