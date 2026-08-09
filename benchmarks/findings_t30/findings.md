# T30 — Oxford Pets generalization run

## Objective

Evaluate whether the grand-benchmark result generalizes to Oxford-IIIT Pets using the equal-compute protocol.

Issue: #338

## Run configuration

Dataset: Oxford-IIIT Pets  
Model: ResNet18  
Initialization: from scratch (`pretrained=False`)  
Image size: 224  
Classes: 37  
Seeds: 42, 43, 44, 45, 46, 47, 48  
Total compute: 40 GPU-epochs per condition

Conditions:

- `no_aug`
- `randaugment`
- `icd_only`
- `aicd_only`
- `bnnr_random`
- `bnnr_xai`

Hardware:

- NVIDIA GeForce RTX 5090
- PyTorch 2.10.0+cu128

Command:

```bash
python benchmarks/run_grand_benchmark.py \
  --dataset pets \
  --device cuda \
  --train-per-class 80