#!/usr/bin/env bash
# Reproduce the ResNet50 / CIFAR-100 augmentation benchmark (5 seeds).
#
# Compares: no_bnnr | randaugment | trivialaugment | bnnr_branch_search
# on an ImageNet-pretrained ResNet50, CIFAR-100, identical training budget
# per condition (only the augmentation strategy varies).
#
# Usage:
#   bash benchmarks/reproduce_resnet50.sh                 # GPU, 5 seeds, defaults
#   DEVICE=cpu SEEDS=42 bash benchmarks/reproduce_resnet50.sh   # quick single-seed
#   EPOCHS=20 IMG_SIZE=224 bash benchmarks/reproduce_resnet50.sh # override knobs
#
# Environment overrides (all optional):
#   SEEDS      default "42,43,44,45,46"
#   DEVICE     default "auto"  (auto|cuda|cpu)
#   EPOCHS     default 15      (epochs per training phase)
#   IMG_SIZE   default 224     (CIFAR-100 resize target)
#   BATCH      default 64
#   RESULTS    default benchmarks/results_resnet50.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="${SEEDS:-42,43,44,45,46}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-15}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH="${BATCH:-64}"
RESULTS="${RESULTS:-benchmarks/results_resnet50.json}"

echo "=============================================================="
echo " ResNet50 / CIFAR-100 augmentation benchmark"
echo "   seeds=${SEEDS} device=${DEVICE} epochs=${EPOCHS} img_size=${IMG_SIZE}"
echo "   results -> ${RESULTS}"
echo "=============================================================="

# PYTHONPATH=src so the script runs from a clean clone without install.
PYTHONPATH="${PYTHONPATH:-}:src" python benchmarks/run_resnet50.py \
  --seeds "${SEEDS}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --img-size "${IMG_SIZE}" \
  --batch-size "${BATCH}" \
  --results "${RESULTS}"

echo ""
echo "Summary:"
PYTHONPATH="${PYTHONPATH:-}:src" python benchmarks/summarize.py \
  --results "${RESULTS}" --markdown

echo ""
echo "Next: review ${RESULTS}, then paste the table above into benchmarks/README.md"
echo "and the root README.md (do not hand-write numbers)."
