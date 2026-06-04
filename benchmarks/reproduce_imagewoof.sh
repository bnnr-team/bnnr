#!/usr/bin/env bash
# Reproduce the ResNet18 / Imagewoof augmentation benchmark (5 seeds).
#
# Fine-grained, low-data, from-scratch regime — compares:
#   no_aug | randaugment | trivialaugment | bnnr_branch_search
# on a ResNet18 (random init), identical training budget per condition
# (only the augmentation strategy varies). Cheap enough for a free Colab T4.
#
# Usage:
#   bash benchmarks/reproduce_imagewoof.sh                       # GPU, 5 seeds
#   DEVICE=cpu SEEDS=42 bash benchmarks/reproduce_imagewoof.sh   # quick single-seed
#   EPOCHS=30 TRAIN_PER_CLASS=150 bash benchmarks/reproduce_imagewoof.sh
#
# Environment overrides (all optional):
#   SEEDS            default "42,43,44,45,46"
#   DEVICE           default "auto"  (auto|cuda|cpu)
#   EPOCHS           default 25      (epochs per training phase)
#   IMG_SIZE         default 128
#   BATCH            default 64
#   TRAIN_PER_CLASS  default 100     (balanced images per class)
#   RESULTS          default benchmarks/results_imagewoof.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="${SEEDS:-42,43,44,45,46}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-25}"
IMG_SIZE="${IMG_SIZE:-128}"
BATCH="${BATCH:-64}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-100}"
RESULTS="${RESULTS:-benchmarks/results_imagewoof.json}"

echo "=============================================================="
echo " ResNet18 / Imagewoof augmentation benchmark (low-data, scratch)"
echo "   seeds=${SEEDS} device=${DEVICE} epochs=${EPOCHS} img_size=${IMG_SIZE}"
echo "   train_per_class=${TRAIN_PER_CLASS}  results -> ${RESULTS}"
echo "=============================================================="

PYTHONPATH="${PYTHONPATH:-}:src" python benchmarks/run_imagewoof.py \
  --seeds "${SEEDS}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --img-size "${IMG_SIZE}" \
  --batch-size "${BATCH}" \
  --train-per-class "${TRAIN_PER_CLASS}" \
  --results "${RESULTS}"

echo ""
echo "Summary:"
PYTHONPATH="${PYTHONPATH:-}:src" python benchmarks/summarize.py \
  --results "${RESULTS}" --markdown

echo ""
echo "Next: review ${RESULTS}, then paste the table above into benchmarks/README.md"
echo "and the root README.md (do not hand-write numbers)."
