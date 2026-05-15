#!/usr/bin/env bash
# End-to-end smoke: classification (train → analyze) + synthetic multilabel demo (train → analyze).
# All artifacts go under $BNNR_SMOKE_RUN_DIR (default: ~/bnnr_manual_runs/smoke_<timestamp>), not the repo.
#
# Usage:
#   ./scripts/run_classification_multilabel_smoke.sh
#   BNNR_SMOKE_RUN_DIR="$HOME/my_runs/smoke1" ./scripts/run_classification_multilabel_smoke.sh
#   BNNR_DEVICE=cpu ./scripts/run_classification_multilabel_smoke.sh
#
# Classification dataset (default: stl10). STL-10 first download is ~2.6 GB — use mnist for a fast check:
#   BNNR_SMOKE_CLASSIFICATION=mnist ./scripts/run_classification_multilabel_smoke.sh
#
# After success, ~/bnnr_manual_runs/last_smoke_run → latest run (open .../stl_analyze/report.html).
# Avoid BNNR_SMOKE_RUN_DIR under /tmp — those paths disappear; prefer $HOME/bnnr_manual_runs/...
#   BNNR_OPEN_REPORTS=0  — do not xdg-open browsers
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${BNNR_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${BNNR_SMOKE_RUN_DIR:-$HOME/bnnr_manual_runs/smoke_${TS}}"
DEVICE="${BNNR_DEVICE:-auto}"
CLS_DS="${BNNR_SMOKE_CLASSIFICATION:-stl10}"

die() { echo "error: $*" >&2; exit 1; }

[[ -x "$PY" ]] || die "missing venv interpreter: $PY (run: cd \"$REPO_ROOT\" && uv sync)"
"$PY" -c "from torch import Tensor" 2>/dev/null || die "PyTorch in $PY is broken; fix .venv (uv sync)"

mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/stl_train" "$RUN_ROOT/stl_analyze" \
  "$RUN_ROOT/ml_work" "$RUN_ROOT/ml_analyze"

if [[ "$RUN_ROOT" == /tmp/* || "$RUN_ROOT" == /var/tmp/* ]]; then
  echo "⚠  RUN_ROOT is under tmp ($RUN_ROOT). Files may disappear after reboot or cleanup." >&2
  echo "   Use e.g. BNNR_SMOKE_RUN_DIR=\"\$HOME/bnnr_manual_runs/my_run\" for a stable path." >&2
  echo >&2
fi

echo "════════════════════════════════════════════════════════════════"
echo "  BNNR smoke run"
echo "  Repo:       $REPO_ROOT"
echo "  Artifacts:  $RUN_ROOT"
echo "  Python:     $PY"
echo "  Device:     $DEVICE"
echo "  Class. ds:  $CLS_DS"
echo "════════════════════════════════════════════════════════════════"
echo

case "$CLS_DS" in
  stl10)
    CLS_CONFIG="examples/configs/classification/stl10_showcase.yaml"
    CLS_MAX_TRAIN=2000
    CLS_MAX_VAL=500
    ;;
  mnist)
    CLS_CONFIG="examples/configs/classification/mnist_example.yaml"
    CLS_MAX_TRAIN=4000
    CLS_MAX_VAL=800
    ;;
  *)
    die "BNNR_SMOKE_CLASSIFICATION must be stl10 or mnist, got: $CLS_DS"
    ;;
esac

# --- Classification: train (cwd = repo for relative config paths) ---
echo "[1/4] $CLS_DS training (quick) …"
cd "$REPO_ROOT"
$PY -m bnnr train \
  --config "$CLS_CONFIG" \
  --dataset "$CLS_DS" \
  --data-dir "$RUN_ROOT/data" \
  --output "$RUN_ROOT/stl_train" \
  --device "$DEVICE" \
  --epochs 2 \
  --batch-size 32 \
  --max-train-samples "$CLS_MAX_TRAIN" \
  --max-val-samples "$CLS_MAX_VAL" \
  --preset light \
  --without-dashboard

mapfile -t STL_CKS < <(ls -t "$RUN_ROOT/stl_train/checkpoints"/iter_*.pt 2>/dev/null || true)
[[ ${#STL_CKS[@]} -gt 0 ]] || die "no checkpoints under $RUN_ROOT/stl_train/checkpoints"
STL_CKPT="${STL_CKS[0]}"
echo "  Using checkpoint: $STL_CKPT"

# analyze expects ./data relative to cwd for named datasets
echo "[2/4] $CLS_DS analyze …"
(
  cd "$RUN_ROOT"
  $PY -m bnnr analyze \
    --model "$STL_CKPT" \
    --data "$CLS_DS" \
    --output "$RUN_ROOT/stl_analyze" \
    --batch-size 32 \
    --device "$DEVICE" \
    --task classification \
    --cv-folds 3 \
    --xai-samples 120 \
    --no-data-quality
)
[[ -f "$RUN_ROOT/stl_analyze/report.html" ]] || die "missing $RUN_ROOT/stl_analyze/report.html"

# --- Multilabel demo: train in ml_work so multilabel_demo_output stays outside repo ---
echo "[3/4] Multilabel demo training (quick) …"
cd "$RUN_ROOT/ml_work"
$PY "$REPO_ROOT/examples/multilabel/multilabel_demo.py" \
  --quick \
  --without-dashboard

mapfile -t ML_CKS < <(ls -t "$RUN_ROOT/ml_work/multilabel_demo_output/checkpoints"/iter_*.pt 2>/dev/null || true)
[[ ${#ML_CKS[@]} -gt 0 ]] || die "no multilabel checkpoints under multilabel_demo_output/checkpoints"
ML_CKPT="${ML_CKS[0]}"
echo "  Using checkpoint: $ML_CKPT"

echo "[4/4] Multilabel analyze (Python API; CLI has no synthetic multilabel dataset) …"
export REPO_ROOT ML_CKPT RUN_ROOT
$PY << 'PY'
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

repo = Path(os.environ["REPO_ROOT"]).resolve()
ckpt_path = Path(os.environ["ML_CKPT"]).resolve()
out = Path(os.environ["RUN_ROOT"]) / "ml_analyze"
out.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "mld", repo / "examples/multilabel/multilabel_demo.py",
)
assert spec and spec.loader
mld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mld)

from bnnr import BNNRConfig, analyze_model
from bnnr.adapter import SimpleTorchAdapter

device = "cuda" if torch.cuda.is_available() else "cpu"
n_train, n_val = 200, 100
val_ds = mld._make_multilabel_dataset(n_val, seed=mld.SEED + 1000)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = mld.MultiLabelCNN(n_labels=mld.N_LABELS)
raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
if isinstance(state, dict) and "model" in state and "optimizer" in state:
    state = state["model"]
model.load_state_dict(state, strict=True)

adapter = SimpleTorchAdapter(
    model=model,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
    target_layers=[model.target_layer],
    device=device,
    multilabel=True,
    multilabel_threshold=0.5,
    eval_metrics=["fbeta_2", "f1_samples", "f1_macro", "accuracy"],
)

cfg = BNNRConfig(
    task="multilabel",
    device=device,
    metrics=["fbeta_2", "f1_samples", "f1_macro", "accuracy", "loss"],
)

report = analyze_model(
    adapter,
    val_loader,
    config=cfg,
    output_dir=out,
    run_data_quality=False,
    xai_enabled=True,
    xai_method="opticam",
    cv_folds=3,
    xai_samples=64,
)
report.to_html(out / "report.html")
print("Wrote:", out / "analysis_report.json", out / "report.html")
PY

[[ -f "$RUN_ROOT/ml_analyze/report.html" ]] || die "missing $RUN_ROOT/ml_analyze/report.html"

RUN_ABS="$(cd "$RUN_ROOT" && pwd)"
CLS_HTML="$RUN_ABS/stl_analyze/report.html"
ML_HTML="$RUN_ABS/ml_analyze/report.html"
LAST_LINK_ROOT="${BNNR_SMOKE_LAST_LINK:-$HOME/bnnr_manual_runs}"
mkdir -p "$LAST_LINK_ROOT"
LAST_LINK="$LAST_LINK_ROOT/last_smoke_run"
ln -sfn "$RUN_ABS" "$LAST_LINK"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Done."
echo "  Classification ($CLS_DS) HTML:  $CLS_HTML"
echo "  Multilabel HTML:               $ML_HTML"
echo "  JSON (classification): $RUN_ABS/stl_analyze/analysis_report.json"
echo "  JSON (multilabel):     $RUN_ABS/ml_analyze/analysis_report.json"
echo "  ────────────────────────────────────────────────────────────"
echo "  Shortcut (stable URL for next runs):"
echo "    $LAST_LINK/stl_analyze/report.html"
echo "    $LAST_LINK/ml_analyze/report.html"
echo "  file:// URLs (paste in browser if needed):"
echo "    file://$CLS_HTML"
echo "    file://$ML_HTML"
echo "════════════════════════════════════════════════════════════════"
if [[ "${BNNR_OPEN_REPORTS:-1}" == "1" ]] && command -v xdg-open >/dev/null 2>&1; then
  echo "Opening reports in browser (xdg-open) …"
  xdg-open "$CLS_HTML" 2>/dev/null || true
  xdg-open "$ML_HTML" 2>/dev/null || true
fi
