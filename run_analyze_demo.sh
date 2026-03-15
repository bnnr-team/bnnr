#!/usr/bin/env bash
# Train quick models on MNIST, CIFAR-10, STL-10 and run bnnr analyze on each.
# Run from repo root: ./run_analyze_demo.sh

set -e
cd "$(dirname "$0")"

CONFIG="config_quick.yaml"
DATA_DIR="data"

# Limit samples for speed (remove or increase for full runs)
MAX_TRAIN="--max-train-samples 3000"
MAX_VAL="--max-val-samples 500"

echo "=== 1. MNIST ==="
python3 -m bnnr train -c "$CONFIG" --dataset mnist --data-dir "$DATA_DIR" \
  -o out_mnist $MAX_TRAIN $MAX_VAL -e 2 --without-dashboard --no-xai

BEST_MNIST=$(python3 -c "
from pathlib import Path
import json
runs = sorted((Path('out_mnist/reports')).glob('run_*'), key=lambda p: p.name, reverse=True)
if not runs:
    raise SystemExit('No run dir found')
report = runs[0] / 'report.json'
data = json.loads(report.read_text())
best = max(data['checkpoints'], key=lambda c: c['metrics'].get('accuracy', 0))
p = Path(best['checkpoint_path'])
if not p.is_absolute():
    p = Path.cwd() / p
print(p.resolve())
")

echo ""
echo "=== 2. CIFAR-10 ==="
python3 -m bnnr train -c "$CONFIG" --dataset cifar10 --data-dir "$DATA_DIR" \
  -o out_cifar10 $MAX_TRAIN $MAX_VAL -e 2 --without-dashboard --no-xai

BEST_CIFAR=$(python3 -c "
from pathlib import Path
import json
runs = sorted((Path('out_cifar10/reports')).glob('run_*'), key=lambda p: p.name, reverse=True)
if not runs:
    raise SystemExit('No run dir found')
report = runs[0] / 'report.json'
data = json.loads(report.read_text())
best = max(data['checkpoints'], key=lambda c: c['metrics'].get('accuracy', 0))
p = Path(best['checkpoint_path'])
if not p.is_absolute():
    p = Path.cwd() / p
print(p.resolve())
")

echo ""
echo "=== 3. STL-10 ==="
python3 -m bnnr train -c config_quick_stl10.yaml \
  --dataset stl10 --data-dir "$DATA_DIR" -o out_stl10 \
  --max-train-samples 2000 --max-val-samples 500 -e 3 --without-dashboard --no-xai

BEST_STL=$(python3 -c "
from pathlib import Path
import json
runs = sorted((Path('out_stl10/reports')).glob('run_*'), key=lambda p: p.name, reverse=True)
if not runs:
    raise SystemExit('No run dir found')
report = runs[0] / 'report.json'
data = json.loads(report.read_text())
best = max(data['checkpoints'], key=lambda c: c['metrics'].get('accuracy', 0))
p = Path(best['checkpoint_path'])
if not p.is_absolute():
    p = Path.cwd() / p
print(p.resolve())
")

echo ""
echo "=== 4. ANALYZE: MNIST ==="
python3 -m bnnr analyze -m "$BEST_MNIST" -d mnist -o out_analyze_mnist

echo ""
echo "=== 5. ANALYZE: CIFAR-10 ==="
python3 -m bnnr analyze -m "$BEST_CIFAR" -d cifar10 -o out_analyze_cifar10

echo ""
echo "=== 6. ANALYZE: STL-10 ==="
python3 -m bnnr analyze -m "$BEST_STL" -d stl10 -o out_analyze_stl10

echo ""
echo "=============================================="
echo "  RAPORTY GOTOWE"
echo "=============================================="
echo ""
echo "  MNIST:     out_analyze_mnist/report.html"
echo "  CIFAR-10:  out_analyze_cifar10/report.html"
echo "  STL-10:    out_analyze_stl10/report.html"
echo ""
echo "Otwórz w przeglądarce np.:"
echo "  xdg-open out_analyze_mnist/report.html"
echo "  xdg-open out_analyze_cifar10/report.html"
echo "  xdg-open out_analyze_stl10/report.html"
echo ""
