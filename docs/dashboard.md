# Dashboard Guide (Live + Replay + Mobile)

## What you will find here
How to use dashboard as a daily workflow tool:
- live monitoring during training,
- replay after training,
- mobile access via QR,
- secure controls,
- export for offline sharing.

## 1) Live mode during training

Start training with dashboard:

```bash
python3 -m bnnr train \
  --config examples/configs/classification/cifar10_example.yaml \
  --dataset cifar10 \
  --max-train-samples 128 \
  --max-val-samples 64 \
  --preset light \
  --with-dashboard \
  --dashboard-port 8080
```

Terminal prints:
- Local URL
- Network URL
- QR code

Open Local URL on desktop first. Then use QR for phone.

## 2) Replay mode (no training)

Replay events from existing run(s):

```bash
python3 -m bnnr dashboard serve --run-dir reports --port 8080
```

Use replay when:
- you want to inspect old runs,
- you do post-mortem/debug,
- you demo results without retraining.

## 3) Mobile access with QR code

Requirements:
1. Phone and machine on same network.
2. Dashboard server running.
3. Firewall allows chosen port (default `8080`).

Steps:
1. Start dashboard (`train --with-dashboard` or `dashboard serve`).
2. Scan terminal QR code.
3. Open Network URL on phone.

If QR opens but page does not load, test Network URL manually in phone browser.

## 4) Secure pause/resume controls

Protect control endpoints with token:

```bash
python3 -m bnnr dashboard serve --run-dir reports --port 8080 --token "change-me"
```

Or set env var:

```bash
export BNNR_DASHBOARD_TOKEN="change-me"
python3 -m bnnr dashboard serve --run-dir reports --port 8080
```

For `train` path use:

```bash
python3 -m bnnr train ... --dashboard-token "change-me"
```

## 5) Pause/resume workflow

1. Start long run with dashboard.
2. Use Pause control in UI (or control endpoint if you automate).
3. Verify status changes to paused.
4. Resume and confirm progress continues.

## 6) Export static report (shareable)

```bash
python3 -m bnnr dashboard export \
  --run-dir reports/run_YYYYMMDD_HHMMSS \
  --out exported_dashboard
```

Artifacts include:
- `index.html`
- `data/events.jsonl`
- `data/state.json`
- copied run `artifacts/`

Open `exported_dashboard/index.html` locally without backend.

## 7) Recommended production workflow

1. Run training with dashboard and token.
2. Watch live branch decisions and KPI trends.
3. Stop server after validation (`Ctrl+C`).
4. Export static snapshot for review/share.
5. Archive `report.json`, `events.jsonl`, export dir.

## 8) Common issues

See `troubleshooting.md` for:
- dashboard dependencies missing,
- zero runs visible,
- mobile/QR/network access issues,
- export rendering issues.
