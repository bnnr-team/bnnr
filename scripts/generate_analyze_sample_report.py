#!/usr/bin/env python3
"""Regenerate docs/assets/analyze-report-sample.html (self-contained, embedded images).

Uses a local MNIST checkpoint if available (plans/reddit-launch/...), otherwise
runs a short train + analyze. Does not commit checkpoints to the repo.

Usage (from repo root):
  .venv-uv/bin/python scripts/generate_analyze_sample_report.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLE_OUT = REPO / "docs" / "assets" / "analyze-report-sample.html"
DEFAULT_CKPT = (
    REPO / "plans" / "reddit-launch" / "generated" / "mnist_train" / "checkpoints"
    / "iter_2_augmentation_1.pt"
)
WORK = REPO / ".tmp" / "analyze_sample_gen"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    out_dir = WORK / "analyze_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    ckpt = DEFAULT_CKPT
    if not ckpt.is_file():
        print(f"No checkpoint at {ckpt}; training short MNIST run...")
        train_out = WORK / "mnist_train"
        _run([
            sys.executable, "-m", "bnnr", "train",
            "--dataset", "mnist",
            "--data-dir", str(WORK / "data"),
            "--output", str(train_out),
            "--max-train-samples", "800",
            "--max-val-samples", "200",
            "--epochs", "2",
            "--device", "cpu",
        ])
        candidates = sorted(train_out.glob("checkpoints/*.pt"))
        if not candidates:
            raise SystemExit(f"No checkpoint produced under {train_out / 'checkpoints'}")
        ckpt = candidates[-1]

    _run([
        sys.executable, "-m", "bnnr", "analyze",
        "--model", str(ckpt),
        "--data", "mnist",
        "--output", str(out_dir),
        "--device", "cpu",
        "--max-worst", "12",
        "--xai-samples", "16",
    ])

    report_html = out_dir / "report.html"
    if not report_html.is_file():
        raise SystemExit(f"Missing {report_html}")

    html = report_html.read_text(encoding="utf-8")
    if 'src="artifacts/' in html:
        raise SystemExit(
            "report.html still contains relative artifact paths; "
            "ensure analyze CLI passes artifact_root to to_html()."
        )
    if "data:image/png;base64," not in html:
        raise SystemExit("report.html has no embedded PNG data URIs.")

    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_html, SAMPLE_OUT)
    size_kb = SAMPLE_OUT.stat().st_size / 1024
    print(f"Wrote {SAMPLE_OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
