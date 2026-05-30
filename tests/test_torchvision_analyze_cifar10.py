"""Smoke test for examples/classification/torchvision_analyze_cifar10.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "classification" / "torchvision_analyze_cifar10.py"


def test_torchvision_analyze_cifar10_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "analyze_out"
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--synthetic",
            "--no-xai",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    html = out_dir / "report.html"
    assert html.is_file(), f"missing {html}"
    assert html.stat().st_size > 1024
