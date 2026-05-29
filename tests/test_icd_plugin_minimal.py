"""Smoke test for examples/classification/icd_plugin_minimal.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "classification" / "icd_plugin_minimal.py"


def test_icd_plugin_minimal_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "icd_out"
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    cache_files = list((out_dir / "xai_cache").glob("*.npy"))
    assert len(cache_files) > 0, "expected precomputed saliency .npy files"
