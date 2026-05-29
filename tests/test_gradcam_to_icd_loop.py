"""Smoke test for examples/integrations/gradcam_to_icd_loop.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "integrations" / "gradcam_to_icd_loop.py"


def test_gradcam_to_icd_loop_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "gradcam_out"
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
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    for name in ("gradcam_overlay.png", "icd_augmented.png"):
        path = out_dir / name
        assert path.is_file(), f"missing {name}"
        assert path.stat().st_size > 1024
