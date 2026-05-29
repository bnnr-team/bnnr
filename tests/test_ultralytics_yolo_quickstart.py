"""Smoke test for examples/integrations/ultralytics_yolo_quickstart.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ultralytics")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "integrations" / "ultralytics_yolo_quickstart.py"


def test_ultralytics_yolo_quickstart_smoke(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    data_dir = tmp_path / "coco128"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT / "examples" / "integrations")]
        ),
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--quick",
            "--device",
            "cpu",
            "--report-dir",
            str(report_dir),
            "--data-dir",
            str(data_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    report_jsons = sorted(report_dir.glob("run_*/report.json"))
    assert report_jsons, f"missing run_*/report.json under {report_dir}"
