"""``import bnnr`` should not eagerly import ``sklearn`` (fragile Colab numpy/scipy stacks)."""

from __future__ import annotations

import subprocess
import sys


def test_import_bnnr_in_fresh_process_does_not_load_sklearn() -> None:
    """Other tests may already import sklearn in-session; use a clean interpreter."""
    code = (
        "import sys\n"
        "import bnnr  # noqa: F401\n"
        "bad = [m for m in sys.modules if m.startswith('sklearn')]\n"
        "assert not bad, bad\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
