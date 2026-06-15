"""``import bnnr`` should not eagerly import ``cv2`` (slow, wheel-fragile in Colab)."""

from __future__ import annotations

import subprocess
import sys


def test_import_bnnr_in_fresh_process_does_not_load_cv2() -> None:
    """Other tests may already import cv2 in-session; use a clean interpreter."""
    code = (
        "import sys\n"
        "import bnnr  # noqa: F401\n"
        "assert 'cv2' not in sys.modules, 'cv2 imported eagerly by import bnnr'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_lazy_cv2_proxy_resolves_real_cv2() -> None:
    """The proxy must transparently delegate to the real cv2 on attribute use."""
    import cv2 as real_cv2

    from bnnr.utils import lazy_cv2

    assert lazy_cv2.cvtColor is real_cv2.cvtColor
    assert lazy_cv2.COLOR_BGR2RGB == real_cv2.COLOR_BGR2RGB
