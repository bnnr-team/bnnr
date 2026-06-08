"""Dependency-free constants shared by the dashboard and the trainer.

This module deliberately has no imports beyond the standard library so the
core training loop can read the pause-signal filename without importing the
FastAPI backend (and therefore without requiring the optional ``dashboard``
dependencies).
"""

from __future__ import annotations

# Signal file the dashboard writes into a run directory to request a pause.
# The trainer polls for it at epoch boundaries; see BNNRTrainer._check_pause.
PAUSE_SIGNAL_FILENAME = ".bnnr_pause"
