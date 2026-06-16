"""Human-readable console output, gated by a single ``verbose`` flag.

Kept separate from structured JSON logging (:func:`bnnr.utils.setup_logging`)
so that library consumers embedding :class:`~bnnr.trainer.BNNRTrainer` can
silence all terminal output by setting ``config.verbose=False`` without losing
machine-readable logs. Training-loop progress that previously used bare
``print()`` (some of it ungated) now routes through a single reporter.
"""

from __future__ import annotations


class ConsoleReporter:
    """Print progress to stdout only when ``verbose`` is set.

    A thin, dependency-free wrapper so every human-facing line in the training
    stack honours one flag. ``flush=True`` by default to keep ordering sane when
    interleaved with tqdm bars.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def print(self, message: str = "", *, flush: bool = True) -> None:
        """Write *message* to stdout when verbose; a no-op otherwise."""
        if self.verbose:
            print(message, flush=flush)
