"""Console output helpers for the benchmark scripts.

These tools print statistical notation — Δ, W⁺, W⁻, ≈, ≤, →, ±, § — as a matter
of course. Python picks stdout's encoding from the platform, and on Windows that
is cp1252, which cannot encode any of them. The result is not a mangled
character: it is ``UnicodeEncodeError`` mid-report, so the user gets a traceback
instead of a table, having already waited for the run.

The fix belongs to the tool, not to whoever invokes it. Requiring every caller to
set ``PYTHONIOENCODING`` would leave a Windows user running

    python benchmarks/summarize_grand.py --results-dir benchmarks/

with a crash and no clue why, and would hide the defect from CI rather than fix
it. Each entry point calls :func:`force_utf8_stdout` from its ``main()``.

Called from ``main()`` rather than at import, deliberately: importing a
summarizer as a library — which the test suite does — must not reconfigure the
importing process's streams.
"""
from __future__ import annotations

import sys

__all__ = ["force_utf8_stdout"]


def force_utf8_stdout() -> None:
    """Re-encode ``stdout`` and ``stderr`` as UTF-8, in place, where possible.

    Idempotent, and a no-op on streams that cannot be reconfigured — pytest's
    capture objects and plain file replacements have no ``reconfigure``, and a
    detached or closed stream raises. Neither case should stop a report being
    printed, so both are swallowed.

    Note this makes the *bytes* correct. A Windows console still needs code page
    65001 to render them, but writing UTF-8 to a cp1252 console produces mojibake
    at worst, where the default produces a traceback and no report at all.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # detached, closed, or not a TextIOWrapper
            pass
