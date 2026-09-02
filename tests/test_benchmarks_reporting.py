"""The footer-to-column contract for ``benchmarks/summarize_grand.py`` (#398).

#398 closes when *"the footer describes the quantity the code actually
computes."* That is not a property of any single function, so it cannot be
tested at the estimator level: it is a property of the report as a whole. These
tests therefore run the summarizer as a subprocess on committed records, parse
what it printed, and recompute every number **from the raw JSON** — a different
path than the one that produced the table.

The distinction the report has to make, and which the footer previously
flattened into one sentence:

* ``Δ vs no_aug`` compares a *condition against no_aug*. It is a difference of
  condition medians, it is descriptive, and no test is attached to it.
* ``p (Holm)``, ``r`` and ``Bootstrap 95% CI`` in the same row compare
  *bnnr_xai against that condition* — a different pair — and the CI is the
  median of the paired differences.

Both are correct. Only a footer claiming a single convention for every Δ on the
page is wrong.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
IMAGEWOOF_JSON = BENCHMARKS_DIR / "results_imagewoof_scratch.json"


def _load(name: str) -> ModuleType:
    if str(BENCHMARKS_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS_DIR))
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sg() -> ModuleType:
    return _load("summarize_grand")


@pytest.fixture(scope="module")
def report() -> str:
    """Run the summarizer exactly as a user would, and capture what it printed."""
    assert IMAGEWOOF_JSON.is_file(), f"missing records: {IMAGEWOOF_JSON}"
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCHMARKS_DIR / "summarize_grand.py"),
            "--results-dir", str(BENCHMARKS_DIR),
            "--datasets", "imagewoof",
            "--markdown",
            "--bootstrap-n", "10000",
        ],
        capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture(scope="module")
def by_cond() -> dict[str, dict[int, float]]:
    """Per-condition, per-seed metric, read straight from the records."""
    runs = json.loads(IMAGEWOOF_JSON.read_text(encoding="utf-8"))["runs"]
    out: dict[str, dict[int, float]] = {}
    for r in runs:
        if r.get("dataset") not in (None, "imagewoof"):
            continue
        metric = r.get("held_out_test_metric")
        if metric is None:
            metric = r.get("val_metric")
        if metric is not None:
            out.setdefault(r["condition"], {})[int(r["seed"])] = float(metric)
    return out


def _paired(by_cond: dict[str, dict[int, float]], a: str, b: str) -> np.ndarray:
    seeds = sorted(set(by_cond[a]) & set(by_cond[b]))
    return np.array([by_cond[a][s] for s in seeds]) - np.array([by_cond[b][s] for s in seeds])


def _condition_rows(report: str) -> dict[str, list[str]]:
    """Parse the per-condition markdown table, keyed by display label."""
    rows: dict[str, list[str]] = {}
    for line in report.splitlines():
        if not line.startswith("| ") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 9 and cells[1].endswith("%"):
            rows[cells[0]] = cells
    return rows


# --------------------------------------------------------------------------- #
# The r column: matched-pairs, signed, bnnr_xai vs the condition
# --------------------------------------------------------------------------- #
def test_condition_table_r_is_matched_pairs_recomputed_from_records(
    report: str, by_cond: dict[str, dict[int, float]], sg: ModuleType
) -> None:
    rows = _condition_rows(report)
    assert rows, "could not parse the condition table"
    checked = 0
    for cid, label in sg.DISPLAY_NAMES.items():
        if cid == "bnnr_xai" or label not in rows or cid not in by_cond:
            continue
        printed = rows[label][7]
        if printed == "—":
            continue
        d = _paired(by_cond, "bnnr_xai", cid)
        nz = d[d != 0.0]
        ranks = sg.wilcoxon_signed_rank(d)
        expected = (ranks.w_plus - ranks.w_minus) / (ranks.w_plus + ranks.w_minus)
        assert float(printed) == pytest.approx(expected, abs=5e-3), f"{cid}: r"
        # and it is genuinely signed -- the defective statistic could not be
        assert -1.0 <= float(printed) <= 1.0
        assert len(nz) == ranks.n
        checked += 1
    assert checked >= 8, f"only {checked} conditions checked"


def test_some_printed_r_values_are_negative(report: str) -> None:
    """The defect #390 reported produced a statistic confined to [0.5, 1.0].

    A report in which no ``r`` is negative is indistinguishable from the old
    output, so this is the cheapest possible regression guard.
    """
    rows = _condition_rows(report)
    values = [float(c[7]) for c in rows.values() if c[7] not in ("—", "")]
    assert values, "no r values parsed"
    assert any(v < 0.0 for v in values), f"no negative r in {values}"


# --------------------------------------------------------------------------- #
# The two contrasts, kept apart
# --------------------------------------------------------------------------- #
def test_delta_vs_no_aug_is_a_difference_of_condition_medians(
    report: str, by_cond: dict[str, dict[int, float]], sg: ModuleType
) -> None:
    """``Δ vs no_aug`` compares a condition against no_aug, not against bnnr_xai.

    It is deliberately *not* a paired quantity: it is a descriptive gap between
    two condition medians with no test attached. The test also asserts that the
    paired median would have differed, so it cannot pass by coincidence.
    """
    import statistics

    rows = _condition_rows(report)
    ref = statistics.median(by_cond["no_aug"].values())
    disagreements = 0
    for cid, label in sg.DISPLAY_NAMES.items():
        if cid == "no_aug" or label not in rows or cid not in by_cond:
            continue
        printed = rows[label][5]
        if printed == "—":
            continue
        expected = (statistics.median(by_cond[cid].values()) - ref) * 100
        assert float(printed.rstrip("p")) == pytest.approx(expected, abs=5e-3), f"{cid}: Δ"
        paired = float(np.median(_paired(by_cond, cid, "no_aug"))) * 100
        if abs(paired - expected) > 1e-2:
            disagreements += 1
    assert disagreements > 0, "fixture cannot distinguish the two estimands"


def test_key_line_delta_is_the_median_of_paired_differences(
    report: str, by_cond: dict[str, dict[int, float]]
) -> None:
    """The headline Δ *is* paired — same pair as the p-value beside it."""
    m = re.search(
        r"KEY: bnnr_xai \(([\d.]+)%\) vs bnnr_random \(([\d.]+)%\).*?Δ=([+-][\d.]+)pp",
        report,
    )
    assert m, "could not find the KEY line"
    med_xai, med_rand, delta = float(m.group(1)), float(m.group(2)), float(m.group(3))

    expected = float(np.median(_paired(by_cond, "bnnr_xai", "bnnr_random"))) * 100
    assert delta == pytest.approx(expected, abs=5e-3)

    # ...and it is not the difference of the two medians printed next to it,
    # which is exactly the confusion the line has to label its way out of.
    assert abs(delta - (med_xai - med_rand)) > 1e-2
    key_line = next(ln for ln in report.splitlines() if "KEY: bnnr_xai" in ln)
    assert "[medians]" in key_line and "[median of paired diffs]" in key_line, (
        "the KEY line prints two marginal medians beside a paired Δ; it must say so"
    )


def test_cross_dataset_delta_is_the_median_of_paired_differences(
    report: str, by_cond: dict[str, dict[int, float]]
) -> None:
    m = re.search(r"\|\s*imagewoof\s*\|\s*\d+\s*\|[^|]+\|[^|]+\|\s*([+-][\d.]+)pp\s*\|", report)
    assert m, "could not find the cross-dataset imagewoof row"
    expected = float(np.median(_paired(by_cond, "bnnr_xai", "bnnr_random"))) * 100
    assert float(m.group(1)) == pytest.approx(expected, abs=5e-3)


def test_bootstrap_ci_column_is_the_paired_interval(
    report: str, by_cond: dict[str, dict[int, float]], sg: ModuleType
) -> None:
    """Recompute each interval from the raw records, not via the summarizer.

    An earlier version of this test obtained its expected value by calling
    ``sg._bootstrap_median_diff_ci`` — the very function that produced the
    printed cell — so it compared the implementation against itself. It passed
    with #390 defect 2 reintroduced, while the table went back to printing the
    retracted ``[-0.37, +0.27]pp``. The interval is the primary defect under
    review, so its guard has to resample here, from the JSON.
    """
    rows = _condition_rows(report)
    checked = 0
    for cid, label in sg.DISPLAY_NAMES.items():
        if cid == "bnnr_xai" or label not in rows or cid not in by_cond:
            continue
        cell = rows[label][8]
        m = re.match(r"\[([+-]?[\d.]+), ([+-]?[\d.]+)\]pp", cell)
        if not m:
            continue

        d = _paired(by_cond, "bnnr_xai", cid)
        n, n_boot, seed = len(d), 10_000, 42
        rng = np.random.default_rng(seed)
        meds = np.array([np.median(d[rng.integers(0, n, n)]) for _ in range(n_boot)])
        lo, hi = np.percentile(meds, [2.5, 97.5])

        assert float(m.group(1)) == pytest.approx(lo * 100, abs=5e-3), f"{cid}: CI lo"
        assert float(m.group(2)) == pytest.approx(hi * 100, abs=5e-3), f"{cid}: CI hi"

        # ...and it is demonstrably not the difference-of-medians interval #390
        # reported, which is the shape a regression would take.
        rng = np.random.default_rng(seed)
        xs = np.array([by_cond["bnnr_xai"][s] for s in sorted(set(by_cond["bnnr_xai"]) & set(by_cond[cid]))])
        ys = np.array([by_cond[cid][s] for s in sorted(set(by_cond["bnnr_xai"]) & set(by_cond[cid]))])
        unp = np.array([np.median(xs[i]) - np.median(ys[i])
                        for i in (rng.integers(0, n, n) for _ in range(n_boot))])
        u_lo, u_hi = np.percentile(unp, [2.5, 97.5])
        if abs(u_lo - lo) > 1e-5 or abs(u_hi - hi) > 1e-5:
            assert (float(m.group(1)), float(m.group(2))) != (
                pytest.approx(u_lo * 100, abs=5e-3), pytest.approx(u_hi * 100, abs=5e-3)
            ), f"{cid}: printed the difference-of-medians interval"
        checked += 1
    assert checked >= 8, f"only {checked} CIs checked"


# --------------------------------------------------------------------------- #
# The footer must name both contrasts, and must not claim one convention
# --------------------------------------------------------------------------- #
def test_footer_separates_the_two_contrasts(report: str) -> None:
    footer = report.split("Statistical notes", 1)
    assert len(footer) == 2, "no statistical notes footer"
    notes = footer[1]

    assert "Δ vs no_aug" in notes, "footer never names the Δ vs no_aug column"
    assert re.search(r"Δ vs no_aug.{0,200}difference of (condition )?medians", notes, re.S), (
        "footer must say Δ vs no_aug is a difference of condition medians"
    )
    assert re.search(r"median of the paired differences", notes), (
        "footer must say the CI is the median of the paired differences"
    )
    assert re.search(r"\(W\+ - W-\)/\(W\+ \+ W-\)", notes), "footer must give the r formula"

    # The blanket claim that produced the mismatch: one convention asserted for
    # every Δ on the page, immediately above a column that does not follow it.
    assert not re.search(
        r"Delta and bootstrap CI: median of the paired differences,\s*"
        r"median\(bnnr_xai - baseline\) — not a difference of medians\.",
        notes,
    ), "footer still claims a single convention for every Δ"


def test_footer_states_the_zero_and_tie_conventions(report: str) -> None:
    notes = report.split("Statistical notes", 1)[1]
    assert "Zeros" in notes and "dropped" in notes
    assert "Ties" in notes and "mid-ranks" in notes
    assert "Holm" in notes


# --------------------------------------------------------------------------- #
# The tools must be able to print their own output on a non-UTF-8 console
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("script", "args", "expect"),
    [
        ("summarize_grand.py",
         ["--results-dir", str(BENCHMARKS_DIR), "--datasets", "imagewoof", "--markdown"],
         "Δ"),
        ("summarize_spurious.py",
         [str(BENCHMARKS_DIR / "findings_t20" / "results_waterbirds_b15.json")],
         "Δ"),
    ],
)
def test_summarizers_print_under_a_non_utf8_stdout(script, args, expect) -> None:
    """These tools print Δ, W⁺, ≈, ≤, →. Windows defaults stdout to cp1252.

    Without ``force_utf8_stdout`` the run dies with ``UnicodeEncodeError`` partway
    through the report — the user waits for the whole thing and gets a traceback.
    Reproduced here by forcing the same encoding through ``PYTHONIOENCODING``,
    which fails identically on Linux.

    The fix belongs to the tool: setting ``PYTHONIOENCODING=utf-8`` in this test's
    environment would make it pass while leaving a real Windows user broken.
    """
    proc = subprocess.run(
        [sys.executable, str(BENCHMARKS_DIR / script), *args],
        capture_output=True, timeout=900, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert proc.returncode == 0, (
        f"{script} crashed under cp1252 stdout:\n"
        + proc.stderr.decode("utf-8", "replace")[-2000:]
    )
    assert expect in proc.stdout.decode("utf-8"), f"{script} lost its non-ASCII output"


def test_every_non_ascii_entry_point_forces_utf8() -> None:
    """A new script that prints non-ASCII must not reintroduce the crash.

    Static rather than behavioural on purpose: running every entry point would
    need GPUs and datasets, but the omission this guards against is visible in
    the source. Any ``benchmarks/*.py`` with a ``__main__`` block and a non-ASCII
    character in it has to call ``force_utf8_stdout()``.
    """
    offenders = []
    for path in sorted(BENCHMARKS_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "__main__" not in src or src.isascii():
            continue
        if "force_utf8_stdout()" not in src:
            offenders.append(path.name)
    assert not offenders, (
        "these entry points print non-ASCII but never call force_utf8_stdout(), "
        f"so they crash on a cp1252 console: {offenders}"
    )


# --------------------------------------------------------------------------- #
# The fill-ranking verdict: same pair, so it must use one estimand
# --------------------------------------------------------------------------- #
def test_meaningful_verdict_never_contradicts_the_printed_delta(sg: ModuleType) -> None:
    """The verdict and the Δ beside it describe the *same* pair, unlike the
    condition table's two columns — so they must agree in direction.

    Fixture: the challenger loses 12 of 13 seed pairs by a nose and wins one by a
    landslide. Its marginal median is far higher; its paired median difference,
    its rank-biserial and its bootstrap interval are all negative.
    """
    ref = [1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103, 104, 105]
    chal = [80, 1, 2, 3, 4, 5, 6, 99, 100, 101, 102, 103, 104]
    seeds = range(len(ref))
    by_strat = {
        "reference": {s: float(v) for s, v in zip(seeds, ref)},
        "challenger": {s: float(v) for s, v in zip(seeds, chal)},
    }

    res = sg._rank_strategies_for_method(
        by_strat, ["reference", "challenger"],
        no_bootstrap=False, n_boot=2000, reference="reference",
    )
    assert res is not None and not res["insufficient"]

    entry = next(e for e in res["entries"] if e["strategy"] == "challenger")
    assert entry["delta"] < 0.0, "fixture no longer produces a negative paired Δ"
    assert entry["r"] < 0.0
    assert res["meaningful"] is False, (
        "verdict declared a winner whose paired Δ, r and CI are all negative"
    )
