"""CPU pytest smoke for the SpuriousBench harness (P2b / T6).

Runs in <3 min on CPU with tiny synthetic data and NO downloads. Three guards:
  1. `_apply_batch_aug` actually mutates pixels (regression guard against the
     P2-core pass-through bug returning).
  2. The dataset class survives pickling under a spawn/forkserver start method
     (the 3.14 portability failure found earlier must stay fixed).
  3. The summarizer's statistics are correct on a controlled fixture (pure numpy,
     no torch) — Wilcoxon / Holm / Wilson / bootstrap.

torch-dependent tests are importorskip-guarded so the file still collects and the
summarizer test still runs in a torch-less environment; on Filip's machine (torch
present) all three run.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, BENCH_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: Python 3.14's dataclasses._is_type resolves string
    # annotations via sys.modules[cls.__module__].__dict__. If the module isn't
    # registered, that lookup returns None and @dataclass raises AttributeError.
    # (This is a test-loader requirement, not a defect in the module itself.)
    import sys
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


# --------------------------------------------------------------------------- #
# 1. pixels-actually-change (torch)
# --------------------------------------------------------------------------- #
def test_apply_batch_aug_mutates_pixels() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("bnnr")
    sr = _load_module("spurious_repair")
    from bnnr.augmentations import ChurchNoise

    # ChurchNoise is device_compatible and label-free — cheapest real aug to prove
    # the dispatch touches pixels. [0,1] batch, as the training path supplies.
    x = torch.rand(4, 3, 32, 32)
    y = torch.zeros(4, dtype=torch.long)
    idx = torch.arange(4)
    aug = ChurchNoise(probability=1.0, intensity=0.5,
                      noise_strength_range=(3.0, 8.0), random_state=0)
    out = sr._apply_batch_aug(aug, x.clone(), y, idx)
    delta = float((out - x).abs().mean().item())
    assert delta > 1e-6, f"augmentation did not change pixels (delta={delta}) — pass-through regression"


# --------------------------------------------------------------------------- #
# 2. pickling under spawn (dataset class must be top-level / picklable)
# --------------------------------------------------------------------------- #
def _roundtrip_pickle(obj):  # top-level so it is itself spawn-safe
    return pickle.loads(pickle.dumps(obj))


def test_dataset_class_pickles_under_spawn() -> None:
    pytest.importorskip("torch")
    sr = _load_module("spurious_repair")
    # The dataset class must be importable by reference (module-level), which is
    # the precondition for DataLoader workers under spawn/forkserver.
    ds_cls = sr._SpuriousDS
    assert ds_cls.__module__ and ds_cls.__qualname__ == "_SpuriousDS"
    # A spawn context must be able to pickle the class object itself.
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        returned = pool.apply(_roundtrip_pickle, (ds_cls,))
    assert returned is ds_cls or returned.__qualname__ == "_SpuriousDS"


# --------------------------------------------------------------------------- #
# 3. summarizer statistics on a controlled fixture (pure numpy — always runs)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fixture_path(tmp_path: Path) -> Path:
    """10-seed fixture: xai beats random by a consistent ~+3pp; dfr >> erm."""
    import random
    random.seed(1)
    runs = []

    def rec(cond, seed, wga, curve=None):
        return {
            "dataset": "waterbirds", "condition": cond, "seed": seed, "budget": 15,
            "worst_group_acc": wga, "weighted_mean_acc": 0.97, "plain_mean_acc": 0.85,
            "avg_minus_worst_pp": (0.97 - wga) * 100, "worst_group_id": 2,
            "group_acc": {"landbird_land": 0.99, "waterbird_land": wga},
            "ebpg_mean": 0.27, "iou_mean": 0.24, "pg_mean": 0.55,
            "ebpg_hard": 0.27, "ebpg_easy": 0.27,
            "wga_per_epoch": curve or [wga], "ebpg_per_epoch": [],
            "selected_candidate": "", "wall_clock_s": 100.0,
            "fill_strategy": "gaussian_blur", "target_layer": "layer4",
            "pipeline_version": "p2-norm-after-aug",
            "faith_probe_size": 800, "faith_batch_size": 16,
            "base_diagnosis": {"is_broken": True},
        }

    for s in range(10):
        br = 0.60 + random.uniform(-0.01, 0.01)
        runs.append(rec("bnnr_random", s, br))
        runs.append(rec("bnnr_xai", s, br + 0.03 + random.uniform(0.0, 0.005)))
        runs.append(rec("erm_continue", s, 0.66 + random.uniform(-0.01, 0.01)))
        runs.append(rec("dfr", s, 0.83 + random.uniform(-0.01, 0.01)))
    p = tmp_path / "results_waterbirds.json"
    p.write_text(json.dumps({"runs": runs}))
    return p


def test_wilcoxon_matches_known_value() -> None:
    """D-EXACT-P: n=10, all-positive, NO ties in |d| -> exact p = 2/1024.

    Only 2 of the 2**10 sign assignments are at least as extreme as W-=0 (the
    all-positive and the all-negative one), so the exact two-sided p is
    2/1024 = 0.001953125 exactly.
    """
    sr = _load_module("summarize_spurious")
    # 10 all-positive paired diffs, all |d| DISTINCT -> exact branch applies.
    d = np.array([0.030, 0.031, 0.028, 0.035, 0.029,
                  0.033, 0.026, 0.032, 0.027, 0.034])
    assert len(np.unique(np.abs(d))) == len(d), "fixture must be tie-free"
    assert sr.wilcoxon_p_method(d) == "exact"
    w_plus, w_minus, p, r = sr.wilcoxon_signed_rank(d)
    assert w_minus == 0.0 and r == pytest.approx(1.0)
    assert p == pytest.approx(2 / 1024, abs=1e-9), f"Wilcoxon exact p drifted: {p}"


def test_wilcoxon_exact_null_distribution_sums_to_2n() -> None:
    """The DP null distribution must be a full enumeration of 2**n sign flips."""
    sr = _load_module("summarize_spurious")
    for n in (1, 5, 10, 17):
        counts = sr._wilcoxon_null_counts(n)
        assert int(counts.sum()) == 2**n
        assert len(counts) == n * (n + 1) // 2 + 1
        assert int(counts[0]) == 1  # only the all-positive assignment gives W-=0


def test_wilcoxon_falls_back_to_approx_on_ties() -> None:
    """A tie in |d| disqualifies exact enumeration -> labeled approx fallback."""
    sr = _load_module("summarize_spurious")
    # 0.03 appears twice -> tie in |d| -> approximation with tie correction.
    d = np.array([0.03, 0.031, 0.028, 0.035, 0.029, 0.033, 0.03, 0.032, 0.027, 0.034])
    assert len(np.unique(np.abs(d))) < len(d), "fixture must contain a tie"
    assert sr.wilcoxon_p_method(d) == "approx"
    w_plus, w_minus, p, r = sr.wilcoxon_signed_rank(d)
    assert w_minus == 0.0 and r == pytest.approx(1.0)
    assert p == pytest.approx(0.00592, abs=1e-4), f"approx p drifted: {p}"


def test_wilcoxon_zero_diffs_force_approx() -> None:
    """Dropped zeros also disqualify the exact branch (n changes, D-EXACT-P)."""
    sr = _load_module("summarize_spurious")
    d = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert sr.wilcoxon_p_method(d) == "approx"


def test_holm_bonferroni_monotone() -> None:
    sr = _load_module("summarize_spurious")
    adj = sr.holm_bonferroni([0.006, 0.006, 0.006, 0.006, 0.006])
    assert all(a == pytest.approx(0.030, abs=1e-6) for a in adj)
    # monotonicity on distinct inputs
    adj2 = sr.holm_bonferroni([0.01, 0.02, 0.04])
    assert adj2 == sorted(adj2)


def test_wilson_ci_known_value() -> None:
    sr = _load_module("summarize_spurious")
    lo, hi = sr.wilson_ci(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3)
    assert hi == pytest.approx(0.943, abs=1e-3)


def test_epochs_to_threshold_dett() -> None:
    sr = _load_module("summarize_spurious")
    # rises to final=0.80; 0.80*final=0.64 first reached at epoch 3 (0.65)
    curve = [0.50, 0.60, 0.65, 0.72, 0.78, 0.80]
    prim, sec = sr.epochs_to_threshold(curve, budget=6)
    assert prim == 3          # 0.65 >= 0.64 (0.80*final)
    assert sec == 4           # 0.72 >= 0.70 first at epoch 4 (1-based)
    # never reaches 0.70 -> censored
    prim2, sec2 = sr.epochs_to_threshold([0.5, 0.55, 0.6], budget=3)
    assert sec2 == "censored"


def test_summarizer_report_builds_and_calibrates(fixture_path: Path) -> None:
    sr = _load_module("summarize_spurious")
    runs = sr.load_runs(fixture_path)
    report = sr.build_report(runs, label="test")
    # n=10 -> inferential path -> p-values present, Holm column present
    assert "p (Wilcoxon)" in report and "p (Holm)" in report
    assert "bnnr_xai vs bnnr_random" in report
    # xai > random consistently -> positive median delta
    assert "+3." in report or "+2." in report


# --------------------------------------------------------------------------- #
# 4. T0b artifact persistence: exception-safety + determinism (no torch needed
#    for the writer tests; the loop test is importorskip-guarded)
# --------------------------------------------------------------------------- #
def test_artifact_write_is_exception_safe(tmp_path: Path, capsys) -> None:
    """A persistence failure must warn and continue, NEVER raise.

    Amendment #1 §3.1: across a ~25 h matrix an artifact crash would cost far
    more than a missing figure input. Here np.savez_compressed is patched to
    raise; the writer must swallow it.
    """
    pytest.importorskip("torch")
    sr = _load_module("spurious_repair")
    from unittest import mock

    import numpy as _np

    maps = {"a.jpg": _np.zeros((4, 4))}
    meta = [{"id": "a.jpg", "img_path": "/x/a.jpg", "mask_path": "/x/a.png",
             "group": 0, "y": 0}]
    per_image = [{"id": "a.jpg", "group": 0, "ebpg": 0.3, "iou": 0.2, "pg": 1.0}]

    with mock.patch.object(_np, "savez_compressed",
                           side_effect=OSError("disk full")):
        sr._save_faith_artifacts(tmp_path / "saliency", "cond_s0",
                                 maps, meta, per_image)          # must not raise
        sr._save_probe_reference(tmp_path / "saliency", meta)    # must not raise
    assert "warning" in capsys.readouterr().out.lower()


def test_artifact_roundtrip_shapes_and_ids(tmp_path: Path) -> None:
    """Written npz carries aligned maps/ids and per-image vectors with ids."""
    pytest.importorskip("torch")
    sr = _load_module("spurious_repair")
    import numpy as _np

    maps = {"a.jpg": _np.ones((4, 4)), "b.jpg": _np.zeros((4, 4))}
    meta = [{"id": "a.jpg", "img_path": "/x/a.jpg", "mask_path": "/x/a.png",
             "group": 0, "y": 0},
            {"id": "b.jpg", "img_path": "/x/b.jpg", "mask_path": "/x/b.png",
             "group": 2, "y": 1}]
    per_image = [{"id": "a.jpg", "group": 0, "ebpg": 0.3, "iou": 0.2, "pg": 1.0},
                 {"id": "b.jpg", "group": 2, "ebpg": 0.5, "iou": 0.4, "pg": 0.0}]
    d = tmp_path / "saliency"
    sr._save_faith_artifacts(d, "bnnr_xai_s0", maps, meta, per_image)
    sr._save_probe_reference(d, meta)

    z = _np.load(d / "bnnr_xai_s0.npz", allow_pickle=True)
    assert z["maps"].shape == (2, 4, 4)
    assert z["maps"].dtype == _np.float16          # float16 for figures only
    assert list(z["map_ids"]) == ["a.jpg", "b.jpg"]
    assert list(z["ids"]) == ["a.jpg", "b.jpg"]    # per-image vectors addressable
    assert z["ebpg"].tolist() == pytest.approx([0.3, 0.5])
    assert (d / "_probe_reference.npz").exists()


def test_probe_reference_written_once(tmp_path: Path) -> None:
    """The reference is a dataset property; a second call must not rewrite it."""
    pytest.importorskip("torch")
    sr = _load_module("spurious_repair")
    meta = [{"id": "a.jpg", "img_path": "/x/a.jpg", "mask_path": "/x/a.png",
             "group": 0, "y": 0}]
    d = tmp_path / "saliency"
    sr._save_probe_reference(d, meta)
    mtime = (d / "_probe_reference.npz").stat().st_mtime_ns
    sr._save_probe_reference(d, meta)
    assert (d / "_probe_reference.npz").stat().st_mtime_ns == mtime


def test_saliency_subset_is_cap_invariant() -> None:
    """The first _SAL_PROBE_PER_GROUP per group must not depend on --faith-cap.

    This is the pre-registration property the whole design rests on: identical
    image ids across conditions/seeds make before/after and seed-averaging valid
    by construction. Replicates eval_faithfulness's selection rule exactly.
    """
    sr = _load_module("spurious_repair")
    n_groups, per = 4, sr._SAL_PROBE_PER_GROUP

    class E:
        def __init__(self, i, g):
            self.img_path, self.mask_path, self.group, self.y = f"{i}.jpg", "m", g, g % 2

    ex = [E(i, i % n_groups) for i in range(400)]

    def first_n(cap):
        counts = {g: 0 for g in range(n_groups)}
        sel = []
        for e in ex:
            if counts[e.group] >= cap:
                continue
            sel.append(e)
            counts[e.group] += 1
            if all(counts[g] >= cap for g in range(n_groups)):
                break
        kept, out = {g: 0 for g in range(n_groups)}, []
        for e in sel:
            if kept[e.group] < per:
                out.append(e.img_path)
                kept[e.group] += 1
        return out

    assert first_n(per) == first_n(500), "saliency subset changed with --faith-cap"


def test_summarizer_calibrates_language_at_small_n(tmp_path: Path) -> None:
    sr = _load_module("summarize_spurious")
    runs = [{
        "dataset": "w", "condition": c, "seed": 0, "budget": 15,
        "worst_group_acc": v, "weighted_mean_acc": 0.9, "plain_mean_acc": 0.8,
        "avg_minus_worst_pp": 10.0, "worst_group_id": 2, "group_acc": {},
        "ebpg_mean": 0.27, "iou_mean": 0.24, "pg_mean": 0.55,
        "wga_per_epoch": [v], "base_diagnosis": {"is_broken": True},
    } for c, v in [("bnnr_xai", 0.64), ("bnnr_random", 0.60)]]
    report = sr.build_report(runs, label=None)
    assert "direction and sign-consistency only" in report
    assert "p (Wilcoxon)" not in report  # suppressed at n=1
