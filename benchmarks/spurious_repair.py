#!/usr/bin/env python3
"""BNNR Spurious-Correlation Repair Benchmark (core / v1).

Question
--------
Take a model that is accurate *for the wrong reason* (it exploits a spurious
background). Do BNNR's XAI-guided augmentations REPAIR it -- raise worst-group
accuracy AND move its attention onto the object -- and does XAI-guided candidate
selection repair better/faster than random selection?

This is the repair counterpart to the T20 grand benchmark. See
BENCHMARK_METHODOLOGY.md for the full design rationale. This file is the CORE:
a working end-to-end skeleton (both datasets, all five conditions, all metrics,
resume-safe JSON, smoke mode) meant to be iterated on.

Conditions (all start from the SAME diagnosed broken base checkpoint, all get the
SAME extra compute budget B, differ only in what they do with it):
  base_frozen  : no further training (the broken reference point)
  erm_continue : keep training ERM, no augmentation (controls for "more epochs")
  dfr          : retrain last layer on group-balanced data (Kirichenko 2022 baseline)
  bnnr_random  : BNNR branch-search, candidate chosen at random (XAI ablation)
  bnnr_xai     : BNNR branch-search, candidate chosen by selection-val (method under test)

Metrics:
  Robustness : worst-group accuracy (WGA), avg-minus-worst gap, prevalence-weighted mean acc
  Faithfulness (vs ground-truth object mask): EBPG, saliency-IoU@0.5, Pointing Game
  Dynamics   : WGA + EBPG logged per epoch -> epochs-to-threshold

Datasets:
  waterbirds    -- fully auto-downloadable (Stanford images + Caltech CUB masks)
  hard_imagenet -- loader provided; needs ImageNet-1k access (masks auto from Box)

Usage:
  # fresh machine, Waterbirds only, smoke test:
  python benchmarks/spurious_repair.py --dataset waterbirds --download --smoke

  # real run:
  python benchmarks/spurious_repair.py --dataset waterbirds --download \
      --seeds 0,1,2,3,4,5,6,7,8,9 --device cuda \
      --output benchmarks/spurious_repair_out
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_THIS.parent))
from console import force_utf8_stdout  # noqa: E402

CONDITIONS = ["base_frozen", "erm_continue", "dfr", "bnnr_random", "bnnr_xai"]


# =========================================================================== #
# Dataset abstraction
# =========================================================================== #
@dataclass
class Example:
    img_path: str
    mask_path: str          # "" if no mask available
    y: int                  # class label
    group: int              # group id (class x spurious-attribute)
    split: int              # 0 train, 1 val, 2 test


@dataclass
class DatasetSpec:
    name: str
    num_classes: int
    group_names: dict[int, str]
    hard_groups: set[int]   # groups where spurious attr conflicts with class
    examples: list[Example]

    def by_split(self, split: int) -> list[Example]:
        return [e for e in self.examples if e.split == split]

    def train_group_prevalence(self) -> dict[int, float]:
        tr = self.by_split(0)
        counts: dict[int, int] = {}
        for e in tr:
            counts[e.group] = counts.get(e.group, 0) + 1
        n = max(1, len(tr))
        return {g: c / n for g, c in counts.items()}


# ---- Waterbirds loader (auto-downloadable) ---- #
_WB_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
_CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
_CUB_SEG_URL = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [download] {dest.name} already present.")
        return
    print(f"  [download] {url}\n           -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, 100 * count * block / total)
            print(f"\r           {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, _hook)  # noqa: S310
    print()
    tmp.rename(dest)


def _extract(archive: Path, target_dir: Path) -> None:
    import tarfile

    print(f"  [extract] {archive.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(target_dir)  # noqa: S202


def ensure_waterbirds(data_dir: Path, do_download: bool) -> tuple[Path, Path]:
    wb_dir = data_dir / "waterbird_complete95_forest2water2"
    cub_dir = data_dir / "CUB_200_2011"
    if do_download:
        arch = data_dir / "waterbirds.tar.gz"
        if not (wb_dir / "metadata.csv").exists():
            _download(_WB_URL, arch)
            _extract(arch, data_dir)
        cub_arch = data_dir / "CUB_200_2011.tgz"
        if not cub_dir.exists():
            _download(_CUB_URL, cub_arch)
            _extract(cub_arch, data_dir)
        seg_dir = cub_dir / "segmentations"
        if not seg_dir.exists():
            seg_arch = data_dir / "segmentations.tgz"
            _download(_CUB_SEG_URL, seg_arch)
            _extract(seg_arch, cub_dir)
            # some tars unpack class folders flat into cub_dir
            if not seg_dir.exists():
                import shutil

                seg_dir.mkdir(parents=True, exist_ok=True)
                for p in cub_dir.iterdir():
                    if p.is_dir() and p.name[0].isdigit() and p.name != "segmentations":
                        shutil.move(str(p), str(seg_dir / p.name))
    return wb_dir, cub_dir


def load_waterbirds(data_dir: Path, do_download: bool) -> DatasetSpec:
    import csv

    wb_dir, cub_dir = ensure_waterbirds(data_dir, do_download)
    meta = wb_dir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(
            f"Waterbirds metadata.csv not found at {meta}. Re-run with --download."
        )
    seg_root = cub_dir / "segmentations"
    examples: list[Example] = []
    with open(meta, newline="") as f:
        for r in csv.DictReader(f):
            y, place, split = int(r["y"]), int(r["place"]), int(r["split"])
            fn = r["img_filename"]
            seg = seg_root / fn.replace(".jpg", ".png")
            examples.append(
                Example(
                    img_path=str(wb_dir / fn),
                    mask_path=str(seg) if seg.exists() else "",
                    y=y,
                    group=y * 2 + place,
                    split=split,
                )
            )
    group_names = {0: "landbird_land", 1: "landbird_water",
                   2: "waterbird_land", 3: "waterbird_water"}
    return DatasetSpec("waterbirds", 2, group_names, {1, 2}, examples)


# ---- Hard ImageNet loader (needs ImageNet access) ---- #
def load_hard_imagenet(data_dir: Path, imagenet_dir: Path | None) -> DatasetSpec:
    """Loader stub wired to the official layout. Hard ImageNet ships only the
    masks; images require ImageNet-1k. This resolves examples when both are
    present and otherwise raises with clear instructions.

    NOTE (core v1): this returns the structure so the harness is dataset-agnostic;
    the exact class->group mapping and mask filename convention are finalized
    against the real download in the next iteration (kept explicit and small so
    it's easy to verify once the data is in hand)."""
    hin_dir = data_dir / "hardImageNet"
    if not hin_dir.exists() or imagenet_dir is None:
        raise FileNotFoundError(
            "Hard ImageNet needs (1) the mask/annotation pack from Box:\n"
            "  curl -L 'https://app.box.com/index.php?rm=box_download_shared_file"
            "&shared_name=ca7qlcfsqlfqul9rzgtuqhb2c6pm62qd&file_id=f_972129165893'"
            " -o hardImageNet.zip && unzip hardImageNet.zip -d "
            f"{data_dir}\n"
            "and (2) ImageNet-1k images (pass --imagenet-dir). If you don't have "
            "ImageNet, run with --dataset waterbirds only; the harness is "
            "dataset-agnostic and Waterbirds is fully auto-downloadable."
        )
    # Real resolution deferred to iteration 2 (needs the actual files to verify
    # the pickle ranking + mask path convention). Fail loudly rather than guess.
    raise NotImplementedError(
        "Hard ImageNet loader: file-layout resolution finalized in iteration 2 "
        "(requires the real Box pack + ImageNet to verify paths). Core v1 ships "
        "and is tested on Waterbirds; this stub is intentionally explicit."
    )


def load_dataset(name: str, data_dir: Path, do_download: bool,
                 imagenet_dir: Path | None) -> DatasetSpec:
    if name == "waterbirds":
        return load_waterbirds(data_dir, do_download)
    if name == "hard_imagenet":
        return load_hard_imagenet(data_dir, imagenet_dir)
    raise ValueError(f"Unknown dataset {name!r} (use waterbirds | hard_imagenet)")


# =========================================================================== #
# Torch dataset
# =========================================================================== #
class _SpuriousDS:
    """Map-style dataset for (image, label, group, mask).

    Deliberately defined at module level and NOT nested inside
    ``build_torch_ds``: locally-defined classes cannot be pickled, and
    ``DataLoader(num_workers>0)`` must pickle the dataset to hand it to worker
    processes under the ``spawn``/``forkserver`` start methods. Python 3.14
    made ``forkserver`` the default on Linux, so a nested class raises
    ``PicklingError: Can't pickle local object`` there while silently working
    on <=3.13 (``fork``, no pickling). See Return Packet P1.

    Does not subclass ``torch.utils.data.Dataset`` so that torch stays a lazy
    import; ``DataLoader`` duck-types map-style datasets via
    ``__getitem__``/``__len__``.
    """

    def __init__(self, examples: list[Example], tf: Any, mtf: Any,
                 img_size: int) -> None:
        self.examples = examples
        self.tf = tf
        self.mtf = mtf
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        e = self.examples[i]
        x = self.tf(Image.open(e.img_path).convert("RGB"))
        if e.mask_path:
            m = (self.mtf(Image.open(e.mask_path).convert("L"))[0] > 0.5).float()
        else:
            m = torch.zeros(self.img_size, self.img_size)
        # The 5th element is the dataset position, used as the ICD/AICD/XAICache
        # ``sample_index`` (Fix C). save_map persists only index-keyed maps, so
        # without this the cache never persists and ICD recomputes OptiCAM every
        # batch. Stable across epochs because the dataset ordering is fixed.
        return x, e.y, e.group, m, i


# ImageNet normalization constants, referenced from the single post-augmentation
# normalization point (D-NORM). Kept module-level so every model-input boundary
# (train, eval_groups, _val_acc, eval_faithfulness) normalizes identically.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
PIPELINE_VERSION = "p2-norm-after-aug"


def save_results_atomic(results: dict[str, Any], results_path: Any) -> None:
    """Atomic results write (T5). Write to a sibling .tmp then os.replace().

    os.replace() is atomic on POSIX (same filesystem), so a crash (Ctrl-C, kill
    -9, OOM) during the write can never leave a truncated results JSON: the
    reader sees either the old complete file or the new complete file, never a
    half-written one. Replaces the previous non-atomic write_text() that could
    corrupt the whole file — including already-computed expensive records — if
    interrupted mid-write.
    """
    from pathlib import Path

    results_path = Path(results_path)
    tmp = results_path.with_suffix(results_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    os.replace(tmp, results_path)


def normalize_batch(x: Any) -> Any:
    """Apply ImageNet normalization to a batch of UNNORMALIZED images in [0, 1].

    This is the single, shared normalization point mandated by D-NORM. Datasets
    now yield tensors in [0, 1] (ToTensor only, no Normalize); every path that
    feeds the model calls this helper, and in the training path it is applied
    *after* augmentation so ICD/AICD/ChurchNoise see real pixel values rather
    than a normalized (partly negative) tensor that the numpy augmentation path
    would clip to zero.
    """
    import torch

    mean = torch.as_tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def build_torch_ds(examples: list[Example], img_size: int, train: bool) -> Any:
    from torchvision import transforms

    # NOTE (D-NORM): Normalize deliberately NOT included here. The dataset yields
    # images in [0, 1]; normalization happens at the model-input boundary via
    # normalize_batch(), after augmentation. See PIPELINE_VERSION.
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor()]
    tf = transforms.Compose(ops)
    mtf = transforms.Compose([transforms.Resize((img_size, img_size)),
                              transforms.ToTensor()])
    return _SpuriousDS(examples, tf, mtf, img_size)


# A third copy of wilson_ci / bootstrap_paired_median / wilcoxon_signed_rank /
# holm_bonferroni used to live here. It had zero call sites in this file and
# carried the #390 defect verbatim (`r = 1 - 2*stat/(n*(n+1))`) plus a fallback
# that ranked |d| with no tie handling at all. Deleted in #398: the estimators
# live in benchmarks/stats.py, and "single implementation for the whole
# benchmarks/ tree" is only true if the unused copies go too.


# =========================================================================== #
# Faithfulness metrics vs ground-truth object mask
# =========================================================================== #
def _resize_nn(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    H, W = mask.shape
    ys = (np.arange(h) * H / h).astype(int).clip(0, H - 1)
    xs = (np.arange(w) * W / w).astype(int).clip(0, W - 1)
    return mask[ys][:, xs]


def faithfulness_metrics(sal: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    """Given a saliency map and a binary object mask, return EBPG, IoU@0.5, PG-hit.
    Returns None if the mask is empty/degenerate."""
    if mask.shape != sal.shape:
        mask = _resize_nn(mask, sal.shape)
    m = mask > 0.5
    if m.mean() < 0.01 or sal.sum() <= 0:
        return None
    # EBPG: fraction of saliency energy inside the mask
    ebpg = float((sal * m).sum() / sal.sum())
    # IoU@0.5: binarize saliency at half its max, IoU with mask
    s = sal / (sal.max() + 1e-12)
    sb = s >= 0.5
    inter = float((sb & m).sum())
    union = float((sb | m).sum())
    iou = inter / union if union > 0 else float("nan")
    # Pointing Game: is the saliency peak inside the mask?
    peak = np.unravel_index(int(np.argmax(sal)), sal.shape)
    pg = float(bool(m[peak]))
    return {"ebpg": ebpg, "iou": iou, "pg": pg}


# =========================================================================== #
# Model / adapter
# =========================================================================== #
def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# D-TL: target layer is the explicit ``layer4`` module output (the
# spurious-correlation literature standard), NOT the adapter's auto-resolved
# ``layer4[2].conv3``. Recorded per run as TARGET_LAYER_NAME.
TARGET_LAYER_NAME = "layer4"


def resolve_target_layers(model: Any) -> list[Any]:
    """Return the pre-registered target layer(s) for a ResNet-50 (D-TL).

    Passed explicitly everywhere target layers flow (adapter construction,
    ICD/AICD, generate_saliency_maps, precompute_cache) so the EBPG gate term
    and all faithfulness numbers use one fixed convention rather than the
    adapter's auto-resolution.
    """
    return [model.layer4]


def build_adapter(num_classes: int, device: str, lr: float, epochs: int,
                  pretrained: bool) -> Any:
    import torch
    from torch import nn
    from torchvision import models

    from bnnr.adapter import SimpleTorchAdapter

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    # D-TL: pass target_layers explicitly instead of relying on auto-resolution.
    return SimpleTorchAdapter(model=m, criterion=nn.CrossEntropyLoss(),
                              optimizer=opt, device=device, scheduler=sched,
                              target_layers=resolve_target_layers(m))


def train_epochs(adapter: Any, loader: Any, epochs: int, device: str,
                 aug: Any = None, freeze_backbone: bool = False,
                 log_prefix: str = "", verbose: bool = False) -> None:
    """Train for `epochs`. If aug is given, apply it to each batch on the
    UNNORMALIZED [0, 1] tensor (Fix A), then normalize before the forward pass.
    If freeze_backbone, only fc trains (used by DFR)."""

    from tqdm import tqdm

    model = adapter.model
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    # One-time proof that augmentation actually changes pixels (Fix A validation).
    # Gated behind verbose so it never clutters a matrix log. Logs the mean abs
    # delta on the first augmented batch; a pass-through would print 0.000000.
    _aug_proof_pending = aug is not None and verbose
    # Epoch-level progress bar shown by default so a training phase never looks
    # frozen (previously it printed nothing unless --verbose was set). The
    # --verbose per-epoch loss line is preserved below. tqdm is already a bnnr
    # dependency (pyproject); no new dependency is introduced.
    desc = (log_prefix or "  training").strip()
    for ep in tqdm(range(epochs), desc=desc, unit="ep", leave=False):
        model.train()
        run = seen = 0
        for x, y, _g, _m, idx in loader:
            x, y = x.to(device), y.to(device)
            # D-NORM: x arrives UNNORMALIZED in [0, 1]. Augment here (on real
            # pixel values), THEN normalize before the forward pass.
            if aug is not None:
                if _aug_proof_pending:
                    import torch as _t
                    x_aug = _apply_batch_aug(aug, x, y, idx)
                    delta = float(_t.abs(x_aug - x).mean().item())
                    print(f"  [aug-proof] {getattr(aug, 'name', type(aug).__name__)}: "
                          f"mean|Δpixel|={delta:.6f} on first batch "
                          f"(0.000000 would mean pass-through)", flush=True)
                    _aug_proof_pending = False
                    x = x_aug
                else:
                    x = _apply_batch_aug(aug, x, y, idx)
            x = normalize_batch(x)
            adapter.optimizer.zero_grad()
            loss = adapter.criterion(model(x), y)
            loss.backward()
            adapter.optimizer.step()
            run += float(loss.item()) * len(y)
            seen += len(y)
        if adapter.scheduler is not None and not freeze_backbone:
            adapter.scheduler.step()
        if log_prefix:
            print(f"  {log_prefix} epoch {ep+1}/{epochs} loss={run/max(1,seen):.4f}",
                  flush=True)


def _apply_batch_aug(aug: Any, x: Any, y: Any, idx: Any) -> Any:
    """Apply a single bnnr augmentation to an UNNORMALIZED [0, 1] batch (Fix A).

    Delegates to bnnr's own ``AugmentationRunner._apply_augmentation_list`` via a
    one-element runner, so the label-aware (ICD/AICD -> ``apply_batch_with_labels``
    with ``sample_indices``) and tensor-native (ChurchNoise -> ``apply_tensor``)
    dispatch paths are exactly the library's tested ones rather than a
    re-implementation. Consumes ``src/bnnr`` but modifies no public API.

    Contract: the runner round-trips tensor -> uint8 HWC -> tensor and returns a
    tensor in the same [0, 1] range it was given. Feeding it a *normalized* batch
    raises ``NormalisedInputError`` instead of augmenting it; hence this is called
    BEFORE ``normalize_batch`` in the training loop.
    """
    from bnnr.augmentation_runner import AugmentationRunner

    runner = AugmentationRunner([aug], async_prefetch=False)
    out, _ = runner.apply_batch(x, y, sample_indices=idx)
    return out


# =========================================================================== #
# Evaluation: per-group accuracy + faithfulness
# =========================================================================== #
def eval_groups(adapter: Any, loader: Any, device: str, n_groups: int
                ) -> dict[int, dict[str, int]]:
    import torch

    adapter.model.eval()
    pg = {g: {"c": 0, "n": 0} for g in range(n_groups)}
    with torch.no_grad():
        for x, y, g, _m, _idx in loader:
            # D-NORM: loader yields [0, 1]; normalize at the model boundary.
            pred = adapter.model(normalize_batch(x.to(device))).argmax(1).cpu()
            for gi, yi, pi in zip(g.tolist(), y.tolist(), pred.tolist()):
                pg[gi]["n"] += 1
                pg[gi]["c"] += int(pi == yi)
    return pg


def summarize_group_acc(pg: dict[int, dict[str, int]],
                        prevalence: dict[int, float]) -> dict[str, Any]:
    accs = {g: pg[g]["c"] / max(1, pg[g]["n"]) for g in pg}
    worst = min(accs.values()) if accs else float("nan")
    worst_g = min(accs, key=accs.get) if accs else -1
    # prevalence-weighted mean (Sagawa convention)
    wsum = sum(prevalence.get(g, 0.0) for g in accs)
    if wsum > 0:
        weighted = sum(accs[g] * prevalence.get(g, 0.0) for g in accs) / wsum
    else:
        weighted = float("nan")
    plain = sum(pg[g]["c"] for g in pg) / max(1, sum(pg[g]["n"] for g in pg))
    return {
        "group_acc": accs,
        # T0/D-GROUP-N: per-group test counts, carried so the summarizer can
        # compute the contractual Wilson CIs (plan §4) without re-running eval.
        "group_n": {g: pg[g]["n"] for g in pg},
        "worst_group_acc": worst,
        "worst_group_id": worst_g,
        "weighted_mean_acc": weighted,
        "plain_mean_acc": plain,
        "avg_minus_worst_pp": (weighted - worst) * 100 if not math.isnan(weighted) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# T0b / Change 5+6: artifact persistence (saliency maps + per-image faithfulness)
# --------------------------------------------------------------------------- #
# COMMITTED CONSTANT (pre-registration): the number of saliency maps kept per
# group. These are the FIRST 8 per group of the already-deterministic probe
# selection in eval_faithfulness -- NOT a second selection rule. That selection
# uses no RNG and is cap-invariant (the first 8 per group are the same whether
# --faith-cap is 8 or 500), so image ids are identical across every condition,
# every seed and every run. That is what makes before/after comparison and
# seed-averaging valid by construction rather than by assumption.
_SAL_PROBE_PER_GROUP = 8


def _example_id(e: Example) -> str:
    """Stable, human-readable image id. Example has no numeric id field, and
    img_path is unique within a split, so the basename is the natural key."""
    return Path(e.img_path).name


def _save_faith_artifacts(out_dir: Path, tag: str,
                          sal_maps: dict[str, np.ndarray],
                          sal_meta: list[dict[str, Any]],
                          per_image: list[dict[str, Any]]) -> None:
    """Write the saliency probe maps and per-image faithfulness vectors.

    HARD REQUIREMENT (amendment #1 §3.1): every write is exception-safe. A
    persistence failure logs a warning and returns; it must NEVER kill a run.
    The measurement is primary, the artifact is secondary -- across a ~25 h
    matrix a crash here would be far more costly than a missing figure input.

    Layout, all under ``<output>/saliency/`` (never ``benchmarks/``):
      ``<condition>_s<seed>.npz``
        maps      float16 [K, H, W]  -- figures/averaging ONLY, not metrics
        map_ids   str    [K]         -- aligned with maps
        map_group int    [K]
        map_y     int    [K]
        ebpg/iou/pg  float32 [M]     -- FULL probe set (M ~ 2000), per image
        ids       str    [M]         -- aligned with ebpg/iou/pg
        group     int    [M]
    Per-image vectors carry ids because ``faithfulness_metrics`` returns None
    for degenerate masks (<1% coverage, per the frozen contract), so entries are
    dropped and a bare parallel list would not be addressable.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        if sal_maps:
            ids = list(sal_maps.keys())
            payload["maps"] = np.stack([sal_maps[i] for i in ids]).astype(np.float16)
            payload["map_ids"] = np.array(ids, dtype=object)
            meta = {m["id"]: m for m in sal_meta}
            payload["map_group"] = np.array([meta[i]["group"] for i in ids], dtype=np.int16)
            payload["map_y"] = np.array([meta[i]["y"] for i in ids], dtype=np.int16)
        if per_image:
            payload["ids"] = np.array([r["id"] for r in per_image], dtype=object)
            payload["group"] = np.array([r["group"] for r in per_image], dtype=np.int16)
            for k in ("ebpg", "iou", "pg"):
                payload[k] = np.array([r[k] for r in per_image], dtype=np.float32)
        if payload:
            np.savez_compressed(out_dir / f"{tag}.npz", **payload)
    except Exception as ex:  # noqa: BLE001 - artifact write must never kill a run
        print(f"  [artifacts] warning: failed to save {tag}: {ex}", flush=True)


def _save_probe_reference(out_dir: Path, sal_meta: list[dict[str, Any]]) -> None:
    """Write image paths + mask paths for the saliency probe ONCE.

    These are dataset properties (identical across every condition and seed), so
    storing them per run would duplicate them 50 times. Exception-safe; skipped
    if the file already exists.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ref = out_dir / "_probe_reference.npz"
        if ref.exists() or not sal_meta:
            return
        np.savez_compressed(
            ref,
            ids=np.array([m["id"] for m in sal_meta], dtype=object),
            img_path=np.array([m["img_path"] for m in sal_meta], dtype=object),
            mask_path=np.array([m["mask_path"] for m in sal_meta], dtype=object),
            group=np.array([m["group"] for m in sal_meta], dtype=np.int16),
            y=np.array([m["y"] for m in sal_meta], dtype=np.int16),
        )
    except Exception as ex:  # noqa: BLE001
        print(f"  [artifacts] warning: failed to save probe reference: {ex}", flush=True)


def eval_faithfulness(adapter: Any, examples: list[Example], device: str,
                      img_size: int, cap_per_group: int, n_groups: int,
                      faith_batch_size: int = 1,
                      artifact_dir: Path | None = None,
                      artifact_tag: str | None = None
                      ) -> dict[int, dict[str, list[float]]]:
    """Faithfulness metrics (EBPG/IoU/PG) vs ground-truth masks.

    T1 (D-PROBE) OUTCOME: batching was IMPLEMENTED (OptiCAMExplainer.explain
    accepts B>1 natively) but is NOT numerically equivalent to batch=1. OptiCAM
    optimizes weights with Adam on `-score.mean()`; the 1/B gradient scaling
    leaves a residual `+B*eps` term in the Adam denominator, so the resulting
    saliency depends on batch size. T1 measured EBPG max|Δ|≈0.047 (mean≈0.011) at
    batch=16 vs 1 — far above the 1e-3 tolerance. Per the pre-registered fallback,
    reported results therefore use `faith_batch_size=1`; the parameter is retained
    for experiments only. Making OptiCAM batch-invariant would require score.sum()
    in src/bnnr/xai.py, which is out of scope (the harness must stay compatible
    with upstream bnnr). The capped example set is still selected deterministically
    up front (stable order). D-EBPG-INPUT: saliency on NORMALIZED in-distribution
    inputs (frozen convention).
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    from bnnr.xai import generate_saliency_maps

    tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor()])
    mtf = transforms.Compose([transforms.Resize((img_size, img_size)),
                              transforms.ToTensor()])
    tl = resolve_target_layers(adapter.model)  # D-TL: explicit layer4
    out: dict[int, dict[str, list[float]]] = {
        g: {"ebpg": [], "iou": [], "pg": []} for g in range(n_groups)
    }
    # Deterministically select the capped set up front (stable input order), so
    # the work is fixed before batching and independent of batch size.
    counts = {g: 0 for g in range(n_groups)}
    selected: list[Example] = []
    for e in examples:
        if not e.mask_path:
            continue
        if counts[e.group] >= cap_per_group:
            continue
        selected.append(e)
        counts[e.group] += 1
        if all(counts[g] >= cap_per_group for g in range(n_groups)):
            break

    from tqdm import tqdm

    def _chunks(seq: list[Example], n: int):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    # T0b: the saliency-map subset is the FIRST _SAL_PROBE_PER_GROUP per group of
    # `selected` -- same deterministic order, no second selection rule.
    sal_keep: set[str] = set()
    sal_meta: list[dict[str, Any]] = []
    kept = {g: 0 for g in range(n_groups)}
    for e in selected:
        if kept[e.group] < _SAL_PROBE_PER_GROUP:
            eid = _example_id(e)
            sal_keep.add(eid)
            sal_meta.append({"id": eid, "img_path": e.img_path,
                             "mask_path": e.mask_path, "group": e.group, "y": e.y})
            kept[e.group] += 1
    sal_maps: dict[str, np.ndarray] = {}
    per_image: list[dict[str, Any]] = []

    for batch in tqdm(list(_chunks(selected, faith_batch_size)),
                      desc="  faithfulness (OptiCAM)", unit="batch", leave=False):
        imgs = torch.stack([tf(Image.open(e.img_path).convert("RGB")) for e in batch]).to(device)
        imgs = normalize_batch(imgs)  # D-NORM: shared normalization point
        ys = torch.tensor([e.y for e in batch]).to(device)
        sal = generate_saliency_maps(adapter.model, imgs, ys, tl, method="opticam")
        for j, e in enumerate(batch):
            m = (mtf(Image.open(e.mask_path).convert("L"))[0] > 0.5).numpy()
            sal_j = np.asarray(sal[j], float)
            fm = faithfulness_metrics(sal_j, m)
            if fm is not None:
                for k in ("ebpg", "iou", "pg"):
                    out[e.group][k].append(fm[k])
            # T0b/Change 6: per-image vectors, carrying the id. Recorded only for
            # images that produced metrics (fm is None => dropped by the frozen
            # <1%-coverage rule, so there is no value to record).
            eid = _example_id(e)
            if fm is not None:
                per_image.append({"id": eid, "group": e.group, "ebpg": fm["ebpg"],
                                  "iou": fm["iou"], "pg": fm["pg"]})
            # T0b/Change 5: keep the MAP regardless of fm -- the map exists even
            # when the mask is degenerate, and a figure may still want it.
            if eid in sal_keep:
                sal_maps[eid] = sal_j

    if artifact_dir is not None and artifact_tag is not None:
        _save_probe_reference(artifact_dir, sal_meta)
        _save_faith_artifacts(artifact_dir, artifact_tag, sal_maps, sal_meta, per_image)
    return out


# =========================================================================== #
# Repair conditions
# =========================================================================== #
def group_balanced_subset(examples: list[Example], seed: int, per_group: int
                          ) -> list[Example]:
    rng = random.Random(seed)
    by_g: dict[int, list[Example]] = {}
    for e in examples:
        by_g.setdefault(e.group, []).append(e)
    out: list[Example] = []
    for g, lst in by_g.items():
        rng.shuffle(lst)
        out.extend(lst[:per_group])
    rng.shuffle(out)
    return out


@dataclass
class RepairResult:
    dataset: str
    condition: str
    seed: int
    budget: int
    worst_group_acc: float = float("nan")
    weighted_mean_acc: float = float("nan")
    plain_mean_acc: float = float("nan")
    avg_minus_worst_pp: float = float("nan")
    worst_group_id: int = -1
    group_acc: dict[str, float] = field(default_factory=dict)
    # T0/D-GROUP-N: per-group test counts keyed by group NAME, aligned with
    # group_acc (schema ADDITION; existing keys unchanged). Enables Wilson CIs.
    group_n: dict[str, int] = field(default_factory=dict)
    ebpg_mean: float = float("nan")
    iou_mean: float = float("nan")
    pg_mean: float = float("nan")
    ebpg_hard: float = float("nan")
    ebpg_easy: float = float("nan")
    wga_per_epoch: list[float] = field(default_factory=list)
    ebpg_per_epoch: list[float] = field(default_factory=list)  # T3/D-DYN
    selected_candidate: str = ""
    wall_clock_s: float = 0.0
    # Self-describing provenance (P2 schema additions; renames forbidden):
    fill_strategy: str = "gaussian_blur"
    target_layer: str = TARGET_LAYER_NAME
    pipeline_version: str = PIPELINE_VERSION
    faith_probe_size: int = 0        # T1/D-PROBE: masks actually probed
    faith_batch_size: int = 1        # T1: MUST be 1 (OptiCAM not batch-invariant)


# --------------------------------------------------------------------------- #
# T3 / D-DYN: per-epoch EBPG dynamics probe
# --------------------------------------------------------------------------- #
# A FIXED, deterministic 50/group probe of masked test images, selected once per
# (spec) with a committed seed so the same images are used every epoch, every
# condition, every run. This is the D-DYN pre-registration: indices are chosen in
# code, not tuned. Kept small (50/group) because a full-set EBPG per epoch is
# unaffordable; the endpoint EBPG still uses the D-PROBE faithfulness set.
_DYN_PROBE_SEED = 20260721   # committed constant (D-DYN); do not tune per run
_DYN_PROBE_PER_GROUP = 50


def build_dyn_probe(spec: DatasetSpec, n_groups: int) -> list[Example]:
    """Deterministic 50/group masked-probe subset for per-epoch EBPG (D-DYN)."""
    masked = [e for e in spec.by_split(2) if e.mask_path]
    by_g: dict[int, list[Example]] = {g: [] for g in range(n_groups)}
    for e in masked:
        by_g.setdefault(e.group, []).append(e)
    rng = random.Random(_DYN_PROBE_SEED)
    probe: list[Example] = []
    for g in range(n_groups):
        lst = sorted(by_g.get(g, []), key=lambda e: e.img_path)  # stable order
        rng.shuffle(lst)
        probe.extend(lst[:_DYN_PROBE_PER_GROUP])
    return probe


def ebpg_on_probe(adapter: Any, probe: list[Example], device: str,
                  img_size: int, n_groups: int, faith_batch_size: int) -> float:
    """Mean EBPG over the fixed D-DYN probe (uses the batched faithfulness path)."""
    if not probe:
        return float("nan")
    fm = eval_faithfulness(adapter, probe, device, img_size,
                           cap_per_group=_DYN_PROBE_PER_GROUP, n_groups=n_groups,
                           faith_batch_size=faith_batch_size)
    vals = [v for g in fm for v in fm[g]["ebpg"]]
    return float(np.mean(vals)) if vals else float("nan")


def run_condition(condition: str, base_state: dict[str, Any], spec: DatasetSpec,
                  args: argparse.Namespace) -> RepairResult:
    from torch.utils.data import DataLoader

    t0 = time.perf_counter()
    device = args.device
    seed = args._current_seed
    set_seed(seed)

    train_ex = spec.by_split(0)
    test_ex = spec.by_split(2)
    val_ex = spec.by_split(1) or test_ex  # waterbirds has a val split

    nw = 0 if args.smoke else args.num_workers
    train_loader = DataLoader(build_torch_ds(train_ex, args.img_size, True),
                              batch_size=args.batch_size, shuffle=True, num_workers=nw)
    test_loader = DataLoader(build_torch_ds(test_ex, args.img_size, False),
                             batch_size=args.batch_size, shuffle=False, num_workers=nw)

    adapter = build_adapter(spec.num_classes, device, args.lr, args.budget,
                            pretrained=not args.no_pretrained)
    adapter.model.load_state_dict(base_state["model"])

    prevalence = spec.train_group_prevalence()
    n_groups = len(spec.group_names)
    wga_curve: list[float] = []
    ebpg_curve: list[float] = []
    selected = ""

    # T3/D-DYN: fixed probe (built once); gated by --dynamics to avoid the
    # per-epoch OptiCAM cost on runs that don't need trajectories.
    dyn_probe = build_dyn_probe(spec, n_groups) if args.dynamics else []

    def _log_wga() -> float:
        pg = eval_groups(adapter, test_loader, device, n_groups)
        s = summarize_group_acc(pg, prevalence)
        return s["worst_group_acc"]

    def _log_ebpg() -> float:
        if not args.dynamics:
            return float("nan")
        return ebpg_on_probe(adapter, dyn_probe, device, args.img_size,
                             n_groups, args.faith_batch_size)

    if condition == "base_frozen":
        pass  # evaluate as-is

    elif condition == "erm_continue":
        for ep in range(args.budget):
            train_epochs(adapter, train_loader, 1, device)
            wga_curve.append(_log_wga())
            if args.dynamics:
                ebpg_curve.append(_log_ebpg())

    elif condition == "dfr":
        # retrain last layer on a group-balanced subset of the *val* split
        bal = group_balanced_subset(val_ex, seed, args.dfr_per_group)
        bal_loader = DataLoader(build_torch_ds(bal, args.img_size, True),
                                batch_size=args.batch_size, shuffle=True, num_workers=nw)
        for ep in range(args.budget):
            train_epochs(adapter, bal_loader, 1, device, freeze_backbone=True)
            wga_curve.append(_log_wga())
            if args.dynamics:
                ebpg_curve.append(_log_ebpg())

    elif condition in ("bnnr_xai", "bnnr_random"):
        selected = _run_bnnr_repair(adapter, base_state, train_loader, val_ex,
                                    test_loader, spec, args, condition,
                                    wga_curve, _log_wga, ebpg_curve, _log_ebpg,
                                    dyn_probe)
    else:
        raise ValueError(condition)

    # final robustness
    pg = eval_groups(adapter, test_loader, device, n_groups)
    rob = summarize_group_acc(pg, prevalence)

    # faithfulness
    fmetrics = eval_faithfulness(adapter, test_ex, device, args.img_size,
                                 args.faith_cap, n_groups,
                                 faith_batch_size=args.faith_batch_size,
                                 artifact_dir=Path(args.output) / "saliency",
                                 artifact_tag=f"{condition}_s{seed}")
    def _pool(gids: set[int], key: str) -> float:
        vals = [v for g in gids for v in fmetrics[g][key]]
        return float(np.mean(vals)) if vals else float("nan")
    all_g = set(range(n_groups))
    easy_g = all_g - spec.hard_groups
    probe_size = sum(len(fmetrics[g]["ebpg"]) for g in range(n_groups))

    res = RepairResult(
        dataset=spec.name, condition=condition, seed=seed, budget=args.budget,
        worst_group_acc=rob["worst_group_acc"],
        weighted_mean_acc=rob["weighted_mean_acc"],
        plain_mean_acc=rob["plain_mean_acc"],
        avg_minus_worst_pp=rob["avg_minus_worst_pp"],
        worst_group_id=rob["worst_group_id"],
        group_acc={spec.group_names[g]: rob["group_acc"].get(g, float("nan"))
                   for g in range(n_groups)},
        group_n={spec.group_names[g]: int(rob["group_n"].get(g, 0))
                 for g in range(n_groups)},
        ebpg_mean=_pool(all_g, "ebpg"), iou_mean=_pool(all_g, "iou"),
        pg_mean=_pool(all_g, "pg"),
        ebpg_hard=_pool(spec.hard_groups, "ebpg"),
        ebpg_easy=_pool(easy_g, "ebpg"),
        wga_per_epoch=wga_curve, ebpg_per_epoch=ebpg_curve,
        selected_candidate=selected,
        wall_clock_s=time.perf_counter() - t0,
        fill_strategy=args.fill_strategy,
        target_layer=TARGET_LAYER_NAME,
        pipeline_version=PIPELINE_VERSION,
        faith_probe_size=probe_size,
        faith_batch_size=args.faith_batch_size,
    )
    return res


def _run_bnnr_repair(adapter: Any, base_state: dict, train_loader: Any,
                     val_ex: list[Example], test_loader: Any, spec: DatasetSpec,
                     args: argparse.Namespace, condition: str,
                     wga_curve: list[float], log_wga: Callable[[], float],
                     ebpg_curve: list[float] | None = None,
                     log_ebpg: Callable[[], float] | None = None,
                     dyn_probe: list[Example] | None = None) -> str:
    """BNNR branch-search repair, mirroring T20's equal-compute structure:
    3 candidates (ICD, AICD, ChurchNoise) each trained budget//3 epochs from the
    base checkpoint; xai keeps best-on-val, random keeps a random one.

    T0/D-ETT-UNGATE: per-epoch WGA is logged for EACH candidate ALWAYS; after
    selection the WINNER's WGA curve is copied into wga_curve, so D-ETT is
    computable on the main matrix. Cost: 3 x budget//3 test-set forward passes
    per run (no OptiCAM), i.e. one extra test pass per candidate epoch.

    T3/D-DYN: when --dynamics is on, the per-epoch EBPG probe is ALSO run per
    candidate epoch and the winner's EBPG curve is copied into ebpg_curve. That
    probe is the expensive part and stays gated.

    ``log_wga`` is retained in the signature for call-site stability but is no
    longer used in the body: the winner's final candidate-epoch WGA already IS
    the deployed endpoint, so re-evaluating it would only duplicate work."""
    import torch
    from torch.utils.data import DataLoader

    from bnnr.augmentations import ChurchNoise
    from bnnr.icd import AICD, ICD
    from bnnr.xai_cache import XAICache

    device = args.device
    seed = args._current_seed
    n_cand = 3
    per = max(1, args.budget // n_cand)
    nw = 0 if args.smoke else args.num_workers
    # D-TL: explicit layer4 for the precompute pass (adapter already carries it,
    # but pass explicitly so the cache-priming layer is unambiguous).
    target_layers = resolve_target_layers(adapter.get_model())
    val_loader = DataLoader(build_torch_ds(val_ex, args.img_size, False),
                            batch_size=args.batch_size, shuffle=False, num_workers=nw)

    run_dir = Path(args.output) / f"{spec.name}_bnnr" / f"s{seed}_{condition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = XAICache(run_dir / "xai_cache")

    class _XYLoader:
        """Adapt our (x, y, group, mask, index) loader to the (x, y, index)
        batches ``precompute_cache`` expects (Fix C).

        Yields LEN-3 batches so ``sample_indices`` reach the cache: ``save_map``
        persists ONLY index-keyed maps, so a len-2 (x, y) yield would compute
        maps and write nothing, and every ICD/AICD batch would then miss and
        recompute OptiCAM online. Images are yielded UNNORMALIZED ([0, 1]) to
        match ICD's own online path (``_compute_online_batch`` feeds the model
        ``image/255``), so cache-hit and cache-miss maps share one input
        distribution.
        """
        def __init__(self, base: Any) -> None:
            self.base = base

        def __iter__(self):
            for x, y, _g, _m, idx in self.base:
                yield x, y, idx

        def __len__(self) -> int:
            return len(self.base)

        @property
        def dataset(self) -> Any:
            return self.base.dataset

    try:
        cache.precompute_cache(adapter.get_model(), _XYLoader(train_loader),
                               target_layers, method="opticam",
                               show_progress=not args.smoke)
    except Exception as ex:
        print(f"  [xai_cache] warning: {ex}")

    def _val_acc(ad: Any) -> float:
        ad.model.eval()
        c = t = 0
        with torch.no_grad():
            for x, y, _g, _m, _idx in val_loader:
                # D-NORM: normalize at the model boundary.
                p = ad.model(normalize_batch(x.to(device))).argmax(1).cpu()
                c += int((p == y).sum())
                t += len(y)
        return c / max(1, t)

    cand_names = ["ICD", "AICD", "ChurchNoise"]
    cand_states, cand_scores = [], []
    cand_wga_curves: list[list[float]] = []
    cand_ebpg_curves: list[list[float]] = []
    prevalence = spec.train_group_prevalence()
    n_groups = len(spec.group_names)
    dynamics = args.dynamics and dyn_probe is not None

    def _cand_wga(ad: Any) -> float:
        pg = eval_groups(ad, test_loader, device, n_groups)
        return summarize_group_acc(pg, prevalence)["worst_group_acc"]

    for i in range(n_cand):
        ad = build_adapter(spec.num_classes, device, args.lr, per, pretrained=not args.no_pretrained)
        ad.model.load_state_dict(base_state["model"])
        model, layers = ad.get_model(), ad.get_target_layers()
        aug = [
            ICD(model=model, target_layers=layers, threshold_percentile=75.0,
                probability=0.5, random_state=seed, cache=cache,
                fill_strategy=args.fill_strategy),
            AICD(model=model, target_layers=layers, threshold_percentile=75.0,
                 probability=0.5, random_state=seed + 1, cache=cache,
                 fill_strategy=args.fill_strategy),
            ChurchNoise(probability=0.5, intensity=0.5,
                        noise_strength_range=(3.0, 8.0), random_state=seed + 2),
        ][i]
        w_curve: list[float] = []
        e_curve: list[float] = []
        for ep in range(per):
            # Emit the pixel-change proof only on the very first epoch of the
            # very first candidate to keep the log clean.
            proof = args.verbose and ep == 0 and i == 0
            train_epochs(ad, train_loader, 1, device, aug=aug, verbose=proof)
            # T0/D-ETT-UNGATE: the per-epoch WGA trajectory is logged ALWAYS
            # (cheap: one forward pass over the test loader, no OptiCAM), so
            # D-ETT is computable on the main matrix without --dynamics.
            w_curve.append(_cand_wga(ad))
            if dynamics:
                # T3/D-DYN: the EBPG probe stays gated - it is the expensive one.
                e_curve.append(ebpg_on_probe(ad, dyn_probe, device, args.img_size,
                                             n_groups, args.faith_batch_size))
        cand_states.append(copy.deepcopy(ad.model.state_dict()))
        cand_scores.append(_val_acc(ad))
        cand_wga_curves.append(w_curve)
        cand_ebpg_curves.append(e_curve)
        print(f"  [{cand_names[i]}] val_acc={cand_scores[i]:.4f}", flush=True)

    if condition == "bnnr_xai":
        best = int(max(range(n_cand), key=lambda k: cand_scores[k]))
    else:
        best = random.Random(seed).randint(0, n_cand - 1)
    adapter.model.load_state_dict(cand_states[best])
    # T0/D-ETT-UNGATE: the WINNER's per-epoch WGA trajectory is kept ALWAYS
    # (last point == deployed endpoint, so no extra eval is needed here).
    # NOTE: the curve spans the winner's budget//3 candidate epochs, NOT the
    # full budget - D-ETT epoch indices for bnnr arms are on that shorter axis
    # and are not directly comparable to erm_continue's budget-length axis
    # (this is the B/3 epoch-split caveat, in trajectory form).
    wga_curve.extend(cand_wga_curves[best])
    # T3/D-DYN: the EBPG trajectory only exists when --dynamics was on.
    if dynamics and ebpg_curve is not None:
        ebpg_curve.extend(cand_ebpg_curves[best])
    return cand_names[best]


# =========================================================================== #
# Base model (trained + diagnosed) with resume
# =========================================================================== #
def train_and_diagnose_base(spec: DatasetSpec, args: argparse.Namespace, seed: int
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
    from torch.utils.data import DataLoader

    set_seed(seed)
    device = args.device
    nw = 0 if args.smoke else args.num_workers
    train_loader = DataLoader(build_torch_ds(spec.by_split(0), args.img_size, True),
                              batch_size=args.batch_size, shuffle=True, num_workers=nw)
    test_loader = DataLoader(build_torch_ds(spec.by_split(2), args.img_size, False),
                             batch_size=args.batch_size, shuffle=False, num_workers=nw)
    adapter = build_adapter(spec.num_classes, device, args.lr, args.base_epochs,
                            pretrained=not args.no_pretrained)
    print(f"[seed {seed}] training base ERM ({args.base_epochs} epochs)...", flush=True)
    train_epochs(adapter, train_loader, args.base_epochs, device,
                 log_prefix="[base]" if args.verbose else "")

    prevalence = spec.train_group_prevalence()
    pg = eval_groups(adapter, test_loader, device, len(spec.group_names))
    rob = summarize_group_acc(pg, prevalence)
    # T0b: the diagnosed base model IS the "before" in before/after, so its
    # artifacts are persisted under the reserved tag `base_diag_s<seed>`.
    fm = eval_faithfulness(adapter, spec.by_split(2), device, args.img_size,
                           args.faith_cap, len(spec.group_names),
                           faith_batch_size=args.faith_batch_size,
                           artifact_dir=Path(args.output) / "saliency",
                           artifact_tag=f"base_diag_s{seed}")
    ebpg_all = [v for g in fm for v in fm[g]["ebpg"]]
    diag = {
        "weighted_mean_acc": rob["weighted_mean_acc"],
        "worst_group_acc": rob["worst_group_acc"],
        "avg_minus_worst_pp": rob["avg_minus_worst_pp"],
        "ebpg_mean": float(np.mean(ebpg_all)) if ebpg_all else float("nan"),
        # T0/D-GROUP-N: base-model per-group counts (same addition as the
        # RepairResult schema; zero extra cost, enables Wilson CIs on the base).
        "group_n": {spec.group_names[g]: int(rob["group_n"].get(g, 0))
                    for g in spec.group_names},
    }
    # a-priori shortcut gate
    diag["is_broken"] = bool(
        diag["weighted_mean_acc"] >= args.diag_min_acc
        and diag["avg_minus_worst_pp"] >= args.diag_min_gap_pp
        and (math.isnan(diag["ebpg_mean"]) or diag["ebpg_mean"] <= args.diag_max_ebpg)
    )
    print(f"[seed {seed}] base diagnosis: acc={diag['weighted_mean_acc']*100:.1f}% "
          f"wga={diag['worst_group_acc']*100:.1f}% gap={diag['avg_minus_worst_pp']:.1f}pp "
          f"ebpg={diag['ebpg_mean']:.3f} -> broken={diag['is_broken']}", flush=True)
    return {"model": adapter.model.state_dict()}, diag


# =========================================================================== #
# Main
# =========================================================================== #
def main() -> None:
    force_utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="waterbirds",
                   choices=["waterbirds", "hard_imagenet"])
    p.add_argument("--data-dir", type=Path, default=Path.home() / "data")
    p.add_argument("--imagenet-dir", type=Path, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", type=Path, default=Path("benchmarks/spurious_repair_out"))
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--base-epochs", type=int, default=15)
    p.add_argument("--budget", type=int, default=15,
                   help="extra-compute budget for each repair condition")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--faith-cap", type=int, default=500,
                   help="max masked probe images per group for faithfulness "
                        "(D-PROBE default: 500/group = 2000 masks on Waterbirds)")
    p.add_argument("--faith-batch-size", type=int, default=1,
                   help="OptiCAM batch size for faithfulness. MUST be 1 for "
                        "reported results: OptiCAM uses score.mean() so its maps "
                        "are batch-size-dependent (T1 measured EBPG max|Δ|≈0.047 "
                        "at batch=16 vs 1). >1 is faster but NOT numerically "
                        "equivalent — experiments only, never P3. (Fixing this "
                        "would require score.sum() in src/bnnr/xai.py, out of "
                        "scope: the harness must stay compatible with upstream bnnr.)")
    p.add_argument("--dynamics", action="store_true",
                   help="T3/D-DYN: log per-epoch WGA + EBPG (50/group fixed probe) "
                        "for every training condition and the deployed bnnr winner. "
                        "Adds per-epoch OptiCAM overhead; off by default.")
    p.add_argument("--dfr-per-group", type=int, default=200)
    p.add_argument("--fill-strategy", default="gaussian_blur",
                   choices=["gaussian_blur", "local_mean", "global_mean",
                            "noise", "solid"],
                   help="ICD/AICD masked-region fill (bnnr default: gaussian_blur). "
                        "The fill ablation itself is deferred to P5; this only sets "
                        "the single value used per run and records it.")
    # diagnosis gate thresholds (pre-registered here in the config)
    p.add_argument("--diag-min-acc", type=float, default=0.80)
    p.add_argument("--diag-min-gap-pp", type=float, default=15.0)
    p.add_argument("--diag-max-ebpg", type=float, default=0.55)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip ImageNet weights (offline/testing; shortcut is weaker)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset + few epochs to test the whole pipeline fast")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    conditions = [c for c in args.conditions.split(",") if c]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    args.output.mkdir(parents=True, exist_ok=True)

    spec = load_dataset(args.dataset, args.data_dir, args.download, args.imagenet_dir)
    if args.smoke:
        # shrink: keep a few per group per split
        keep: list[Example] = []
        seen: dict[tuple[int, int], int] = {}
        for e in spec.examples:
            k = (e.split, e.group)
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= 8:
                keep.append(e)
        spec.examples = keep
        args.base_epochs = min(args.base_epochs, 2)
        args.budget = min(args.budget, 2)
        args.faith_cap = min(args.faith_cap, 4)
        args.dfr_per_group = min(args.dfr_per_group, 4)

    from collections import Counter
    cc = Counter((e.split, e.group) for e in spec.examples)
    sn = {0: "train", 1: "val", 2: "test"}
    print(f"Dataset: {spec.name}  groups: {spec.group_names}")
    for s in (0, 1, 2):
        print(f"  {sn[s]:5s}: " + ", ".join(
            f"{spec.group_names[g]}={cc.get((s, g), 0)}" for g in spec.group_names))
    n_mask = sum(1 for e in spec.examples if e.mask_path)
    print(f"  masks: {n_mask}/{len(spec.examples)}")

    results_path = args.output / f"results_{spec.name}.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {"runs": []}
    done = {(r["condition"], r["seed"]) for r in results["runs"]}

    for seed in seeds:
        args._current_seed = seed
        base_state, diag = train_and_diagnose_base(spec, args, seed)
        for cond in conditions:
            if (cond, seed) in done:
                print(f"SKIP {cond} seed={seed} (already done)")
                continue
            print(f"\n>>> {spec.name}  condition={cond}  seed={seed}")
            res = run_condition(cond, base_state, spec, args)
            rec = asdict(res)
            rec["base_diagnosis"] = diag
            results["runs"].append(rec)
            save_results_atomic(results, results_path)
            print(f"    wga={res.worst_group_acc*100:.1f}%  "
                  f"ebpg={res.ebpg_mean:.3f}  ({res.wall_clock_s:.0f}s)")

    print(f"\nDone. {len(results['runs'])} run records in {results_path}")


if __name__ == "__main__":
    main()
