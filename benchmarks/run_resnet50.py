#!/usr/bin/env python3
"""ResNet50 / CIFAR-100 augmentation benchmark.

Compares four training conditions on a *real* architecture (ResNet50) and a
harder dataset (CIFAR-100, 100 classes) than the demo-CNN / CIFAR-10 table in
``benchmarks/run.py``:

    no_bnnr         RandomCrop + RandomHorizontalFlip (standard baseline)
    randaugment     + torchvision RandAugment
    trivialaugment  + torchvision TrivialAugmentWide
    bnnr_branch_search   full BNNR loop (ICD + AICD + ChurchNoise, branch search)

Protocol honesty (mirrors ``docs/benchmarks.md``):
  * Same backbone, optimizer, scheduler, epochs, and seeds across conditions.
  * The augmentation strategy is the *only* thing that varies per condition.
  * BNNR branch search spends more compute (baseline phase + branch search);
    this is disclosed in the results JSON and the README table.

The ResNet50 wrapper normalizes inputs *inside* the model (ImageNet mean/std as
registered buffers) so every condition can feed plain ``ToTensor()`` tensors in
``[0, 1]``. This keeps pretrained weights happy while remaining compatible with
BNNR's ICD/AICD augmentations, which operate on uint8-range tensors.

This script builds the runnable infrastructure. Numbers land in the README only
after a real GPU run (see ``benchmarks/reproduce_resnet50.sh``).

Examples
--------
    # Quick smoke test (CPU-friendly, no pretrained download, tiny subset)
    python benchmarks/run_resnet50.py --smoke

    # Full benchmark, 5 seeds, GPU
    python benchmarks/run_resnet50.py --seeds 42,43,44,45,46 --device cuda

    # Single condition / single seed
    python benchmarks/run_resnet50.py --conditions no_bnnr --seeds 42 --device cuda

    python benchmarks/summarize.py --results benchmarks/results_resnet50.json --markdown
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARKS_DIR.parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lib import (  # noqa: E402
    ConditionSpec,
    _extract_baseline_from_report,
    _make_run_dir,
    _result_entry,
    _run_config,
    _run_plain_epochs,
    build_bnnr_candidate_augmentations,
    export_attention_maps,
    git_head,
    load_results,
    save_results,
    torch_info,
)

DEFAULT_RESULTS = BENCHMARKS_DIR / "results_resnet50.json"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "runs_resnet50"

# ImageNet normalization, applied inside the model (see _NormalizedResNet).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# OptiCAM overlay indices into the CIFAR-100 test split (10k images).
_XAI_VAL_INDICES = [0, 127, 255, 512]


CONDITIONS: dict[str, ConditionSpec] = {
    "no_bnnr": ConditionSpec(
        id="no_bnnr",
        label="Without BNNR (crop + flip only)",
        strategy="plain_training",
        description=(
            "Standard CIFAR-100 training: RandomCrop + RandomHorizontalFlip. "
            "No BNNR batch augmentations and no branch search."
        ),
        augmentation_names=(),
        max_iterations=0,
    ),
    "randaugment": ConditionSpec(
        id="randaugment",
        label="RandAugment (torchvision)",
        strategy="randaugment",
        description=(
            "External baseline: RandomCrop + RandomHorizontalFlip + "
            "torchvision RandAugment. Policy-based random augmentation, "
            "no saliency guidance."
        ),
        augmentation_names=("RandAugment",),
        max_iterations=0,
    ),
    "trivialaugment": ConditionSpec(
        id="trivialaugment",
        label="TrivialAugmentWide (torchvision)",
        strategy="trivialaugment",
        description=(
            "External baseline: RandomCrop + RandomHorizontalFlip + "
            "torchvision TrivialAugmentWide. Parameter-free SOTA-ish random "
            "augmentation, no saliency guidance."
        ),
        augmentation_names=("TrivialAugmentWide",),
        max_iterations=0,
    ),
    "bnnr_branch_search": ConditionSpec(
        id="bnnr_branch_search",
        label="BNNR branch search (ICD + AICD + ChurchNoise)",
        strategy="bnnr_branch_search",
        description=(
            "Full BNNR loop: baseline phase, then branch search over "
            "ICD (mask high-saliency regions), AICD (mask background), and "
            "ChurchNoise. Keeps branches that improve validation accuracy."
        ),
        augmentation_names=("ICD", "AICD", "augmentation_1"),
        max_iterations=3,
    ),
}

_ARCH_TARGET_HINT = {
    "resnet50": "backbone.layer4[-1]",
    "resnet18": "backbone.layer4[-1]",
}


# ---------------------------------------------------------------------------
# Model: torchvision ResNet with in-model ImageNet normalization
# ---------------------------------------------------------------------------


def _build_resnet(arch: str, num_classes: int, pretrained: bool) -> Any:
    """torchvision ResNet adapted to ``num_classes`` with normalization baked in.

    Inputs are expected in ``[0, 1]`` (plain ``ToTensor()``); the model
    normalizes with ImageNet statistics internally so pretrained weights work
    and BNNR's uint8-range augmentations stay compatible upstream.
    """
    import torch
    from torch import nn
    from torchvision import models

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported arch {arch!r} (use resnet50 or resnet18)")

    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

    class _NormalizedResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, x: Any) -> Any:
            x = (x - self.mean) / self.std
            return self.backbone(x)

    return _NormalizedResNet()


def _target_layers(model: Any) -> list[Any]:
    return [model.backbone.layer4[-1]]


# ---------------------------------------------------------------------------
# Data: CIFAR-100 loaders (one builder per augmentation policy)
# ---------------------------------------------------------------------------


def _cifar100_loaders(
    *,
    img_size: int,
    batch_size: int,
    seed: int,
    policy: str,
    max_train: int | None,
    max_val: int | None,
) -> tuple[Any, Any]:
    """Return ``(train_loader, val_loader)`` for the requested augmentation policy.

    ``policy`` is one of ``base`` | ``randaugment`` | ``trivialaugment``. The val
    transform is identical across policies (resize + ToTensor).
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from bnnr.pipelines import _IndexedDataset, _maybe_subset

    base_train = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=img_size // 8),
        transforms.RandomHorizontalFlip(),
    ]
    if policy == "randaugment":
        base_train.append(transforms.RandAugment())
    elif policy == "trivialaugment":
        base_train.append(transforms.TrivialAugmentWide())
    elif policy != "base":
        raise ValueError(f"Unknown policy {policy!r}")
    base_train.append(transforms.ToTensor())

    train_tf = transforms.Compose(base_train)
    val_tf = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )

    data_root = str(REPO_ROOT / "data")
    train_ds = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR100(data_root, train=False, download=True, transform=val_tf)
    train_ds = _maybe_subset(train_ds, max_train)
    val_ds = _maybe_subset(val_ds, max_val)

    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _IndexedDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=generator,
    )
    val_loader = DataLoader(
        _IndexedDataset(val_ds), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def _build_adapter(
    *,
    arch: str,
    num_classes: int,
    pretrained: bool,
    lr: float,
    device: str,
    epochs: int,
) -> Any:
    import torch
    from torch import nn

    from bnnr.adapter import SimpleTorchAdapter

    model = _build_resnet(arch, num_classes, pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return SimpleTorchAdapter(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        target_layers=_target_layers(model),
        device=device,
        scheduler=scheduler,
    )


# ---------------------------------------------------------------------------
# Config + per-condition runners
# ---------------------------------------------------------------------------


def _base_config(*, epochs: int, device: str) -> Any:
    from bnnr.config_model import BNNRConfig

    return BNNRConfig(
        task="classification",
        device=device,
        m_epochs=epochs,
        selection_metric="accuracy",
        metrics=["accuracy", "f1_macro", "loss"],
        xai_method="opticam",
    )


def _stamp(entry: dict[str, Any], *, arch: str, img_size: int, pretrained: bool) -> dict[str, Any]:
    """Override the cifar10 defaults baked into lib._result_entry."""
    entry["dataset"] = "cifar100"
    entry["model"] = arch
    entry["img_size"] = img_size
    entry["pretrained"] = pretrained
    return entry


def _run_plain_condition(
    *,
    condition: ConditionSpec,
    policy: str,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = _make_run_dir(output_root, condition.id, seed)
    base_cfg = _base_config(epochs=args.epochs, device=args.device)
    cfg = _run_config(base_cfg, seed=seed, device=args.device, run_dir=run_dir, xai=False)

    adapter = _build_adapter(
        arch=args.arch,
        num_classes=100,
        pretrained=args.pretrained,
        lr=args.lr,
        device=cfg.device,
        epochs=args.epochs,
    )
    train_loader, val_loader = _cifar100_loaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=seed,
        policy=policy,
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
    )
    bench = {"xai_val_indices": _XAI_VAL_INDICES, "xai_method": "opticam"}
    extra = {"augmentation_policy": policy} if policy != "base" else None
    entry = _run_plain_epochs(
        condition=condition,
        cfg=cfg,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        augmentations=[],
        run_dir=run_dir,
        bench=bench,
        extra_meta=extra,
        export_xai=not args.no_xai,
    )
    return _stamp(entry, arch=args.arch, img_size=args.img_size, pretrained=args.pretrained)


def _run_bnnr_condition(
    *,
    condition: ConditionSpec,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    from bnnr.core import BNNRTrainer

    run_dir = _make_run_dir(output_root, condition.id, seed)
    base_cfg = _base_config(epochs=args.epochs, device=args.device)
    cfg = _run_config(
        base_cfg,
        seed=seed,
        device=args.device,
        run_dir=run_dir,
        xai=True,
        max_iterations=condition.max_iterations,
    )

    adapter = _build_adapter(
        arch=args.arch,
        num_classes=100,
        pretrained=args.pretrained,
        lr=args.lr,
        device=cfg.device,
        epochs=args.epochs,
    )
    train_loader, val_loader = _cifar100_loaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=seed,
        policy="base",
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
    )
    augmentations = build_bnnr_candidate_augmentations(
        adapter.get_model(), adapter.get_target_layers(), seed
    )

    print(
        f"\n{'='*60}\n"
        f"  BNNR BRANCH SEARCH ({args.arch} / cifar100)\n"
        f"  candidates={[a.name for a in augmentations]}\n"
        f"  max_iterations={condition.max_iterations}  m_epochs={cfg.m_epochs}\n"
        f"  seed={seed}  device={cfg.device}\n"
        f"{'='*60}",
        flush=True,
    )

    t0 = time.perf_counter()
    trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
    result = trainer.run()
    elapsed_s = time.perf_counter() - t0

    sel = cfg.selection_metric
    best = float(result.best_metrics.get(sel, 0.0))
    baseline_val = _extract_baseline_from_report(result.report_json_path, sel)
    gain = round((best - baseline_val) * 100, 2) if baseline_val is not None else None

    selected = [p for p in (result.best_path or "").split("+") if p and p != "baseline"]
    if not selected and result.best_path and result.best_path != "baseline":
        selected = [result.best_path]

    if args.no_xai:
        xai_meta = {"xai_dir": None, "overlay_paths": [], "aggregate_stats": {}}
    else:
        xai_meta = export_attention_maps(
            adapter,
            val_loader,
            sample_indices=_XAI_VAL_INDICES,
            output_dir=run_dir / "xai",
            xai_method="opticam",
        )

    entry = _result_entry(
        condition=condition,
        cfg=cfg,
        best_val=result.best_metrics,
        best_epoch=None,
        elapsed_s=elapsed_s,
        run_dir=run_dir,
        report_path=result.report_json_path,
        best_path=result.best_path,
        xai_meta=xai_meta,
        baseline_val=baseline_val,
        gain_pp=gain,
        extra={
            "selected_augmentations": selected,
            "augmentation_names": [a.name for a in augmentations],
        },
    )
    return _stamp(entry, arch=args.arch, img_size=args.img_size, pretrained=args.pretrained)


_POLICY_BY_CONDITION = {
    "no_bnnr": "base",
    "randaugment": "randaugment",
    "trivialaugment": "trivialaugment",
}


def run_condition(
    *,
    condition_id: str,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    spec = CONDITIONS[condition_id]
    if spec.strategy == "bnnr_branch_search":
        return _run_bnnr_condition(
            condition=spec, seed=seed, args=args, output_root=output_root
        )
    return _run_plain_condition(
        condition=spec,
        policy=_POLICY_BY_CONDITION[condition_id],
        seed=seed,
        args=args,
        output_root=output_root,
    )


# ---------------------------------------------------------------------------
# Results document + CLI
# ---------------------------------------------------------------------------


def _benchmark_document(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "benchmark_id": "cifar100_resnet50_augmentation_comparison",
        "model": f"{args.arch} (torchvision, pretrained={args.pretrained})",
        "dataset": "cifar100 (torchvision, full train/test split)",
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "optimizer": f"SGD(lr={args.lr}, momentum=0.9, wd=5e-4)",
        "scheduler": "CosineAnnealingLR",
        "epochs_per_phase": args.epochs,
        "primary_metric": "validation accuracy (best epoch)",
        "normalization": "ImageNet mean/std applied inside the model",
        "attention_method": "OptiCAM overlays on fixed CIFAR-100 test indices",
        "target_layer": _ARCH_TARGET_HINT.get(args.arch, "backbone.layer4[-1]"),
        "conditions": {
            cid: {
                "label": spec.label,
                "strategy": spec.strategy,
                "max_iterations": spec.max_iterations,
                "augmentations": list(spec.augmentation_names),
                "description": spec.description,
            }
            for cid, spec in CONDITIONS.items()
        },
        "protocol_caveats": [
            "BNNR branch search uses more compute than the fixed-epoch baselines "
            "(baseline phase + branch search). This is by design and disclosed here.",
            "Same backbone, optimizer, scheduler, epochs and seeds across conditions; "
            "only the augmentation strategy varies.",
            "Not an ImageNet-SOTA claim. CIFAR-100 transfer benchmark for "
            "augmentation-strategy comparison.",
        ],
    }


def _estimate(args: argparse.Namespace, n_seeds: int, conds: list[str]) -> str:
    per_cond_min = {
        "no_bnnr": args.epochs * 1.0,
        "randaugment": args.epochs * 1.1,
        "trivialaugment": args.epochs * 1.1,
        "bnnr_branch_search": args.epochs * (args.max_iterations + 1) * 1.2,
    }
    total = sum(per_cond_min.get(c, args.epochs) for c in conds) * n_seeds
    if args.device != "cpu":
        total *= 0.2
    return f"~{total/60:.1f}h wall-clock estimate ({len(conds)} conditions x {n_seeds} seeds, {args.device})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResNet50 / CIFAR-100 augmentation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated seeds")
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help=f"Comma-separated conditions ({', '.join(CONDITIONS)})",
    )
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per training phase")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224, help="Resize target (224 uses pretrained features)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=3, help="BNNR branch-search iterations")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained backbone (--no-pretrained to train from scratch)",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--no-xai", action="store_true", help="Skip OptiCAM overlay export (faster; no cv2 dependency)")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run: 1 epoch, 256/128 subset, img-size 64, no pretrained download",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.img_size = 64
        args.pretrained = False
        args.max_train_samples = 256
        args.max_val_samples = 128
        args.max_iterations = 1
        if args.seeds == "42,43,44,45,46":
            args.seeds = "42"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in CONDITIONS:
            parser.error(f"Unknown condition {c!r}. Choose from: {', '.join(CONDITIONS)}")

    print("ResNet50 / CIFAR-100 augmentation benchmark")
    print(f"  arch={args.arch} pretrained={args.pretrained} img_size={args.img_size}")
    print(f"  seeds={seeds}  conditions={conds}  epochs={args.epochs}  device={args.device}")
    print(f"  {_estimate(args, len(seeds), conds)}")
    print(f"  results -> {args.results}")
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    data = load_results(args.results)
    doc = _benchmark_document(args)
    data.update(doc)
    data.setdefault("runs", [])
    data["hardware"] = torch_info()
    data["git_head"] = git_head()
    data["generated_at"] = datetime.now(timezone.utc).isoformat()

    for seed in seeds:
        for cid in conds:
            print(f"\n>>> condition={cid} seed={seed}")
            try:
                entry = run_condition(
                    condition_id=cid, seed=seed, args=args, output_root=args.output_root
                )
            except Exception as exc:  # noqa: BLE001 — record and continue the matrix
                print(f"    FAILED ({cid}, seed={seed}): {exc}", file=sys.stderr)
                entry = {
                    "condition": cid,
                    "seed": seed,
                    "dataset": "cifar100",
                    "model": args.arch,
                    "error": str(exc),
                }
            data["runs"].append(entry)
            save_results(args.results, data)  # checkpoint after every run
            val = entry.get("val_metric")
            if val is not None:
                print(f"    {cid} seed={seed}: val_accuracy={val:.4f}")

    print(f"\nDone. {len(data['runs'])} run records in {args.results}")
    print(f"Summarize: python benchmarks/summarize.py --results {args.results} --markdown")


if __name__ == "__main__":
    main()
