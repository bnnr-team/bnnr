"""Test bnnr analyze przez API — odpowiednik:
  python3 -m bnnr analyze --model checkpoints/iter_0_baseline.pt --data mnist --output ./out_analyze --task classification
"""
from pathlib import Path

import torch

from bnnr.analyze import analyze_model
from bnnr.core import BNNRConfig
from bnnr.pipelines import build_pipeline

def main():
    model_path = Path("checkpoints/iter_0_baseline.pt")
    output_dir = Path("./out_analyze")
    task = "classification"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cfg = BNNRConfig(
        task=task,
        device="auto",
        metrics=["accuracy", "f1_macro", "loss"],
    )

    adapter, _train_loader, val_loader, _augs = build_pipeline(
        dataset_name="mnist",
        config=cfg,
        data_dir=Path("data"),
        batch_size=64,
        custom_data_path=None,
    )

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt["model"]
        if isinstance(ckpt, dict) and "model" in ckpt
        else ckpt
    )
    if isinstance(state, dict) and "model" in state and "optimizer" in state:
        state = state["model"]
    model_obj = getattr(adapter, "model", None)
    if model_obj is not None:
        model_obj.load_state_dict(state, strict=True)
    else:
        adapter.load_state_dict(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze_model(
        adapter,
        val_loader,
        config=cfg,
        task=task,
        output_dir=output_dir,
        run_data_quality=True,
        max_worst=20,
        xai_enabled=True,
    )
    report.save(output_dir)
    report.to_html(output_dir / "report.html")

    print("Analysis saved to", output_dir)
    print("  ", output_dir / "analysis_report.json")
    print("  ", output_dir / "report.html")
    print("Metrics:", report.metrics)

if __name__ == "__main__":
    main()