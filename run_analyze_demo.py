#!/usr/bin/env python3
"""Train quick models on MNIST, CIFAR-10, STL-10 and run bnnr analyze on each."""
import os
import subprocess
import sys
from pathlib import Path

def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check)

def get_best_checkpoint(out_dir: str) -> Path:
    from pathlib import Path
    import json
    runs = sorted(Path(out_dir).glob("reports/run_*"), key=lambda p: p.name, reverse=True)
    if not runs:
        raise SystemExit(f"No run dir in {out_dir}")
    report = runs[0] / "report.json"
    data = json.loads(report.read_text())
    best = max(data["checkpoints"], key=lambda c: c["metrics"].get("accuracy", 0))
    p = Path(best["checkpoint_path"])
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()

def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    
    config = "config_quick.yaml"
    data_dir = "data"
    max_train = "--max-train-samples"
    max_val = "--max-val-samples"

    print("=== 1. MNIST ===")
    run([sys.executable, "-m", "bnnr", "train", "-c", config, "--dataset", "mnist",
        "--data-dir", data_dir, "-o", "out_mnist", max_train, "3000", max_val, "500",
        "-e", "2", "--without-dashboard", "--no-xai"])
    best_mnist = get_best_checkpoint("out_mnist")

    print("\n=== 2. CIFAR-10 ===")
    run([sys.executable, "-m", "bnnr", "train", "-c", config, "--dataset", "cifar10",
        "--data-dir", data_dir, "-o", "out_cifar10", max_train, "3000", max_val, "500",
        "-e", "2", "--without-dashboard", "--no-xai"])
    best_cifar = get_best_checkpoint("out_cifar10")

    print("\n=== 3. STL-10 ===")
    run([sys.executable, "-m", "bnnr", "train", "-c", "config_quick_stl10.yaml",
        "--dataset", "stl10", "--data-dir", data_dir, "-o", "out_stl10",
        max_train, "2000", max_val, "500", "-e", "3", "--without-dashboard", "--no-xai"])
    best_stl = get_best_checkpoint("out_stl10")

    print("\n=== 4. ANALYZE: MNIST ===")
    run([sys.executable, "-m", "bnnr", "analyze", "-m", str(best_mnist), "-d", "mnist", "-o", "out_analyze_mnist"])

    print("\n=== 5. ANALYZE: CIFAR-10 ===")
    run([sys.executable, "-m", "bnnr", "analyze", "-m", str(best_cifar), "-d", "cifar10", "-o", "out_analyze_cifar10"])

    print("\n=== 6. ANALYZE: STL-10 ===")
    run([sys.executable, "-m", "bnnr", "analyze", "-m", str(best_stl), "-d", "stl10", "-o", "out_analyze_stl10"])

    print("\n" + "=" * 46)
    print("  RAPORTY GOTOWE")
    print("=" * 46)
    print("\n  MNIST:     out_analyze_mnist/report.html")
    print("  CIFAR-10:  out_analyze_cifar10/report.html")
    print("  STL-10:    out_analyze_stl10/report.html\n")

if __name__ == "__main__":
    main()
