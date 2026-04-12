#!/usr/bin/env python3
"""Static validation for user-facing notebooks (no kernel, no training).

Checks each catalog notebook from docs/notebooks.md:
- Valid Jupyter nbformat v4 JSON
- Expected kernelspec (python3)
- Every code cell compiles as Python after stripping IPython line magics (% / !)

This catches broken syntax, bad JSON, and missing files before users open Colab/local.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep in sync with docs/notebooks.md — Notebook Catalog table.
USER_NOTEBOOKS: list[str] = [
    "examples/bnnr_augmentations_guide.ipynb",
    "examples/classification/bnnr_classification_demo.ipynb",
    "examples/multilabel/bnnr_multilabel_demo.ipynb",
    "examples/bnnr_custom_data.ipynb",
    "examples/detection/bnnr_detection_demo.ipynb",
]


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _strip_ipython_line_magics(source: str) -> str:
    """Remove lines that are IPython/Jupyter magics or shell escapes."""
    out: list[str] = []
    for line in source.splitlines():
        lead = line.lstrip()
        if lead.startswith("%") or lead.startswith("!"):
            continue
        out.append(line)
    return "\n".join(out)


def _validate_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]

    if data.get("nbformat") != 4:
        errors.append(f"{path}: expected nbformat 4, got {data.get('nbformat')!r}")
    nb_minor = data.get("nbformat_minor")
    if not isinstance(nb_minor, int):
        errors.append(f"{path}: expected integer nbformat_minor, got {nb_minor!r}")

    meta = data.get("metadata") or {}
    ks = meta.get("kernelspec") or {}
    if ks.get("name") != "python3":
        errors.append(f"{path}: kernelspec.name should be 'python3', got {ks.get('name')!r}")

    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append(f"{path}: no cells array")
        return errors

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"{path}: cell {idx} is not an object")
            continue
        if cell.get("cell_type") != "code":
            continue
        raw = _cell_source(cell)
        body = _strip_ipython_line_magics(raw).strip()
        if not body:
            continue
        cell_name = f"{path}: code cell {idx}"
        try:
            compile(body, cell_name, "exec")
        except SyntaxError as exc:
            errors.append(f"{cell_name}: SyntaxError: {exc.msg} (line {exc.lineno})")

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    all_errors: list[str] = []
    for rel in USER_NOTEBOOKS:
        path = root / rel
        if not path.is_file():
            all_errors.append(f"missing notebook file: {rel}")
            continue
        all_errors.extend(_validate_notebook(path))

    if all_errors:
        print("Notebook validation failed:", file=sys.stderr)
        for line in all_errors:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"OK: validated {len(USER_NOTEBOOKS)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
