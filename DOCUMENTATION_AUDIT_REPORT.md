# BNNR Documentation Audit Report

**Audit date:** 2026-05-31  
**Package version (ground truth):** `0.4.11` (`src/bnnr/version.py`)  
**Method:** Code-first verification — `src/bnnr/`, `tests/`, `pyproject.toml`.

**Out of scope:** `docs/roadmap.md`, `plans/`, README section “What makes BNNR different”, `CHANGELOG.md` (historical).

---

## 1. Executive Summary

| Metric | Count |
|--------|------:|
| Documentation files reviewed (`.md`) | 36 |
| Jupyter notebooks reviewed | 6 |
| Example Python scripts reviewed | 11 |
| Inconsistencies found | 12 |
| Documentation files corrected | 10 |
| Supporting scripts updated | 1 |

No changes to `src/bnnr/` implementation code.

---

## 2. Findings (summary)

| ID | File | Issue | Fix |
|----|------|-------|-----|
| F-01 | `docs/cli.md` | Missing `stl10` in train datasets | Added |
| F-02 | `docs/configuration.md` | `detection_xai_method` documented as `saliency` | Corrected to `activation` \| `occlusion` |
| F-03 | `docs/augmentations.md` | Incomplete preset list | Added `demo`, `none`, `screening` |
| F-04 | `docs/api_reference.md` | Stable vs deprecated API conflated | Restructured with `__all__` table |
| F-05 | `docs/configuration.md`, `docs/README.md` | `BNNRConfig` attributed to `core.py` | Point to `config_model.py` |
| F-06 | `docs/configuration.md` | Missing `duplicate_hamming_threshold` | Documented (default `10`) |
| F-07 | `docs/artifacts.md` | Nonexistent `worst_predictions` JSON key | Replaced with actual keys |
| F-08 | `docs/analyze.md` | Outdated limitations (v0.2.x) | Updated to current behavior |
| F-09 | `docs/detection.md` | Incomplete detection config table | Full field list from `config_model.py` |
| F-10 | `docs/artifacts.md` | Missing `events.jsonl` schema version | Documented `2.1` |
| F-11 | `scripts/validate_user_notebooks.py` | 5 vs 6 catalog notebooks | Added cooking notebook |
| F-12 | `docs/getting_started.md` | Incomplete dataset list | Added `stl10`, `fashion_mnist` |

---

## 3. High-Risk Inconsistencies

1. Missing `stl10` in CLI train docs — users could assume unsupported dataset.
2. Invalid `detection_xai_method: saliency` — config validation would fail or misconfigure runs.
3. API reference implied all top-level imports are stable — deprecation warnings not documented.
4. `worst_predictions` in artifacts doc — JSON parsers would fail on missing key.

---

## 4. API Mismatches (resolved)

See corrected `docs/cli.md`, `docs/configuration.md`, `docs/api_reference.md`. Key alignments:

- **CLI datasets:** `mnist`, `fashion_mnist`, `cifar10`, `stl10`, `imagefolder`, `coco_mini`, `yolo`
- **Analyze tasks:** `classification`, `multilabel` only (not detection)
- **Stable API:** 23 symbols in `bnnr.__all__`; 57 backward-compatible deprecated imports
- **`EVENT_SCHEMA_VERSION`:** `"2.1"`

---

## 5. Example Fixes

| Asset | Result |
|-------|--------|
| 11× `examples/**/*.py` | `py_compile` — all OK |
| 6 notebooks | `validate_user_notebooks.py` — all OK |

---

## 6. Files Modified in This PR

`docs/cli.md`, `docs/configuration.md`, `docs/augmentations.md`, `docs/api_reference.md`, `docs/README.md`, `docs/artifacts.md`, `docs/analyze.md`, `docs/detection.md`, `docs/getting_started.md`, `scripts/validate_user_notebooks.py`, `DOCUMENTATION_AUDIT_REPORT.md`
