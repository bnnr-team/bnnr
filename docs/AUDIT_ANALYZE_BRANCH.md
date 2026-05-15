# Audyt gałęzi `feature/analyze-report-wow` — `analyze` / STL-10 / syntetyk

Data przebiegu: 2026-04-21. Gałąź: `feature/analyze-report-wow`. Środowisko: Linux, Python 3.12, PyTorch **CPU** (`torch==2.11.0+cpu`) po `uv sync --extra dev --extra dashboard` oraz ręcznej podmianie torch na CPU (CUDA wheel w tym środowisku wymagał `libcudnn.so.9`).

## Wyniki zautomatyzowane

| Artefakt | Wynik |
|----------|--------|
| `pytest tests/test_analyze.py` (20 testów) | **OK** (~42 s, `--no-cov`) |
| `scripts/run_classification_multilabel_smoke.sh` (MNIST + multilabel) | **OK** (~83 s); artefakty: `~/bnnr_audit_runs/smoke_opt_1776776061/` |
| `scripts/audit_synthetic_multiclass_analyze.py` | **OK** — `run_evaluation` vs `analyze_model`: identyczna `accuracy` (delta 0) |

## STL-10 (klasyfikacja)

- Pełny smoke ze skryptu z `BNNR_SMOKE_CLASSIFICATION=stl10` **nie został dokończony** w tej sesji (pierwsze pobranie ~2.6 GB + czas treningu); uruchomienie lokalne:  
  `BNNR_SMOKE_CLASSIFICATION=stl10 BNNR_OPEN_REPORTS=0 ./scripts/run_classification_multilabel_smoke.sh`
- Ścieżka MNIST w smoke i `bnnr analyze` **pokrywa ten sam kod** co STL-10 (pipeline + checkpoint + CLI analyze); regresje specyficzne dla STL dotyczą głównie architektury i rozmiaru obrazu w [`pipelines.py`](src/bnnr/pipelines.py) oraz config [`examples/configs/classification/stl10_showcase.yaml`](examples/configs/classification/stl10_showcase.yaml).

## CLI vs Python API (parity)

- Test `test_api_cli_parity` weryfikuje spójność kluczy JSON (`metrics`, `per_class_accuracy`, `confusion`, `executive_summary`, `findings`) oraz sensowny zakres `accuracy`.
- Testy CLI (`test_analyze_cli`, `test_analyze_cli_with_xai`, `test_api_cli_parity`) wymagają **lokalnej kopii MNIST** (niestabilne pobieranie z domyślnych mirrorów). Dodano `_seed_mnist_data_dir` w [`tests/test_analyze.py`](tests/test_analyze.py): kopiuje z `BNNR_TEST_MNIST_CACHE/MNIST` lub z `~/.cache/bnnr_test_mnist/MNIST`.

## Uwagi środowiskowe

1. **`uv sync`** wymaga istnienia [`src/bnnr/dashboard/frontend/dist/index.html`](src/bnnr/dashboard/frontend/dist/index.html) (Hatch `force-include`); w dev zwykle buduje się `dashboard_web` (`npm run build`) albo tworzy się minimalny stub tylko pod instalację editable.
2. Po `uv sync --extra dashboard` domyślny torch z lockfile może być **CUDA** — na hostach bez cuDNN użyć CPU wheels z `https://download.pytorch.org/whl/cpu` (jak w tej sesji).
3. `opencv-python`: po eksperymentach z odinstalowaniem pakietów warto zweryfikować `import cv2; cv2.imwrite` (data quality w treningu).

## Rekomendacja merge

- Logika `analyze` + testy jednostkowe: **gotowe do review** przy zielonym CI z cache MNIST (zmienna lub job cache).
- Przed merge do `main`: potwierdzić smoke **STL-10** na maszynie z miejscem na dataset i ewentualnie GPU.
