# Releasing BNNR on PyPI

## Before you push a release tag

Run the same checks as GitHub Actions (from the repo root, with dev deps installed):

```bash
uv run ruff check src/ tests/
uv run mypy src/bnnr/
uv run pytest
```

Fix any failures **before** tagging. Pushing `v*` triggers the **publish-pypi** job only after the full CI matrix and `build-package` succeed.

## Version bump

1. **Version** — bump `version` in `pyproject.toml` and `__version__` in `src/bnnr/__init__.py` (and dashboard `FastAPI(..., version=...)` in `src/bnnr/dashboard/backend.py` if you keep it in sync). Update `CHANGELOG.md`, `README.md` if they mention the version, `examples/detection/bnnr_detection_demo.ipynb` if the install pin changes, then `uv lock`.

2. **Build** — dashboard frontend must be built (see `README.md` / CI). Then:

   ```bash
   pip install hatch
   hatch build
   ```

3. **Upload** — CI uses [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) on tag `v*`. For a manual upload:

   ```bash
   pip install twine
   twine upload dist/bnnr-<version>*
   ```

4. **Verify** — in a clean venv: `pip install "bnnr>=<version>"`; run `pytest` or the detection notebook on Colab.

5. **Tag** — after `main` contains the release commit, create and push the annotated tag (e.g. `v0.2.5`) so CI publishes the wheel.
