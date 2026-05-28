# Releasing BNNR on PyPI

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## Before you push a release tag

Run the same checks as GitHub Actions (from the repo root, with dev deps installed):

```bash
uv run ruff check src/ tests/
uv run mypy src/bnnr/
uv run pytest
```

Fix any failures **before** tagging. Pushing `v*` triggers the **publish-pypi** job only after the full CI matrix and `build-package` succeed.

## Version bump

1. **Version** — bump `__version__` in [`src/bnnr/version.py`](src/bnnr/version.py) and `version` in `pyproject.toml` (must match). `src/bnnr/__init__.py` re-exports from `bnnr.version`; analyze `REPORT_SCHEMA_VERSION` follows the same value. Dashboard `FastAPI` reads `bnnr.version.__version__`. Update `CHANGELOG.md`, `README.md` / `README.pypi.md` if they mention the version, `examples/detection/bnnr_detection_demo.ipynb` if the install pin changes, then `uv lock`.
2. **Analyze sample HTML** — after an analyze or version change affecting reports, regenerate the public preview:

   ```bash
   .venv-uv/bin/python scripts/generate_analyze_sample_report.py
   ```

   Confirm `docs/assets/analyze-report-sample.html` shows `v<version>` (not a stale schema number) and embedded base64 images.

3. **Build** — dashboard frontend must be built (see `README.md` / CI). Then:

   ```bash
   pip install hatch
   hatch build
   ```

4. **Upload** — CI uses [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) on tag `v*`. For a manual upload:

   ```bash
   pip install twine
   twine upload dist/bnnr-<version>*
   ```

5. **Verify** — in a clean venv: `pip install "bnnr>=<version>"`; run `pytest` or the detection notebook on Colab.

6. **Tag** — after `main` contains the release commit, create and push the annotated tag (e.g. `v0.4.8`) so CI publishes the wheel.
