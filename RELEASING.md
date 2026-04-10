# Releasing BNNR on PyPI

1. **Version** — bump `version` in `pyproject.toml` and `__version__` in `src/bnnr/__init__.py` (and dashboard `FastAPI(..., version=...)` if you keep it in sync). Update `CHANGELOG.md` and `README.md` if they mention the version.

2. **Build** — dashboard frontend must be built (see `README.md` / CI). Then:

   ```bash
   pip install hatch
   hatch build
   ```

3. **Upload** — use [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) or API token:

   ```bash
   pip install twine
   twine upload dist/bnnr-0.2.3*
   ```

4. **Verify** — `pip install "bnnr>=0.2.3"` in a clean venv; run `pytest` or open `examples/detection/bnnr_detection_demo.ipynb` on Colab.

Tag the release in Git (`v0.2.3`) after the wheel is on PyPI.
