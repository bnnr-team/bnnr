# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

BNNR (Bulletproof Neural Network Recipe) is a Python library for automatically improving PyTorch vision models using XAI. It has a React/Vite dashboard frontend at `dashboard_web/` that gets built into the Python package at `src/bnnr/dashboard/frontend/dist/`.

### Development environment

- **Python 3.12** with a virtualenv at `/workspace/.venv`
- **Node.js 22** for the dashboard frontend build
- Activate the venv: `source /workspace/.venv/bin/activate`
- No external services (databases, Docker, etc.) are required

### Key commands

| Task | Command |
|------|---------|
| Install Python deps | `pip install -e ".[dev,dashboard]"` |
| Build dashboard frontend | `cd dashboard_web && npm ci && npm run build` |
| Run tests | `pytest` |
| Lint | `ruff check src tests` |
| Type check | `mypy src` |
| Run demo | `python -m bnnr demo` |
| Run CLI training | `python -m bnnr train --dataset cifar10 --preset light --with-dashboard` |

### Gotchas and non-obvious notes

- The dashboard frontend **must be built before** the Python editable install, because `hatch` includes `src/bnnr/dashboard/frontend/dist/` in the wheel. If the dist folder is missing, the dashboard won't serve its UI.
- `python3.12-venv` system package is required (not installed by default on VM). The update script handles this.
- The `bnnr demo` command auto-downloads CIFAR-10 (~170MB) on first run. Tests use synthetic data and do not require network access after initial setup.
- dbus/GCM errors in terminal output during demo are Chrome background noise and are harmless.
- Tests run on CPU only (`device="cpu"`); no GPU is needed.
- pytest config is in `pyproject.toml` with `testpaths = ["tests"]` and coverage reporting enabled.
- Tests marked `yolo26` may require large model downloads; they are safe to skip locally.
