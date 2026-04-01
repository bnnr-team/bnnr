# Development

## What you will find here
The repository's local developer workflow: setup, tests, linting, type checks, and dashboard frontend build.

## When to use this page
Use this when contributing changes or validating local modifications.

## Install developer dependencies

```bash
python3 -m pip install -e ".[dev,dashboard]"
```

## Run tests

```bash
pytest
```

`pytest` settings are defined in `pyproject.toml` (`testpaths = ["tests"]`).

## Run lint and types

```bash
ruff check src tests
mypy src
```

## Dashboard frontend (React/Vite)

```bash
cd dashboard_web
npm install
npm run dev
npm run build
```

Notes:
- Backend can auto-build frontend when using `start_dashboard(...)` if `dashboard_web/` exists and `npm` is available.
- Build output goes to `dashboard_web/dist/` (gitignored). To ship static UI inside the wheel, copy that tree into `src/bnnr/dashboard/frontend/dist/` before `hatch build`.
