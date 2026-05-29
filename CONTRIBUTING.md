# Contributing to BNNR

Thank you for helping improve BNNR. This guide covers local setup, quality checks, and how to open issues and pull requests.

## Development setup

```bash
git clone https://github.com/bnnr-team/bnnr.git
cd bnnr

# Dashboard UI (required for `bnnr[dashboard]` and README screenshot script)
cd dashboard_web && npm ci && npm run build && cd ..

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
```

Optional: install Playwright browsers for dashboard capture scripts:

```bash
playwright install chromium
```

## Quality checks

Run before opening a PR:

```bash
pytest
ruff check src tests
mypy src
```

Pre-commit (optional):

```bash
pre-commit install
pre-commit run --all-files
```

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Keep changes focused; link related issues when applicable.
3. Add or update tests for behavior changes.
4. Update docs (`README.md`, `docs/`) when CLI or user-facing behavior changes.
5. Ensure `pytest` and `ruff check` pass locally.
6. Open a PR with a clear summary and test plan. PRs must pass CI and [Dependency Review](.github/workflows/dependency-review.yml).

## Good first issues

Maintainers can seed starter issues with:

```bash
gh label sync --file .github/labels.yml
gh issue create --title "..." --label "good first issue"
```

Suggested starter tasks:

1. Add a dataset preset example under `examples/configs/`
2. Improve `docs/troubleshooting.md` for a common install error
3. Add a unit test for an augmentation in `tests/`
4. Fix a typo or clarify a doc section in `docs/getting_started.md`
5. Export dashboard HTML snapshot test in `tests/`
6. Add type hints to a small module in `src/bnnr/`
7. Document a new CLI flag in `docs/cli.md`
8. Add `ruff format` consistency in a single package submodule
9. Improve error message when `--data-path` is missing for `imagefolder`
10. Add a notebook cell explaining ICD vs AICD in `examples/`

Look for issues labeled **good first issue** on GitHub.

## Code of conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Maintainer checklist

One-time repo setup (Discussions, label sync, starter issues): [.github/MAINTAINER_SETUP.md](.github/MAINTAINER_SETUP.md).

## Questions

- **Bug reports:** use the Bug report issue template.
- **Features:** use the Feature request template.
- **How-to questions:** GitHub Discussions (when enabled) or open a Question issue.
