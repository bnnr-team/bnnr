# Releasing BNNR on PyPI

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/bnnr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/bnnr)

## Automated releases (release-please)

Releases are automated. You do **not** bump versions or push tags by hand.

1. **Land work normally.** Merge PRs into `main` using [Conventional Commit](https://www.conventionalcommits.org/) titles: `fix:` (patch), `feat:` (minor), `feat!:` or a `BREAKING CHANGE:` footer (major). Other prefixes (`docs:`, `chore:`, `test:`, `ci:`) do not trigger a release on their own.
2. **release-please opens a Release PR.** [`.github/workflows/release-please.yml`](.github/workflows/release-please.yml) maintains a `chore(main): release X.Y.Z` PR that computes the next version, updates `CHANGELOG.md`, and bumps the version everywhere it is annotated with `x-release-please-version` (`src/bnnr/version.py`, `CITATION.cff`, `README.md`, `README.pypi.md`, the analyze sample report, and the roadmap/citation docs). `pyproject.toml` is bumped by the `python` release type.
3. **Merge the Release PR when you want to ship.** Merging it makes release-please create the GitHub release and the `vX.Y.Z` tag, and the same workflow run then builds the wheel and publishes to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no token). If `VERCEL_DEPLOY_HOOK_URL` is set, it also pings the website to redeploy with the new version.

To add a version-bearing file to the automation, add an `x-release-please-version` comment on the line and list the file under `extra-files` in [`release-please-config.json`](release-please-config.json).

### One-time setup (already done unless noted)

- **PyPI Trusted Publisher** must list the workflow file **`release-please.yml`** (publishing moved here from `ci.yml`). Update it under the PyPI project → Publishing settings, otherwise the publish step fails with an OIDC error.
- **`VERCEL_DEPLOY_HOOK_URL`** repo secret (optional): a Vercel Deploy Hook URL for `bnnr-website`; when present the release redeploys the site.

## Before merging (sanity)

CI runs on the Release PR's merge to `main` and gates nothing you need to run by hand, but locally you can still verify:

```bash
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m mypy src/bnnr/
.venv/bin/python -m pytest
```

## Manual fallback

If you ever need to publish without release-please, build (`pip install build && python -m build` with the dashboard frontend built) and `twine upload dist/bnnr-<version>*`, then verify in a clean venv with `pip install "bnnr>=<version>"`.
