# Maintainer setup

Canonical **Discussions** URL: `https://github.com/bnnr-team/bnnr/discussions`

## One-time GitHub settings

### 1. Enable Discussions

Repository **Settings → General → Features → Discussions** → enable.

Suggested categories:

- General (Q&A)
- Show and Tell
- Ideas
- Help

Pin a short welcome post (project scope, links to [getting started](docs/getting_started.md) and [troubleshooting](docs/troubleshooting.md)).

### 2. Sync labels

```bash
gh label sync --file .github/labels.yml
```

### 3. Good first issues

See [CONTRIBUTING.md](../CONTRIBUTING.md). Example:

```bash
gh issue create --title "Add type hints to a small module in src/bnnr/" \
  --label "good first issue,documentation" --body "Starter task for new contributors."
```

### 4. Releases

Use [CHANGELOG.md](../CHANGELOG.md) for GitHub Release notes:

```bash
gh release create v0.4.8 --title "v0.4.8" --notes-file CHANGELOG.md
```

(Trim to the section for that version before publishing, or copy the relevant block into the release body.)

### 5. Branch protection (`main`)

**Settings → Branches → Add rule** for `main`:

- Require a pull request before merging
- Require status checks to pass:
  - `quality-linux`
  - `test-ubuntu (3.10)` / `test-ubuntu (3.11)` / `test-ubuntu (3.12)` (or require the `test-ubuntu` check group if available)
  - `test` (matrix jobs on Windows/macOS)
  - `dependency-review`
  - `notebooks-smoke`
- Do not require `publish-pypi` (runs only on version tags)

### 6. Dependabot

- [`.github/dependabot.yml`](dependabot.yml) covers pip, GitHub Actions, and `dashboard_web` npm.
- Enable **Dependabot security updates** under **Settings → Code security and analysis**.

### 7. Code scanning autofix

To avoid dozens of one-alert PRs (CodeQL / Copilot autofix):

**Settings → Code security and analysis → Code scanning** — disable automatic PR creation per finding, or use batch fixes on a maintainer branch.

Consolidated quality fixes belong in normal PRs with full CI, not in parallel autofix branches.

## Repo automation summary

| Workflow | Purpose |
|----------|---------|
| [`ci.yml`](workflows/ci.yml) | ruff, mypy, pytest matrix, build wheel, PyPI on `v*` tags |
| [`dependency-review.yml`](workflows/dependency-review.yml) | Block PRs that introduce known-vulnerable dependencies |
| CodeQL (default setup) | Security analysis (weekly) |
| Dependabot | Weekly dependency PRs |
