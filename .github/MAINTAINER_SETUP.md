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
