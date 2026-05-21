# Maintainer setup (week 1 community)

Canonical **Discussions** URL: `https://github.com/bnnr-team/bnnr/discussions` (repo-level; use this on the website and in issue templates).

## One-time GitHub settings

### 1. Enable Discussions (plan 4.3)

Repository **Settings → General → Features → Discussions** → enable.

Suggested categories:

- General (Q&A)
- Show and Tell
- Ideas
- Help

### 2. Sync labels (plan 4.4)

```bash
gh label sync --file .github/labels.yml
```

### 3. Seed good first issues (plan 4.4)

See the list in [CONTRIBUTING.md](../CONTRIBUTING.md). Example:

```bash
gh issue create --title "Add type hints to a small module in src/bnnr/" \
  --label "good first issue,documentation" --body "Starter task for new contributors."
```

Repeat for 5–10 issues from the CONTRIBUTING list.
