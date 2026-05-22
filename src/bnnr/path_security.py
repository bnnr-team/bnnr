"""Path boundary helpers for CodeQL-safe file operations."""

from __future__ import annotations

import re
from pathlib import Path

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def resolve_directory(path: Path, *, name: str = "directory") -> Path:
    """Resolve *path* and require an existing directory."""
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def ensure_child(parent: Path, child: Path) -> Path:
    """Return *child* resolved and verified to lie under *parent* (or equal)."""
    resolved_parent = parent.resolve()
    resolved_child = child.resolve()
    try:
        resolved_child.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes trusted root {resolved_parent}: {resolved_child}"
        ) from exc
    return resolved_child


def child_path(parent: Path, *parts: str) -> Path:
    """Build a path under a trusted *parent* using literal segments only."""
    resolved_parent = parent.resolve()
    if not parts:
        return resolved_parent
    return ensure_child(resolved_parent, resolved_parent.joinpath(*parts))


def resolve_output_directory(path: Path, *, name: str = "output directory") -> Path:
    """Resolve export/output directory (may be outside any run dir)."""
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    return resolved


def validate_run_id(run_id: str) -> str:
    """Validate dashboard/API run identifiers (single path segment, no traversal)."""
    if not run_id or run_id in {".", ".."}:
        raise ValueError("Invalid run id")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run id")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid run id")
    return run_id


def resolve_events_jsonl(events_file: Path) -> Path:
    """Resolve an events JSONL path and require the expected filename."""
    resolved = events_file.resolve()
    if resolved.name != "events.jsonl":
        raise ValueError(f"Invalid events file path: {events_file}")
    return resolved
