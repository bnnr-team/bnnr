"""Tests for the pause/resume functionality.

Covers:
- File-based signal detection in ``_check_pause``
- Dashboard backend ``POST /control`` and ``GET /status`` endpoints
- Integration: pause file blocks, resume unblocks
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")

from bnnr.dashboard.backend import (  # noqa: E402
    PAUSE_SIGNAL_FILENAME,
    create_dashboard_app,
)

# ── Helpers ──────────────────────────────────────────────────────────

def _write_events(run_dir: Path, rows: list | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for row in (rows or []):
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def anyio_backend() -> str:
    """Run async API tests only on asyncio in this environment."""
    return "asyncio"


# ── _check_pause unit tests ──────────────────────────────────────────

def test_check_pause_returns_immediately_when_no_file(temp_dir: Path) -> None:
    """_check_pause must not block when the pause file is absent."""
    from unittest.mock import MagicMock

    reporter = MagicMock()
    reporter.run_dir = temp_dir

    from bnnr.core import BNNRTrainer

    # We only need the method, not a fully initialised trainer, so we
    # construct one via __new__ and manually set the required attributes.
    trainer = object.__new__(BNNRTrainer)
    trainer.reporter = reporter
    trainer.config = MagicMock()
    trainer.config.verbose = False
    trainer.logger = MagicMock()

    t0 = time.monotonic()
    trainer._check_pause()
    assert time.monotonic() - t0 < 1.0, "_check_pause blocked despite no pause file"


def test_check_pause_blocks_then_resumes(temp_dir: Path) -> None:
    """_check_pause must block while .bnnr_pause exists and return once removed."""
    from unittest.mock import MagicMock

    reporter = MagicMock()
    reporter.run_dir = temp_dir

    from bnnr.core import BNNRTrainer

    trainer = object.__new__(BNNRTrainer)
    trainer.reporter = reporter
    trainer.config = MagicMock()
    trainer.config.verbose = False
    trainer.logger = MagicMock()

    pause_file = temp_dir / PAUSE_SIGNAL_FILENAME
    pause_file.touch()

    unblocked = threading.Event()

    def run_check() -> None:
        trainer._check_pause()
        unblocked.set()

    thread = threading.Thread(target=run_check)
    thread.start()

    # Give it a moment to enter the loop
    time.sleep(0.3)
    assert not unblocked.is_set(), "_check_pause returned while pause file exists"

    # Remove the pause file — should unblock
    pause_file.unlink()
    thread.join(timeout=3.0)
    assert unblocked.is_set(), "_check_pause did not return after pause file removal"


# ── Backend endpoint tests ───────────────────────────────────────────

@pytest.mark.anyio
async def test_control_pause_creates_file(temp_dir: Path) -> None:
    _write_events(temp_dir / "run_a", [])
    app = create_dashboard_app(temp_dir)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/api/run/run_a/control", json={"action": "pause"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["paused"] is True
    assert (temp_dir / "run_a" / PAUSE_SIGNAL_FILENAME).exists()


@pytest.mark.anyio
async def test_control_resume_deletes_file(temp_dir: Path) -> None:
    run_dir = temp_dir / "run_a"
    _write_events(run_dir, [])
    (run_dir / PAUSE_SIGNAL_FILENAME).touch()

    app = create_dashboard_app(temp_dir)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/api/run/run_a/control", json={"action": "resume"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["paused"] is False
    assert not (run_dir / PAUSE_SIGNAL_FILENAME).exists()


@pytest.mark.anyio
async def test_control_unknown_action_returns_400(temp_dir: Path) -> None:
    _write_events(temp_dir / "run_a", [])
    app = create_dashboard_app(temp_dir)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/api/run/run_a/control", json={"action": "explode"})
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_status_endpoint(temp_dir: Path) -> None:
    run_dir = temp_dir / "run_a"
    _write_events(run_dir, [])
    app = create_dashboard_app(temp_dir)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/api/run/run_a/status")
    assert resp.status_code == 200
    assert resp.json()["paused"] is False

    (run_dir / PAUSE_SIGNAL_FILENAME).touch()
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/api/run/run_a/status")
    assert resp.json()["paused"] is True
