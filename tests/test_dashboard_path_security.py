"""Path traversal guards for dashboard backend, exporter, and events."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from bnnr.dashboard.backend import _resolve_run_dir, _safe_artifact_path  # noqa: E402
from bnnr.dashboard.exporter import export_dashboard_snapshot  # noqa: E402
from bnnr.events import load_events  # noqa: E402
from bnnr.path_security import child_path, validate_run_id


def _make_run_dir(tmp_path) -> tuple:
    run_root = tmp_path / "reports"
    run_dir = run_root / "run_20260101_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "sequence": 1,
                "run_id": "run_20260101_120000",
                "timestamp": "2026-01-01T00:00:00Z",
                "type": "run_started",
                "payload": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_root, run_dir


class TestValidateRunId:
    def test_accepts_run_timestamp_name(self) -> None:
        assert validate_run_id("run_20260101_120000") == "run_20260101_120000"

    def test_rejects_traversal(self) -> None:
        with pytest.raises(ValueError):
            validate_run_id("../etc")


class TestResolveRunDir:
    def test_resolves_valid_run(self, tmp_path) -> None:
        run_root, run_dir = _make_run_dir(tmp_path)
        resolved = _resolve_run_dir(run_root, run_dir.name)
        assert resolved == run_dir.resolve()

    def test_rejects_traversal_run_id(self, tmp_path) -> None:
        run_root, _ = _make_run_dir(tmp_path)
        with pytest.raises(HTTPException) as exc:
            _resolve_run_dir(run_root, "../../etc/passwd")
        assert exc.value.status_code == 400


class TestSafeArtifactPath:
    def test_rejects_escape(self, tmp_path) -> None:
        run_root, run_dir = _make_run_dir(tmp_path)
        with pytest.raises(HTTPException) as exc:
            _safe_artifact_path(run_dir, "../../../etc/passwd")
        assert exc.value.status_code == 400

    def test_returns_file_under_run(self, tmp_path) -> None:
        run_root, run_dir = _make_run_dir(tmp_path)
        artifact = run_dir / "artifacts" / "sample.png"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"png")
        resolved = _safe_artifact_path(run_dir, "artifacts/sample.png")
        assert resolved == artifact.resolve()


class TestExportSiblingOutDir:
    def test_out_dir_can_be_outside_run_dir(self, tmp_path) -> None:
        run_root, run_dir = _make_run_dir(tmp_path)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "a.txt").write_text("artifact", encoding="utf-8")

        out_dir = tmp_path / "exported_snapshot"
        exported = export_dashboard_snapshot(run_dir, out_dir)
        assert exported.exists()
        assert child_path(out_dir.resolve(), "index.html").is_file()


class TestLoadEventsPath:
    def test_rejects_wrong_filename(self, tmp_path) -> None:
        bad = tmp_path / "secrets.json"
        bad.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_events(bad)

    def test_accepts_events_jsonl(self, tmp_path) -> None:
        _, run_dir = _make_run_dir(tmp_path)
        events = load_events(run_dir / "events.jsonl")
        assert len(events) == 1
