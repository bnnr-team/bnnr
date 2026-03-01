"""Tests for bnnr.dashboard.backend helper functions.

Covers _normalize_run_root, _trim_state_for_api, list_runs,
and basic create_dashboard_app smoke test.
"""

from __future__ import annotations

from typing import Any

from bnnr.dashboard.backend import (
    _normalize_run_root,
    _trim_state_for_api,
    list_runs,
)

# ---------------------------------------------------------------------------
# _normalize_run_root
# ---------------------------------------------------------------------------


class TestNormalizeRunRoot:
    def test_returns_parent_when_events_jsonl_exists(self, tmp_path):
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text("")
        result = _normalize_run_root(run_dir)
        assert result == tmp_path

    def test_returns_same_when_no_events(self, tmp_path):
        result = _normalize_run_root(tmp_path)
        assert result == tmp_path


# ---------------------------------------------------------------------------
# _trim_state_for_api
# ---------------------------------------------------------------------------


class TestTrimStateForApi:
    def test_removes_trim_fields(self):
        state: dict[str, Any] = {
            "epochs": [1, 2, 3],
            "samples": [{"a": 1}],
            "sample_predictions": [{"b": 2}],
            "xai": {"maps": []},
            "metrics_timeline": [{"acc": 0.9}],
        }
        trimmed = _trim_state_for_api(state)
        assert "epochs" not in trimmed
        assert "samples" not in trimmed
        assert "sample_predictions" not in trimmed
        assert "xai" not in trimmed
        assert "metrics_timeline" in trimmed

    def test_limits_confusion_timeline(self):
        # Create more than _MAX_CONFUSION_ENTRIES (30)
        timeline = [{"epoch": i} for i in range(50)]
        state: dict[str, Any] = {"confusion_timeline": timeline}
        trimmed = _trim_state_for_api(state)
        assert len(trimmed["confusion_timeline"]) <= 30

    def test_preserves_small_confusion_timeline(self):
        timeline = [{"epoch": i} for i in range(5)]
        state: dict[str, Any] = {"confusion_timeline": timeline}
        trimmed = _trim_state_for_api(state)
        assert len(trimmed["confusion_timeline"]) == 5

    def test_empty_state(self):
        result = _trim_state_for_api({})
        assert result == {}


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_dir(self, tmp_path):
        runs = list_runs(tmp_path)
        assert runs == []

    def test_nonexistent_dir(self, tmp_path):
        runs = list_runs(tmp_path / "nonexistent")
        assert runs == []

    def test_single_run(self, tmp_path):
        run = tmp_path / "run_001"
        run.mkdir()
        (run / "events.jsonl").write_text('{"type": "test"}\n')
        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["id"] == "run_001"
        assert "path" in runs[0]
        assert "updated_at" in runs[0]
        assert "events_size_bytes" in runs[0]

    def test_multiple_runs(self, tmp_path):
        for name in ("run_a", "run_b", "run_c"):
            d = tmp_path / name
            d.mkdir()
            (d / "events.jsonl").write_text("{}\n")
        runs = list_runs(tmp_path)
        assert len(runs) == 3
        ids = {r["id"] for r in runs}
        assert ids == {"run_a", "run_b", "run_c"}

    def test_skips_dirs_without_events(self, tmp_path):
        (tmp_path / "run_no_events").mkdir()
        (tmp_path / "run_with_events").mkdir()
        (tmp_path / "run_with_events" / "events.jsonl").write_text("{}\n")
        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["id"] == "run_with_events"

    def test_skips_files(self, tmp_path):
        (tmp_path / "not_a_dir.txt").write_text("hello")
        runs = list_runs(tmp_path)
        assert runs == []

    def test_normalizes_when_called_with_run_dir(self, tmp_path):
        """list_runs works when given a run dir (with events.jsonl) instead of root."""
        run = tmp_path / "run_x"
        run.mkdir()
        (run / "events.jsonl").write_text("{}\n")
        # Pass the run dir itself — should normalize to parent
        runs = list_runs(run)
        assert len(runs) == 1


# ---------------------------------------------------------------------------
# create_dashboard_app smoke test
# ---------------------------------------------------------------------------


class TestCreateDashboardApp:
    def test_app_creation(self, tmp_path):
        from bnnr.dashboard.backend import create_dashboard_app

        app = create_dashboard_app(tmp_path)
        assert app is not None
        assert app.title == "BNNR Dashboard API"

    def test_app_with_static_dir(self, tmp_path):
        from bnnr.dashboard.backend import create_dashboard_app

        static = tmp_path / "static"
        static.mkdir()
        (static / "index.html").write_text("<html></html>")
        (static / "assets").mkdir()
        app = create_dashboard_app(tmp_path, static_dir=static)
        assert app is not None

    def test_app_with_auth_token(self, tmp_path):
        from bnnr.dashboard.backend import create_dashboard_app

        app = create_dashboard_app(tmp_path, auth_token="secret123")
        assert app is not None
