"""Tests for bnnr.dashboard.serve — frontend discovery, QR code, LAN IP.

Covers _get_lan_ip, _print_qr_code, _dist_build_mtime,
_frontend_dist_candidates, _frontend_source_candidates,
_find_frontend_dist, and the dashboard __init__ lazy imports.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from bnnr.dashboard.serve import (
    _dist_build_mtime,
    _find_frontend_dist,
    _frontend_dist_candidates,
    _frontend_source_candidates,
    _get_lan_ip,
    _print_qr_code,
)

# ---------------------------------------------------------------------------
# _get_lan_ip
# ---------------------------------------------------------------------------


class TestGetLanIp:
    def test_returns_string(self):
        ip = _get_lan_ip()
        assert isinstance(ip, str)
        # Should be dotted-quad or localhost
        assert "." in ip

    def test_fallback_on_error(self):
        with patch("bnnr.dashboard.serve.socket") as mock_socket:
            mock_socket.AF_INET = 2
            mock_socket.SOCK_DGRAM = 2
            mock_socket.socket.side_effect = OSError("fail")
            ip = _get_lan_ip()
        assert ip == "127.0.0.1"


# ---------------------------------------------------------------------------
# _print_qr_code
# ---------------------------------------------------------------------------


class TestPrintQrCode:
    def test_without_qrcode_library(self, capsys):
        """Without qrcode installed, should print hint."""
        with patch("builtins.__import__", side_effect=ImportError("no qrcode")):
            _print_qr_code("http://localhost:8080")
        captured = capsys.readouterr()
        assert "qrcode" in captured.out.lower() or "pip install" in captured.out.lower()


# ---------------------------------------------------------------------------
# _dist_build_mtime
# ---------------------------------------------------------------------------


class TestDistBuildMtime:
    def test_no_index_html_returns_zero(self, tmp_path):
        assert _dist_build_mtime(tmp_path) == 0.0

    def test_with_index_html(self, tmp_path):
        index = tmp_path / "index.html"
        index.write_text("<html></html>")
        mtime = _dist_build_mtime(tmp_path)
        assert mtime > 0


# ---------------------------------------------------------------------------
# _frontend_dist_candidates
# ---------------------------------------------------------------------------


class TestFrontendDistCandidates:
    def test_returns_list_of_paths(self):
        candidates = _frontend_dist_candidates()
        assert isinstance(candidates, list)
        assert all(isinstance(p, Path) for p in candidates)

    def test_no_duplicates(self):
        candidates = _frontend_dist_candidates()
        resolved = [c.resolve() for c in candidates]
        assert len(resolved) == len(set(resolved))


# ---------------------------------------------------------------------------
# _frontend_source_candidates
# ---------------------------------------------------------------------------


class TestFrontendSourceCandidates:
    def test_returns_list_of_paths(self):
        candidates = _frontend_source_candidates()
        assert isinstance(candidates, list)
        assert all(isinstance(p, Path) for p in candidates)


# ---------------------------------------------------------------------------
# _find_frontend_dist
# ---------------------------------------------------------------------------


class TestFindFrontendDist:
    def test_returns_none_when_nothing_found(self, tmp_path):
        """With no valid dist dirs, should return None."""
        with patch("bnnr.dashboard.serve._frontend_dist_candidates", return_value=[tmp_path / "fake"]):
            with patch("bnnr.dashboard.serve._frontend_source_candidates", return_value=[]):
                result = _find_frontend_dist(auto_build=False)
        assert result is None

    def test_returns_path_when_index_exists(self, tmp_path):
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html></html>")
        with patch("bnnr.dashboard.serve._frontend_dist_candidates", return_value=[dist]):
            result = _find_frontend_dist(auto_build=False)
        assert result == dist

    def test_prefers_newest_dist(self, tmp_path):
        import time

        dist1 = tmp_path / "dist1"
        dist1.mkdir()
        (dist1 / "index.html").write_text("<html>old</html>")

        time.sleep(0.05)

        dist2 = tmp_path / "dist2"
        dist2.mkdir()
        (dist2 / "index.html").write_text("<html>new</html>")

        with patch("bnnr.dashboard.serve._frontend_dist_candidates", return_value=[dist1, dist2]):
            result = _find_frontend_dist(auto_build=False)
        assert result == dist2


# ---------------------------------------------------------------------------
# dashboard __init__.py lazy imports
# ---------------------------------------------------------------------------


class TestDashboardLazyImport:
    def test_create_dashboard_app_import(self):
        from bnnr.dashboard import create_dashboard_app

        assert callable(create_dashboard_app)

    def test_list_runs_import(self):
        from bnnr.dashboard import list_runs

        assert callable(list_runs)

    def test_start_dashboard_import(self):
        from bnnr.dashboard import start_dashboard

        assert callable(start_dashboard)

    def test_unknown_attr_raises(self):
        import bnnr.dashboard

        with pytest.raises(AttributeError):
            _ = bnnr.dashboard.nonexistent_thing  # type: ignore[attr-defined]
