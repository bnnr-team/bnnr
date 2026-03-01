"""Tests for LAN / mobile dashboard access helpers."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from bnnr.dashboard.serve import _get_lan_ip, _print_qr_code

# ── _get_lan_ip ──────────────────────────────────────────────────────────────

def test_get_lan_ip_returns_valid_ipv4() -> None:
    ip = _get_lan_ip()
    # Must be a valid IPv4 address (4 groups of 1-3 digits)
    assert re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip), f"Not a valid IPv4: {ip}"


def test_get_lan_ip_fallback_on_socket_error() -> None:
    with patch("bnnr.dashboard.serve.socket.socket") as mock_sock:
        mock_sock.side_effect = OSError("no network")
        ip = _get_lan_ip()
    assert ip == "127.0.0.1"


def test_get_lan_ip_fallback_on_connect_error() -> None:
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)
    mock_instance.connect.side_effect = OSError("unreachable")

    with patch("bnnr.dashboard.serve.socket.socket", return_value=mock_instance):
        ip = _get_lan_ip()
    assert ip == "127.0.0.1"


# ── _print_qr_code ──────────────────────────────────────────────────────────

def test_print_qr_code_without_library(capsys) -> None:
    """When qrcode is not installed, a helpful hint is printed."""
    with patch.dict("sys.modules", {"qrcode": None}):
        _print_qr_code("http://192.168.1.42:8080/")
    captured = capsys.readouterr().out
    assert "qrcode" in captured.lower()


def test_print_qr_code_with_library(capsys) -> None:
    """When qrcode IS available, the output contains block characters (QR code)."""
    try:
        import qrcode  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("qrcode not installed")

    _print_qr_code("http://192.168.1.42:8080/")
    captured = capsys.readouterr().out
    # QR code rendering uses full-block Unicode chars
    assert "\u2588" in captured
    assert "Scan" in captured


# ── start_dashboard always binds to 0.0.0.0 ──────────────────────────────────

def test_start_dashboard_always_uses_0000(temp_dir) -> None:
    """Dashboard always binds to 0.0.0.0 so it is reachable from LAN."""
    captured_kwargs: dict = {}

    def fake_uvicorn_run(**kwargs) -> None:  # type: ignore[no-untyped-def]
        captured_kwargs.update(kwargs)

    mock_uvicorn = MagicMock()
    mock_uvicorn.run = fake_uvicorn_run

    mock_backend = MagicMock()
    mock_backend.create_dashboard_app = MagicMock(return_value=MagicMock())

    with patch.dict("sys.modules", {
        "uvicorn": mock_uvicorn,
        "bnnr.dashboard.backend": mock_backend,
    }):
        # Re-import to pick up the mocked modules
        import importlib

        import bnnr.dashboard.serve as serve_mod
        importlib.reload(serve_mod)
        serve_mod.start_dashboard(
            run_root=temp_dir / "reports",
            port=9999,
            auto_open=False,
        )

    # Give the background thread a moment to call uvicorn.run
    import time
    time.sleep(0.5)
    assert captured_kwargs.get("host") == "0.0.0.0"
    assert captured_kwargs.get("port") == 9999


def test_start_dashboard_returns_lan_url(temp_dir) -> None:
    """The returned URL should be the LAN URL, not localhost."""
    mock_uvicorn = MagicMock()
    mock_uvicorn.run = MagicMock()

    mock_backend = MagicMock()
    mock_backend.create_dashboard_app = MagicMock(return_value=MagicMock())

    with patch.dict("sys.modules", {
        "uvicorn": mock_uvicorn,
        "bnnr.dashboard.backend": mock_backend,
    }):
        import importlib

        import bnnr.dashboard.serve as serve_mod
        importlib.reload(serve_mod)
        url = serve_mod.start_dashboard(
            run_root=temp_dir / "reports",
            port=9999,
            auto_open=False,
        )

    # Should contain the LAN IP, not 127.0.0.1
    lan_ip = _get_lan_ip()
    assert f"http://{lan_ip}:9999/" == url
