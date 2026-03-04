"""Public helper to start the BNNR dashboard in a background thread.

Every script — CLI, examples, and user-written — should call
:func:`start_dashboard` instead of rolling its own ``uvicorn.run``
wrapper.  This guarantees that:

* The server always binds to ``0.0.0.0`` so phones on the same LAN can
  connect.
* A QR code with the LAN URL is always printed.
* Frontend discovery / auto-build logic is consistent.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_lan_ip() -> str:
    """Detect the machine's LAN IP address.

    Uses a UDP connect trick (no data is sent) to find the default
    outbound interface address.  Falls back to ``127.0.0.1`` on any error.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def _pick_bind_host(port: int) -> str:
    """Pick the best bind host for the dashboard server.

    Prefer ``0.0.0.0`` for LAN access. If that bind is not permitted in the
    environment, gracefully fall back to ``127.0.0.1`` so local access still
    works.
    """
    for host in ("0.0.0.0", "127.0.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
            return host
        except OSError:
            continue
    # If both probes fail, keep historical behavior so uvicorn surfaces
    # the concrete bind error to the user.
    return "0.0.0.0"


def _print_qr_code(url: str) -> None:
    """Print a terminal-friendly QR code for *url*.

    Requires the optional ``qrcode`` package.  If it is not installed the
    function prints a hint and returns silently.
    """
    try:
        import qrcode  # type: ignore[import-untyped]
    except ImportError:
        print("  (Install 'qrcode' for a scannable QR code: pip install qrcode)")
        return

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)

    matrix = qr.get_matrix()
    print()
    print("  Scan to open on phone:")
    for row in matrix:
        line = "  "
        for cell in row:
            line += "\u2588\u2588" if not cell else "  "
        print(line)


def _dist_build_mtime(dist_dir: Path) -> float:
    """Return the newest file mtime inside *dist_dir* (proxy for build age)."""
    index = dist_dir / "index.html"
    if not index.exists():
        return 0.0
    newest = index.stat().st_mtime
    try:
        for path in dist_dir.rglob("*"):
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
    except OSError:
        pass
    return newest


def _frontend_dist_candidates() -> list[Path]:
    """Return all plausible frontend dist directories (deduplicated, resolved)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "frontend" / "dist",          # installed wheel / editable pkg
        here.parents[3] / "dashboard_web" / "dist", # repo root relative to package src
        Path.cwd() / "dashboard_web" / "dist",      # cwd (user ran CLI from repo root)
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            r = c
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _frontend_source_candidates() -> list[Path]:
    """Return plausible dashboard_web source directories for auto-build."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "dashboard_web",
        Path.cwd() / "dashboard_web",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            r = c
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _find_frontend_dist(auto_build: bool = True) -> Path | None:
    """Locate the newest built frontend dist directory.

    Search order:
    1. ``<package>/dashboard/frontend/dist``  (installed wheel)
    2. ``<repo-root>/dashboard_web/dist``     (editable dev checkout)
    3. ``<cwd>/dashboard_web/dist``           (user running CLI from repo root)

    When multiple valid candidates exist the one with the **newest** build
    (highest mtime across contained files) is returned.  This ensures that
    after ``npm run build`` the freshly built assets are always picked up
    without needing to pass ``--frontend-dist`` manually.

    If *auto_build* is ``True`` and no dist is found but a ``dashboard_web``
    source tree with ``package.json`` exists, ``npm install && npm run build``
    is attempted automatically.
    """
    valid = [d for d in _frontend_dist_candidates() if (d / "index.html").exists()]
    if valid:
        return max(valid, key=_dist_build_mtime)

    if not auto_build or shutil.which("npm") is None:
        return None

    # Try auto-building from the first available source tree
    for frontend_dir in _frontend_source_candidates():
        if not (frontend_dir / "package.json").exists():
            continue
        try:
            print(f"[dashboard] Building frontend in: {frontend_dir}")
            _shell = sys.platform == "win32"  # Windows needs shell=True to find npm.cmd
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, capture_output=True, shell=_shell)
            subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True, capture_output=True, shell=_shell)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            print(f"[dashboard] Frontend auto-build failed in {frontend_dir}: {exc}")
        break  # attempt only the first source candidate

    valid = [d for d in _frontend_dist_candidates() if (d / "index.html").exists()]
    return max(valid, key=_dist_build_mtime) if valid else None


# ── Public API ────────────────────────────────────────────────────────────────


def start_dashboard(
    run_root: Path,
    port: int = 8080,
    *,
    auto_open: bool = True,
    build_frontend: bool = True,
    auth_token: str | None = None,
) -> str:
    """Start the BNNR dashboard server in a background daemon thread.

    The server always binds to ``0.0.0.0`` so it is reachable from any
    device on the same LAN.  A QR code encoding the LAN URL is printed to
    the terminal so users can scan it with a phone.

    Parameters
    ----------
    run_root:
        Directory where ``events.jsonl`` (and artifacts) are written.
    port:
        TCP port for the dashboard HTTP server.
    auto_open:
        Open the local URL in the default browser after a short delay.
    build_frontend:
        If the frontend dist is missing, attempt ``npm run build``.

    Returns
    -------
    str
        The LAN URL (e.g. ``http://192.168.1.42:8080/``).
    """
    try:
        import uvicorn  # noqa: I001
    except ImportError:
        print(
            "[dashboard] Missing dependency 'uvicorn'. "
            "Install with:  pip install -e '.[dashboard]'"
        )
        print("[dashboard] Training will continue without live dashboard.")
        return f"http://127.0.0.1:{port}/"

    try:
        from bnnr.dashboard.backend import create_dashboard_app  # noqa: I001
    except ImportError:
        print(
            "[dashboard] Dashboard backend not available. "
            "Install with:  pip install -e '.[dashboard]'"
        )
        return f"http://127.0.0.1:{port}/"

    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    static_dir = _find_frontend_dist(auto_build=build_frontend)
    dashboard_app = create_dashboard_app(run_root.resolve(), static_dir=static_dir, auth_token=auth_token)

    local_url = f"http://127.0.0.1:{port}/"
    lan_ip = _get_lan_ip()
    lan_url = f"http://{lan_ip}:{port}/"

    bind_host = _pick_bind_host(port)
    if bind_host != "0.0.0.0":
        print(
            "[dashboard] Could not bind to 0.0.0.0; "
            f"falling back to local-only host {bind_host}.",
        )

    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": dashboard_app,
            "host": bind_host,  # noqa: S104
            "port": port,
            "log_level": "warning",
        },
        daemon=True,
        name="BNNRDashboardServer",
    )
    thread.start()

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  BNNR DASHBOARD  |  LIVE MONITORING")
    print("-" * 64)
    print(f"  Local URL    : {local_url}")
    print(f"  Network URL  : {lan_url}")
    if static_dir:
        print(f"  Frontend     : {static_dir}")
    else:
        print("  Frontend     : not built (API-only mode)")
        print("  Build with   : cd dashboard_web && npm install && npm run build")
    print("=" * 64)

    _print_qr_code(lan_url)

    print()

    if auto_open:
        threading.Timer(1.5, lambda: webbrowser.open(local_url)).start()

    return lan_url
