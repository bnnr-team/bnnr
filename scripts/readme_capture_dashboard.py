#!/usr/bin/env python3
"""Capture README dashboard tab screenshots without loading full bnnr (no torch).

Usage (from repo root):
  python3 scripts/readme_capture_dashboard.py

Requires: fastapi, uvicorn, playwright (install in active env).
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUN_ROOT = ROOT / "examples/classification/reports/butterfly_cooking"
RUN_ID = "run_20260505_150426"
OUT_DIR = ROOT / "docs/assets"
FRONTEND_DIST = SRC / "bnnr/dashboard/frontend/dist"
PORT = 18765

TABS = [
    ("overview", "Overview"),
    ("tree", "Branch Tree"),
    ("metrics", "Metrics"),
    ("samples", "Samples & XAI"),
    ("analysis", "Analysis"),
    ("insight", "Dataset Insight"),
]


def _bootstrap_bnnr_light() -> None:
    """Register minimal bnnr package stubs so dashboard modules load without torch."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    if "bnnr" not in sys.modules:
        pkg = types.ModuleType("bnnr")
        pkg.__path__ = [str(SRC / "bnnr")]
        sys.modules["bnnr"] = pkg

    if "bnnr.dashboard" not in sys.modules:
        dash = types.ModuleType("bnnr.dashboard")
        dash.__path__ = [str(SRC / "bnnr" / "dashboard")]
        sys.modules["bnnr.dashboard"] = dash


def main() -> None:
    import os

    # Use user-level browser cache when sandbox redirects Playwright paths.
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
        home_cache = Path.home() / ".cache/ms-playwright"
        if home_cache.is_dir():
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(home_cache)

    _bootstrap_bnnr_light()
    from bnnr.dashboard.backend import create_dashboard_app  # noqa: E402

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "playwright not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    if not (FRONTEND_DIST / "index.html").exists():
        raise SystemExit(f"Missing frontend dist: {FRONTEND_DIST}. Run: cd dashboard_web && npm run build")

    run_dir = RUN_ROOT / RUN_ID
    if not (run_dir / "events.jsonl").exists():
        raise SystemExit(f"Missing events.jsonl in {run_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    app = create_dashboard_app(
        RUN_ROOT.resolve(),
        static_dir=FRONTEND_DIST.resolve(),
        mode="serve",
    )

    import uvicorn  # noqa: E402

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = __import__("threading").Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{PORT}/?run={RUN_ID}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            import urllib.request

            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/runs", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit("Dashboard server did not become ready in time")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(base_url, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(2000)

        for tab_id, label in TABS:
            page.locator(f'button.tab-btn:has-text("{label}")').click()
            page.wait_for_timeout(1500)
            out = OUT_DIR / f"dashboard-{tab_id}.png"
            page.screenshot(path=str(out), full_page=False)
            print(f"Wrote {out}")

        browser.close()

    server.should_exit = True
    thread.join(timeout=5)
    print("Done.")


if __name__ == "__main__":
    main()
