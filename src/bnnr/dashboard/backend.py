"""FastAPI backend for live and replay BNNR dashboard state."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as _PydanticBaseModel

from bnnr.dashboard.exporter import export_dashboard_snapshot
from bnnr.events import IncrementalReplayState, load_events, load_events_from_offset

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    mtime: float
    state: dict[str, Any]
    built_at: float = field(default_factory=time.monotonic)
    byte_offset: int = 0
    replay_state: Optional[IncrementalReplayState] = None

# Minimum seconds between full replays (even if the file changed).
# Keep low for near-live dashboard updates while avoiding replay storms.
_MIN_REPLAY_INTERVAL = 0.4

# WebSocket polling interval for checking new events in events.jsonl.
_WS_POLL_INTERVAL = 0.4

# Fields that are huge but not needed by the dashboard frontend
_TRIM_FIELDS = ("epochs", "samples", "sample_predictions", "xai")

# Max confusion entries to keep (most recent)
_MAX_CONFUSION_ENTRIES = 30

# Signal file name used for pause/resume
PAUSE_SIGNAL_FILENAME = ".bnnr_pause"


class ControlAction(_PydanticBaseModel):
    """Request body for the pause/resume control endpoint."""

    action: str  # "pause" | "resume"


def _trim_state_for_api(state: dict[str, Any]) -> dict[str, Any]:
    """Return a lighter copy of state for API responses.

    Removes bulky fields that the frontend doesn't need and limits
    confusion timeline size to prevent multi-MB responses.
    """
    out = {}
    for k, v in state.items():
        if k in _TRIM_FIELDS:
            continue
        if k == "confusion_timeline" and isinstance(v, list) and len(v) > _MAX_CONFUSION_ENTRIES:
            out[k] = v[-_MAX_CONFUSION_ENTRIES:]
        else:
            out[k] = v
    return out


def _normalize_run_root(run_root: Path) -> Path:
    """If *run_root* itself is a run directory (contains events.jsonl), return its parent.

    This lets users pass either ``reports/`` **or** ``reports/run_NAME`` and
    get the expected behaviour.
    """
    if (run_root / "events.jsonl").exists():
        return run_root.parent
    return run_root


def list_runs(run_root: Path) -> list[dict[str, Any]]:
    run_root = _normalize_run_root(run_root)
    if not run_root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for item in sorted(run_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not item.is_dir():
            continue
        events_file = item / "events.jsonl"
        if not events_file.exists():
            continue
        runs.append(
            {
                "id": item.name,
                "path": str(item),
                "updated_at": item.stat().st_mtime,
                "events_size_bytes": events_file.stat().st_size,
            }
        )
    return runs


def _resolve_run_dir(run_root: Path, run_id: str) -> Path:
    run_dir = (run_root / run_id).resolve()
    if not run_dir.exists() or run_root.resolve() not in run_dir.parents:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_dir


def _safe_artifact_path(run_root: Path, path: str) -> Path:
    resolved_root = run_root.resolve()
    candidate = (run_root / path).resolve()
    if resolved_root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        parts = Path(path).parts
        if len(parts) > 1:
            alt = (run_root / Path(*parts[1:])).resolve()
            if alt.exists() and alt.is_file() and resolved_root in alt.parents:
                return alt
        raise HTTPException(status_code=404, detail="Artifact not found")
    return candidate


def create_dashboard_app(
    run_root: Path,
    static_dir: Optional[Path] = None,
    auth_token: Optional[str] = None,
    mode: str = "live",
) -> FastAPI:
    """Create the FastAPI dashboard application.

    Parameters
    ----------
    run_root : Path
        Root directory containing run sub-directories.
    static_dir : Path | None
        Optional directory with built frontend assets.
    auth_token : str | None
        Optional token to protect control endpoints (pause/resume).
        Falls back to the ``BNNR_DASHBOARD_TOKEN`` environment variable.
    mode : str
        Dashboard mode — ``"live"`` during active training, ``"serve"`` for
        replay of completed/interrupted runs.
    """
    run_root = _normalize_run_root(run_root)
    app = FastAPI(title="BNNR Dashboard API", version="0.3.0")
    state_cache: dict[str, _CacheEntry] = {}
    static_dir = static_dir.resolve() if static_dir is not None else None

    # Resolve auth token (explicit > env var > disabled)
    _auth_token: Optional[str] = auth_token or os.environ.get("BNNR_DASHBOARD_TOKEN")

    async def _require_control_auth(
        x_bnnr_token: Optional[str] = Header(None, alias="X-BNNR-Token"),
    ) -> None:
        """Dependency that gates mutating control endpoints behind a token."""
        if _auth_token is None:
            return  # auth not configured — allow all
        if x_bnnr_token != _auth_token:
            raise HTTPException(status_code=403, detail="Invalid or missing control token")

    def _events_file(run_id: str) -> Path:
        run_dir = _resolve_run_dir(run_root, run_id)
        events_file = run_dir / "events.jsonl"
        if not events_file.exists():
            raise HTTPException(status_code=404, detail="Run has no events.jsonl")
        return events_file

    def _state_for_run(run_id: str) -> dict[str, Any]:
        events_file = _events_file(run_id)
        mtime = events_file.stat().st_mtime
        now = time.monotonic()
        cached = state_cache.get(run_id)

        # Use cache if:
        # a) file hasn't changed, OR
        # b) we replayed recently (within _MIN_REPLAY_INTERVAL)
        if cached is not None:
            if cached.mtime >= mtime:
                return cached.state
            if (now - cached.built_at) < _MIN_REPLAY_INTERVAL:
                return cached.state

        # Incremental replay: if we have a cached replay state and the
        # file grew (but was not truncated), only read and apply new events.
        file_size = events_file.stat().st_size
        if (
            cached is not None
            and cached.replay_state is not None
            and cached.byte_offset > 0
            and file_size >= cached.byte_offset
        ):
            new_events, new_offset = load_events_from_offset(events_file, cached.byte_offset)
            if new_events:
                cached.replay_state.apply_events(new_events)
            state = cached.replay_state.finalize()
            state_cache[run_id] = _CacheEntry(
                mtime=mtime, state=state, built_at=now,
                byte_offset=new_offset, replay_state=cached.replay_state,
            )
            return state

        # Full replay (first time or file was truncated/replaced)
        irs = IncrementalReplayState()
        all_events = load_events(events_file)
        irs.apply_events(all_events)
        state = irs.finalize()
        state_cache[run_id] = _CacheEntry(
            mtime=mtime, state=state, built_at=now,
            byte_offset=file_size, replay_state=irs,
        )
        return state

    @app.get("/api/runs")
    def api_runs() -> dict[str, Any]:
        return {"runs": list_runs(run_root)}

    @app.get("/api/mode")
    def api_mode() -> dict[str, str]:
        return {"mode": mode}

    @app.get("/")
    def app_root() -> Any:
        if static_dir is not None and (static_dir / "index.html").exists():
            return FileResponse(static_dir / "index.html")
        return {
            "message": "BNNR Dashboard backend is running.",
            "hint": "Build frontend and pass --frontend-dist to serve static UI.",
        }

    if static_dir is not None:
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="dashboard-assets")

        @app.get("/favicon.ico")
        def favicon() -> Any:
            fav = static_dir / "favicon.ico"
            if fav.exists():
                return FileResponse(fav)
            raise HTTPException(status_code=404, detail="favicon.ico not found")

    @app.get("/api/run/{run_id}/state")
    def api_run_state(run_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        return _trim_state_for_api(state)

    @app.get("/api/run/{run_id}/kpi-trends")
    def api_run_kpi_trends(run_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        return {
            "metrics_timeline": state.get("metrics_timeline", []),
            "confusion_timeline": state.get("confusion_timeline", [])[-_MAX_CONFUSION_ENTRIES:],
        }

    @app.get("/api/run/{run_id}/branch-graph")
    def api_run_branch_graph(run_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        return {
            "branch_graph": state.get("branch_graph", {"nodes": [], "edges": []}),
            "selected_path": state.get("selected_path", ["baseline"]),
        }

    @app.get("/api/run/{run_id}/class-metrics")
    def api_run_class_metrics(run_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        return {"per_class_timeline": state.get("per_class_timeline", {})}

    @app.get("/api/run/{run_id}/xai-insights")
    def api_run_xai_insights(run_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        return {
            "xai_insights_timeline": state.get("xai_insights_timeline", []),
            "per_class_timeline": state.get("per_class_timeline", {}),
        }

    @app.get("/api/run/{run_id}/samples")
    def api_run_samples(run_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        return {"probe_set": state.get("probe_set", []), "sample_timelines": state.get("sample_timelines", {})}

    @app.get("/api/run/{run_id}/sample/{sample_id}/timeline")
    def api_run_sample_timeline(run_id: str, sample_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        rows = state.get("sample_timelines", {}).get(sample_id, [])
        return {"sample_id": sample_id, "timeline": rows}

    @app.post("/api/run/{run_id}/export", dependencies=[Depends(_require_control_auth)])
    def api_run_export(run_id: str) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_root, run_id)
        out_dir = run_dir / "dashboard_export" / datetime.now().strftime("%Y%m%d_%H%M%S")
        exported = export_dashboard_snapshot(run_dir=run_dir, out_dir=out_dir, frontend_dist=static_dir)
        return {"ok": True, "path": str(exported), "index": str(exported / "index.html")}

    @app.get("/api/run/{run_id}/events")
    def api_run_events(
        run_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=2000),
    ) -> dict[str, Any]:
        events_file = _events_file(run_id)
        # Read only the lines we need instead of loading the entire file.
        # For large runs (10k+ events) this avoids deserialising everything.
        events: list[dict[str, Any]] = []
        total = 0
        with events_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if total >= offset and len(events) < limit:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping invalid JSONL line in %s (api offset=%s line=%s): %s",
                            events_file,
                            offset,
                            total,
                            exc,
                        )
                total += 1
        return {"events": events, "offset": offset, "limit": limit, "total": total}

    @app.get("/artifacts/{path:path}")
    def api_artifacts(path: str) -> FileResponse:
        # path format: <run_id>/artifacts/<...>
        artifact = _safe_artifact_path(run_root, path)
        return FileResponse(artifact)

    @app.get("/api/run/{run_id}/branch/{branch_id}")
    def api_run_branch_detail(run_id: str, branch_id: str) -> dict[str, Any]:
        state = _state_for_run(run_id)
        branch_data = state.get("branches", {}).get(branch_id, {})
        node_info = None
        for n in state.get("branch_graph", {}).get("nodes", []):
            if n.get("id") == branch_id:
                node_info = n
                break
        decision = None
        for d in state.get("decision_history", []):
            if d.get("selected_branch_id") == branch_id:
                decision = d
                break
        snapshot = state.get("sample_branch_snapshots", {}).get(branch_id, {})
        return {
            "branch_id": branch_id,
            "branch_data": branch_data,
            "node_info": node_info,
            "decision": decision,
            "snapshot": snapshot,
        }

    # ── Pause / Resume control ─────────────────────────────────────────
    @app.post("/api/run/{run_id}/control", dependencies=[Depends(_require_control_auth)])
    async def api_run_control(run_id: str, body: ControlAction) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_root, run_id)
        pause_file = run_dir / PAUSE_SIGNAL_FILENAME
        if body.action == "pause":
            pause_file.touch(exist_ok=True)
            return {"ok": True, "paused": True}
        elif body.action == "resume":
            if pause_file.exists():
                pause_file.unlink(missing_ok=True)
            return {"ok": True, "paused": False}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {body.action}")

    @app.get("/api/run/{run_id}/status")
    async def api_run_status(run_id: str) -> dict[str, Any]:
        run_dir = _resolve_run_dir(run_root, run_id)
        paused = (run_dir / PAUSE_SIGNAL_FILENAME).exists()
        return {"paused": paused}

    @app.websocket("/ws/run/{run_id}")
    async def ws_run(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        try:
            events_file = _events_file(run_id)
        except HTTPException as exc:
            # WebSocket endpoints cannot return normal HTTPException responses
            # after handshake. Close gracefully with an app-specific code.
            await websocket.send_json({"type": "error", "message": exc.detail})
            await websocket.close(code=4404)
            return
        offset = events_file.stat().st_size
        last_heartbeat = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                if now - last_heartbeat >= 2.0:
                    await websocket.send_json({"type": "heartbeat"})
                    last_heartbeat = now
                await asyncio.sleep(_WS_POLL_INTERVAL)
                if not events_file.exists():
                    continue
                size = events_file.stat().st_size
                if size <= offset:
                    continue
                lines_sent = 0
                with events_file.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        # Don't send raw events — just notify the frontend to refetch
                        lines_sent += 1
                    offset = handle.tell()
                # Send a single notification so frontend knows to refresh
                if lines_sent > 0:
                    await websocket.send_json({"type": "new_events", "count": lines_sent})
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
            await websocket.close()

    if static_dir is not None:
        @app.get("/{ui_path:path}")
        def spa_fallback(ui_path: str) -> Any:
            if ui_path.startswith("api/") or ui_path.startswith("ws/") or ui_path.startswith("artifacts/"):
                raise HTTPException(status_code=404, detail="Not found")
            # Serve actual static files (logos, images, etc.) if they exist
            candidate = (static_dir / ui_path).resolve()
            if (
                candidate.is_file()
                and static_dir.resolve() in candidate.parents
            ):
                return FileResponse(candidate)
            # SPA fallback — serve index.html for client-side routing
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            raise HTTPException(status_code=404, detail="UI not found")

    return app
