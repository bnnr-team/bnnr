"""Event logging, replay state, and JSONL event stream utilities."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

EventType = Literal[
    "run_started",
    "probe_set_initialized",
    "dataset_profile",
    "pipeline_phase",
    "branch_created",
    "epoch_end",
    "branch_evaluated",
    "branch_selected",
    "sample_snapshot",
    "sample_prediction_snapshot",
    "xai_snapshot",
    "pipeline_complete",
]

EVENT_SCHEMA_VERSION = "2.1"
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]{256,}$")


@dataclass(frozen=True)
class BNNREvent:
    schema_version: str
    sequence: int
    run_id: str
    timestamp: str
    type: EventType
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
        }


class EventSink(Protocol):
    def emit(self, event_type: EventType, payload: dict[str, Any], *, force: bool = False) -> None:
        ...

    def close(self) -> None:
        ...


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlEventSink:
    def __init__(
        self,
        target_file: Path,
        run_id: str,
        *,
        sample_every_epochs: int = 1,
        xai_every_epochs: int = 1,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self.target_file = target_file
        self.run_id = run_id
        self.sample_every_epochs = max(1, sample_every_epochs)
        self.xai_every_epochs = max(1, xai_every_epochs)
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._sequence = 0
        self._lock = threading.Lock()
        self._last_emit_ts = 0.0
        self._last_emit_by_type: dict[str, float] = {}
        self.target_file.parent.mkdir(parents=True, exist_ok=True)

    def _throttled(self, event_type: EventType, payload: dict[str, Any]) -> bool:
        if event_type == "sample_snapshot":
            epoch = int(payload.get("epoch", 1))
            if epoch % self.sample_every_epochs != 0:
                return True
        if event_type == "xai_snapshot":
            epoch = int(payload.get("epoch", 1))
            if epoch % self.xai_every_epochs != 0:
                return True
        if self.min_interval_seconds > 0:
            now = time.monotonic()
            last_ts = self._last_emit_by_type.get(event_type, self._last_emit_ts)
            if now - last_ts < self.min_interval_seconds:
                return True
            self._last_emit_ts = now
            self._last_emit_by_type[event_type] = now
        return False

    def emit(self, event_type: EventType, payload: dict[str, Any], *, force: bool = False) -> None:
        if not force and self._throttled(event_type, payload):
            return
        if _contains_large_base64(payload):
            raise ValueError("Event payload looks like base64 blob; store artifact paths instead.")
        with self._lock:
            self._sequence += 1
            event = BNNREvent(
                schema_version=EVENT_SCHEMA_VERSION,
                sequence=self._sequence,
                run_id=self.run_id,
                timestamp=_utc_now_iso(),
                type=event_type,
                payload=payload,
            )
            with self.target_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")

    def close(self) -> None:
        return None


def _contains_large_base64(value: Any) -> bool:
    if isinstance(value, str):
        compact = value.strip()
        return len(compact) >= 256 and bool(_BASE64_RE.match(compact))
    if isinstance(value, dict):
        return any(_contains_large_base64(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_large_base64(v) for v in value)
    return False


def load_events(events_file: Path) -> list[dict[str, Any]]:
    if not events_file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in events_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_events_from_offset(events_file: Path, byte_offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read new events starting from *byte_offset*.

    Returns ``(new_events, new_byte_offset)`` so the caller can resume
    from where it left off on the next call.
    """
    if not events_file.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    with events_file.open("r", encoding="utf-8") as fh:
        fh.seek(byte_offset)
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
        new_offset = fh.tell()
    return rows, new_offset


class IncrementalReplayState:
    """Holds mutable replay accumulators so events can be applied incrementally."""

    def __init__(self) -> None:
        self.metrics_timeline: list[dict[str, Any]] = []
        self.confusion_timeline: list[dict[str, Any]] = []
        self.per_class_timeline: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.sample_timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.xai_insights_timeline: list[dict[str, Any]] = []
        self.branch_graph_nodes: dict[str, dict[str, Any]] = {
            "root:baseline": {
                "id": "root:baseline",
                "label": "baseline",
                "depth": 0,
                "status": "root",
                "best": True,
            }
        }
        self.branch_graph_edges: list[dict[str, str]] = []
        self.probe_set: list[dict[str, Any]] = []
        self.decision_history: list[dict[str, Any]] = []
        self.sample_branch_snapshots: dict[str, dict[str, Any]] = {}

        self.state: dict[str, Any] = {
            "run": {},
            "epochs": [],
            "branches": {},
            "selected_path": ["root:baseline"],
            "selected_path_edges": [],
            "samples": [],
            "sample_predictions": [],
            "xai": [],
            "metrics_timeline": self.metrics_timeline,
            "confusion_timeline": self.confusion_timeline,
            "per_class_timeline": self.per_class_timeline,
            "sample_timelines": self.sample_timelines,
            "branch_graph": {"nodes": [], "edges": self.branch_graph_edges},
            "probe_set": self.probe_set,
            "decision_history": self.decision_history,
            "sample_branch_snapshots": self.sample_branch_snapshots,
            "xai_insights_timeline": self.xai_insights_timeline,
            "pipeline_phase": None,
            "pipeline_complete": False,
        }

    def apply_events(self, events: list[dict[str, Any]]) -> None:
        """Apply a batch of events to the running state."""
        _apply_events_to_state(events, self.state, self)

    def finalize(self) -> dict[str, Any]:
        """Return the finalized state dict (sorts timelines, builds graph)."""
        return _finalize_state(self.state, self)


def _apply_events_to_state(
    events: list[dict[str, Any]],
    state: dict[str, Any],
    acc: IncrementalReplayState,
) -> None:
    """Core event-processing loop shared by full and incremental replay."""
    metrics_timeline = acc.metrics_timeline
    confusion_timeline = acc.confusion_timeline
    per_class_timeline = acc.per_class_timeline
    sample_timelines = acc.sample_timelines
    xai_insights_timeline = acc.xai_insights_timeline
    branch_graph_nodes = acc.branch_graph_nodes
    branch_graph_edges = acc.branch_graph_edges
    probe_set = acc.probe_set
    decision_history = acc.decision_history
    sample_branch_snapshots = acc.sample_branch_snapshots

    for event in events:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "run_started":
            state["run"] = payload
            metric_units = payload.get("metric_units")
            if not isinstance(metric_units, dict):
                # Determine task type from config embedded in the payload
                config = payload.get("config", {})
                task = config.get("task", "classification") if isinstance(config, dict) else "classification"
                if task == "multilabel":
                    payload["metric_units"] = {
                        "f1_samples": "%",
                        "f1_macro": "%",
                        "accuracy": "%",
                        "loss": "unitless",
                    }
                else:
                    payload["metric_units"] = {
                        "accuracy": "%",
                        "f1_macro": "%",
                        "loss": "unitless",
                    }
            # Store task type at top level for frontend to distinguish
            config = payload.get("config", {})
            if isinstance(config, dict):
                state["task"] = config.get("task", "classification")
                if not isinstance(payload.get("sample_every"), int):
                    payload["sample_every"] = int(config.get("event_sample_every_epochs", 1))
                if not isinstance(payload.get("xai_every"), int):
                    payload["xai_every"] = int(config.get("event_xai_every_epochs", 1))
            if not isinstance(payload.get("class_names"), list):
                payload["class_names"] = []
        elif event_type == "probe_set_initialized":
            if isinstance(payload.get("probes"), list):
                probe_set = payload.get("probes", [])
                state["probe_set"] = probe_set
                class_ids = sorted({int(p.get("class_id", 0)) for p in probe_set if isinstance(p, dict)})
                if class_ids and not state["run"].get("class_names"):
                    state["run"]["class_names"] = [f"class_{cid}" for cid in class_ids]
        elif event_type == "dataset_profile":
            state["dataset_profile"] = payload
            # Also update class_names in run metadata if provided
            class_names = payload.get("class_names")
            if isinstance(class_names, list) and class_names:
                state["run"]["class_names"] = class_names
        elif event_type == "pipeline_phase":
            # Track the current pipeline phase for dashboard display.
            phase_name = payload.get("phase", "")
            phase_status = payload.get("status", "started")
            if phase_status == "started":
                state["pipeline_phase"] = {
                    "phase": phase_name,
                    "message": payload.get("message", ""),
                    "timestamp": event.get("timestamp", ""),
                }
            elif phase_status in ("completed", "skipped"):
                state["pipeline_phase"] = None
        elif event_type == "pipeline_complete":
            state["pipeline_complete"] = True
            state["pipeline_phase"] = None
        elif event_type == "branch_created":
            branch_id = str(payload.get("branch_id", ""))
            parent_id = str(payload.get("parent_id", "root:baseline"))
            if not branch_id:
                continue
            parent_depth = int(branch_graph_nodes.get(parent_id, {}).get("depth", -1))
            depth = parent_depth + 1 if parent_depth >= 0 else int(payload.get("depth", payload.get("iteration", 0)))
            if branch_id not in branch_graph_nodes:
                branch_graph_nodes[branch_id] = {
                    "id": branch_id,
                    "label": str(payload.get("label", branch_id)),
                    "depth": depth,
                    "status": "candidate",
                    "best": False,
                    "augmentation": payload.get("augmentation"),
                    "iteration": int(payload.get("iteration", 0)),
                }
            else:
                branch_graph_nodes[branch_id]["depth"] = depth
            edge = {"from": parent_id, "to": branch_id}
            if edge not in branch_graph_edges:
                branch_graph_edges.append(edge)
        elif event_type == "epoch_end":
            state["epochs"].append(payload)
            raw_metrics = payload.get("metrics", {})
            row: dict[str, Any] = {
                "iteration": int(payload.get("iteration", 0)),
                "epoch": int(payload.get("epoch", 0)),
                "branch": str(payload.get("branch", "baseline")),
                "loss": float(raw_metrics.get("loss", 0.0)),
                # Classification metrics
                "accuracy": float(raw_metrics.get("accuracy", 0.0)),
                "f1_macro": float(raw_metrics.get("f1_macro", 0.0)),
                # Multilabel metrics (may be 0 for non-multilabel runs)
                "f1_samples": float(raw_metrics.get("f1_samples", 0.0)),
                # Detection metrics (may be 0 for classification runs)
                "map_50": float(raw_metrics.get("map_50", 0.0)),
                "map_50_95": float(raw_metrics.get("map_50_95", 0.0)),
                "is_best_epoch": bool(payload.get("is_best_epoch", False)),
            }
            # Forward any additional metrics (fbeta_*, jaccard_*, etc.)
            # so the dashboard and downstream consumers can access them.
            _known_keys = {"loss", "accuracy", "f1_macro", "f1_samples", "map_50", "map_50_95"}
            for mk, mv in raw_metrics.items():
                if mk not in _known_keys:
                    try:
                        row[mk] = float(mv)
                    except (TypeError, ValueError):
                        pass
            metrics_timeline.append(row)
            per_class = payload.get("per_class_accuracy", {})
            xai_insights_raw = payload.get("xai_insights", {})
            xai_diagnoses_raw = payload.get("xai_diagnoses", {})
            if isinstance(per_class, dict):
                for cls, metric_row in per_class.items():
                    if isinstance(metric_row, dict):
                        entry: dict[str, Any] = {
                            "iteration": row["iteration"],
                            "epoch": row["epoch"],
                            "branch": row["branch"],
                            "accuracy": float(metric_row["accuracy"] if "accuracy" in metric_row else metric_row.get("f1", 0.0)),
                            "support": int(metric_row.get("support", 0)),
                            "precision": float(metric_row.get("precision", 0.0)),
                            "recall": float(metric_row.get("recall", 0.0)),
                            "f1": float(metric_row.get("f1", 0.0)),
                        }
                        # Attach per-class XAI insight text if present
                        insight = metric_row.get("xai_insight") or (xai_insights_raw.get(str(cls)) if isinstance(xai_insights_raw, dict) else None)
                        if insight:
                            entry["xai_insight"] = str(insight)
                        # Attach structured diagnosis if present
                        diag = xai_diagnoses_raw.get(str(cls)) if isinstance(xai_diagnoses_raw, dict) else None
                        if isinstance(diag, dict):
                            entry["severity"] = diag.get("severity", "")
                            entry["quality_score"] = diag.get("quality_score", 0.0)
                            entry["trend"] = diag.get("trend", "")
                            entry["confused_with"] = diag.get("confused_with", [])
                            entry["short_text"] = diag.get("short_text", "")
                            entry["quality_breakdown"] = diag.get("quality_breakdown", {})
                            entry["augmentation_impact"] = diag.get("augmentation_impact", "")
                            entry["baseline_delta"] = diag.get("baseline_delta", {})
                        per_class_timeline[str(cls)].append(entry)
            # Store xai_insights as a timeline entry
            if isinstance(xai_insights_raw, dict) and xai_insights_raw:
                timeline_entry: dict[str, Any] = {
                    "iteration": row["iteration"],
                    "epoch": row["epoch"],
                    "branch": row["branch"],
                    "insights": xai_insights_raw,
                }
                if isinstance(xai_diagnoses_raw, dict) and xai_diagnoses_raw:
                    timeline_entry["diagnoses"] = xai_diagnoses_raw
                xai_insights_timeline.append(timeline_entry)
            confusion = payload.get("confusion", {})
            if isinstance(confusion, dict) and confusion.get("matrix"):
                confusion_timeline.append(
                    {
                        "iteration": row["iteration"],
                        "epoch": row["epoch"],
                        "branch": row["branch"],
                        "matrix": confusion["matrix"],
                        "labels": confusion.get("labels", []),
                    }
                )
        elif event_type == "branch_evaluated":
            branch_id = str(payload.get("branch_id", payload.get("branch", "unknown")))
            state["branches"][branch_id] = payload
            node = branch_graph_nodes.setdefault(
                branch_id,
                {
                    "id": branch_id,
                    "label": str(payload.get("branch", branch_id)),
                    "depth": int(payload.get("iteration", 0)),
                    "status": "candidate",
                    "best": False,
                },
            )
            node["status"] = "evaluated"
            # Store best-epoch info for dashboard display
            best_epoch = payload.get("best_epoch")
            total_epochs = payload.get("total_epochs")
            if best_epoch is not None:
                node["best_epoch"] = int(best_epoch)
            if total_epochs is not None:
                node["total_epochs"] = int(total_epochs)
        elif event_type == "branch_selected":
            selected = str(payload.get("selected_branch_id", payload.get("selected_branch", "none")))
            iteration = int(payload.get("iteration", 0))
            if selected != "none":
                if selected not in state["selected_path"]:
                    parent = state["selected_path"][-1]
                    state["selected_path"].append(selected)
                    state["selected_path_edges"].append({"from": parent, "to": selected})
                    edge = {"from": parent, "to": selected}
                    if edge not in branch_graph_edges:
                        branch_graph_edges.append(edge)
                node = branch_graph_nodes.setdefault(
                    selected,
                    {"id": selected, "label": selected, "depth": iteration, "best": False},
                )
                node["best"] = True
                node["status"] = "selected"
                state["branch_decision"] = payload
            else:
                # "none" selected — mark all candidates from this iteration as rejected
                for nid, node_data in branch_graph_nodes.items():
                    if node_data.get("iteration") == iteration and node_data.get("status") == "evaluated":
                        node_data["status"] = "rejected"
            results = payload.get("results", {})
            baseline_metrics = payload.get("baseline_metrics", {})
            baseline_acc = None
            if isinstance(baseline_metrics, dict):
                raw = baseline_metrics.get("accuracy")
                if isinstance(raw, (int, float)):
                    baseline_acc = float(raw)
            decision_history.append(
                {
                    "iteration": int(payload.get("iteration", 0)),
                    "selected_branch_id": selected,
                    "selected_branch_label": str(payload.get("selected_branch", selected)),
                    "decision_reason": str(payload.get("decision_reason", "")),
                    "baseline_accuracy": baseline_acc,
                    "results": results if isinstance(results, dict) else {},
                }
            )
        elif event_type == "sample_snapshot":
            state["samples"].append(payload)
            branch_id = str(payload.get("branch_id", ""))
            if branch_id:
                sample_branch_snapshots[branch_id] = payload
        elif event_type == "sample_prediction_snapshot":
            state["sample_predictions"].append(payload)
            sample_id = str(payload.get("sample_id", "unknown"))
            rows = sample_timelines[sample_id]
            # Merge duplicate timeline points (same sample+iter+epoch+branch),
            # upgrading partial payloads with later richer artifacts/details.
            merged = False
            for idx, row in enumerate(rows):
                if (
                    int(row.get("iteration", 0)) == int(payload.get("iteration", 0))
                    and int(row.get("epoch", 0)) == int(payload.get("epoch", 0))
                    and str(row.get("branch", "")) == str(payload.get("branch", ""))
                    and str(row.get("branch_id", "")) == str(payload.get("branch_id", ""))
                ):
                    _raw_existing = row.get("artifacts")
                    existing_artifacts: dict[str, Any] = _raw_existing if isinstance(_raw_existing, dict) else {}
                    _raw_incoming = payload.get("artifacts")
                    incoming_artifacts: dict[str, Any] = _raw_incoming if isinstance(_raw_incoming, dict) else {}
                    merged_artifacts = {
                        "original": incoming_artifacts.get("original") or existing_artifacts.get("original"),
                        "augmented": incoming_artifacts.get("augmented") or existing_artifacts.get("augmented"),
                        "xai": incoming_artifacts.get("xai") or existing_artifacts.get("xai"),
                    }
                    rows[idx] = {
                        **row,
                        **payload,
                        "artifacts": merged_artifacts,
                    }
                    merged = True
                    break
            if not merged:
                rows.append(payload)
        elif event_type == "xai_snapshot":
            state["xai"].append(payload)


def _finalize_state(state: dict[str, Any], acc: IncrementalReplayState) -> dict[str, Any]:
    """Sort timelines and build the final branch graph from accumulators."""
    state["metrics_timeline"] = sorted(
        acc.metrics_timeline, key=lambda r: (r["iteration"], r["epoch"]),
    )
    state["sample_timelines"] = {
        sample_id: sorted(
            rows, key=lambda r: (int(r.get("iteration", 0)), int(r.get("epoch", 0))),
        )
        for sample_id, rows in acc.sample_timelines.items()
    }
    state["per_class_timeline"] = dict(acc.per_class_timeline)
    state["branch_graph"] = {
        "nodes": sorted(
            acc.branch_graph_nodes.values(),
            key=lambda n: (int(n.get("depth", 0)), str(n.get("id", ""))),
        ),
        "edges": acc.branch_graph_edges,
    }
    return state


def replay_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Full replay from scratch (backward-compatible API)."""
    irs = IncrementalReplayState()
    irs.apply_events(events)
    return irs.finalize()
