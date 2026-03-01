"""Tests for event logging and replay utilities."""

from __future__ import annotations

import pytest

from bnnr.events import JsonlEventSink, load_events, replay_events


def test_jsonl_event_sink_serialization_and_replay(temp_dir) -> None:
    events_file = temp_dir / "events.jsonl"
    sink = JsonlEventSink(events_file, run_id="run_test")
    sink.emit("run_started", {"run_name": "run_test"})
    sink.emit("probe_set_initialized", {"probes": [{"sample_id": "sample_11", "class_id": 1, "index": 0}]})
    sink.emit("epoch_end", {"iteration": 0, "epoch": 1, "branch": "baseline", "metrics": {"accuracy": 0.5}})
    sink.emit("branch_evaluated", {"iteration": 1, "branch": "aug_a", "metrics": {"accuracy": 0.6}})
    sink.emit("branch_selected", {"iteration": 1, "selected_branch_id": "iter_1:aug_a"})
    sink.emit("sample_snapshot", {"iteration": 1, "epoch": 1, "sample_pairs": [["artifacts/a.png", "artifacts/b.png"]]})
    sink.emit(
        "sample_prediction_snapshot",
        {"sample_id": "sample_11", "iteration": 1, "epoch": 1, "branch": "aug_a", "predicted_class": 2, "confidence": 0.81},
    )
    sink.emit("xai_snapshot", {"iteration": 1, "epoch": 1, "artifact_paths": ["artifacts/x1.png"]})
    sink.close()

    events = load_events(events_file)
    assert len(events) == 8
    assert events[0]["schema_version"] == "2.1"
    state = replay_events(events)
    assert state["selected_path"][-1] == "iter_1:aug_a"
    assert state["branches"]["aug_a"]["metrics"]["accuracy"] == 0.6
    assert state["xai"][0]["artifact_paths"][0] == "artifacts/x1.png"
    assert state["sample_timelines"]["sample_11"][0]["confidence"] == 0.81


def test_event_sink_rejects_base64_payload(temp_dir) -> None:
    events_file = temp_dir / "events.jsonl"
    sink = JsonlEventSink(events_file, run_id="run_test")
    with pytest.raises(ValueError):
        sink.emit("sample_snapshot", {"blob": "A" * 300})


def test_event_sink_throttling(temp_dir) -> None:
    events_file = temp_dir / "events.jsonl"
    sink = JsonlEventSink(events_file, run_id="run_test", sample_every_epochs=2, xai_every_epochs=3)
    sink.emit("sample_snapshot", {"epoch": 1, "sample_pairs": [["a", "b"]]})
    sink.emit("sample_snapshot", {"epoch": 2, "sample_pairs": [["a", "b"]]})
    sink.emit("xai_snapshot", {"epoch": 1, "xai_paths": ["x"]})
    sink.emit("xai_snapshot", {"epoch": 3, "xai_paths": ["x"]})
    events = load_events(events_file)
    assert len(events) == 2
    assert events[0]["type"] == "sample_snapshot"
    assert events[1]["type"] == "xai_snapshot"
