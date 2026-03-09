"""Report schema for bnnr analyze: structured findings, recommendations, executive summary.

Schema is versioned for future compare and extensions. Used only by analyze pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

REPORT_SCHEMA_VERSION = "0.2.0"


@dataclass
class ExecutiveSummary:
    """30-second overview: health, key findings, top actions."""

    health_status: str  # e.g. "ok", "warning", "critical"
    health_score: float  # 0–1
    key_findings: list[str] = field(default_factory=list)
    top_actions: list[str] = field(default_factory=list)
    critical_classes: list[str] = field(default_factory=list)
    severity: str = "unknown"  # low, medium, high, critical

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """Root-cause oriented finding with evidence and confidence."""

    title: str
    finding_type: str  # e.g. zero_recall_class, confused_pair, low_xai_quality
    description: str
    evidence: list[str] = field(default_factory=list)
    interpretation: str = ""
    confidence: str = "medium"  # low, medium, high
    severity: str = "medium"  # low, medium, high, critical
    class_ids: list[str] = field(default_factory=list)
    linked_pattern_ids: list[str] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    """Structured recommendation linked to findings."""

    title: str
    scope: str  # e.g. "class_3", "global", "confused_pair_3_5"
    why: str
    action: str
    expected_impact: str = ""
    confidence: str = "medium"
    priority: int = 0  # lower = higher priority
    linked_finding_ids: list[str] = field(default_factory=list)
    example_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassDiagnostic:
    """Per-class metrics and severity for ranking."""

    class_id: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    pred_count: int  # number of samples predicted as this class
    severity: str = "ok"  # ok, warning, critical
    rank: int = 0  # 1 = worst

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailurePattern:
    """Extended failure pattern with type and evidence."""

    pattern_type: str  # confused_pair, zero_recall_class, low_xai_quality, etc.
    description: str
    severity: str = "medium"
    count: int = 0
    class_a: str = ""
    class_b: str = ""
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XAIClassSummary:
    """Aggregated XAI quality and flags for a single class."""

    class_id: str
    mean_quality: float
    sample_count: int
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XAIExample:
    """Single XAI example (image + overlay + short explanation)."""

    index: int
    true_label: int
    pred_label: int
    confidence: float
    image_path: str
    overlay_path: str
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualitySummary:
    """High-level dataset health summary used by analyze."""

    total_samples_scanned: int
    duplicate_pairs: int
    flagged_images: int
    class_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossValidationResults:
    """Optional k-fold cross-validation results."""

    n_folds: int
    global_metrics: dict[str, float] = field(default_factory=dict)
    per_fold_metrics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterView:
    """Cluster visualisation for confusing examples (projected to 2D/3D)."""

    name: str
    description: str
    points: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialize_for_json(obj: Any) -> Any:
    """Convert dataclass/list/dict to JSON-serializable form."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        return [serialize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    return obj
