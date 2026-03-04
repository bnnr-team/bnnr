import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ConfusionEntry, DataQualityWarning, DatasetProfile, DuplicateGroup, PerClassRow } from "../types";
import { useChartColors } from "../ThemeContext";
import { ConfusionChord } from "./ConfusionChord";
import { resolveArtifact } from "../hooks/useRunState";

const BAR_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2",
  "#d97706", "#4f46e5", "#0f766e", "#be185d", "#65a30d",
  "#1d4ed8", "#15803d", "#b91c1c", "#6d28d9", "#0e7490",
];

interface Props {
  datasetProfile?: DatasetProfile;
  perClassTimeline: Record<string, PerClassRow[]>;
  confusionTimeline: ConfusionEntry[];
  classNames: string[];
  activeRun: string;
  offline: boolean;
}

function imbalanceLabel(ratio: number): { text: string; cls: string } {
  if (!isFinite(ratio) || ratio <= 0) return { text: "Unknown", cls: "imb-unknown" };
  if (ratio <= 1.5) return { text: "Balanced", cls: "imb-good" };
  if (ratio <= 3.0) return { text: "Moderate", cls: "imb-moderate" };
  return { text: "Severe", cls: "imb-severe" };
}

function formatShape(shape: number[]): string {
  if (shape.length !== 3) return "N/A";
  const [c, h, w] = shape;
  const colorMode = c === 1 ? "Grayscale" : c === 3 ? "RGB" : `${c}ch`;
  return `${h}×${w} ${colorMode}`;
}

export function DatasetInsight({
  datasetProfile,
  perClassTimeline,
  confusionTimeline,
  classNames,
  activeRun,
  offline,
}: Props) {
  const cc = useChartColors();
  /* ---- Section 1: Class Distribution ---- */
  const distBars = useMemo(() => {
    if (!datasetProfile) return [];
    return Object.entries(datasetProfile.class_distribution)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([classId, count]) => {
        const name = classNames[Number(classId)] ?? datasetProfile.class_names?.[Number(classId)] ?? `class_${classId}`;
        return { classId, name, count };
      });
  }, [datasetProfile, classNames]);

  const imb = datasetProfile
    ? imbalanceLabel(datasetProfile.imbalance_ratio)
    : { text: "N/A", cls: "" };

  /* ---- Section 2: Hardest Classes (from latest per-class data) ---- */
  const hardestClasses = useMemo(() => {
    const entries = Object.entries(perClassTimeline);
    if (entries.length === 0) return [];

    // Get latest accuracy for each class (from the trunk/selected path — latest entry)
    const latestPerClass: { classId: string; name: string; accuracy: number; support: number }[] = [];
    for (const [cls, rows] of entries) {
      if (rows.length === 0) continue;
      const last = rows[rows.length - 1];
      const name =
        classNames[Number(cls)] ??
        (datasetProfile?.class_names?.[Number(cls)]) ??
        `class_${cls}`;
      latestPerClass.push({
        classId: cls,
        name,
        accuracy: last.accuracy,
        support: last.support,
      });
    }

    // Compute mean accuracy
    const meanAcc =
      latestPerClass.length > 0
        ? latestPerClass.reduce((s, c) => s + c.accuracy, 0) / latestPerClass.length
        : 0;

    return latestPerClass
      .map((c) => ({ ...c, delta: c.accuracy - meanAcc }))
      .sort((a, b) => a.accuracy - b.accuracy)
      .slice(0, 5);
  }, [perClassTimeline, classNames, datasetProfile]);

  /* ---- Section 3: Confused Pairs (from latest confusion matrix) ---- */
  const latestConfusion = useMemo(() => {
    for (let i = confusionTimeline.length - 1; i >= 0; i--) {
      const entry = confusionTimeline[i];
      if (entry.matrix && entry.matrix.length > 0 && entry.matrix[0]?.length > 0) {
        return entry;
      }
    }
    return null;
  }, [confusionTimeline]);

  const chordClassNames = useMemo(() => {
    if (!latestConfusion) return [];
    return latestConfusion.labels.map((idx) => {
      return classNames[idx] ?? datasetProfile?.class_names?.[idx] ?? `${idx}`;
    });
  }, [latestConfusion, classNames, datasetProfile]);

  /* ---- Render ---- */
  return (
    <div className="tab-content">
      {/* Section 1: Class Distribution */}
      <div className="card">
        <h3>Class Distribution</h3>
        {datasetProfile ? (
          <>
            <div className="insight-summary-row">
              <span className="insight-chip">
                {datasetProfile.num_classes} classes
              </span>
              <span className="insight-chip">
                {datasetProfile.total_train_samples.toLocaleString()} train
              </span>
              <span className="insight-chip">
                {datasetProfile.total_val_samples.toLocaleString()} val
              </span>
              <span className="insight-chip">
                {formatShape(datasetProfile.image_shape)}
              </span>
              <span className={`insight-badge ${imb.cls}`}>
                Imbalance: {datasetProfile.imbalance_ratio.toFixed(2)}× — {imb.text}
              </span>
            </div>
            <div className="chart-box chart-box--dist">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distBars} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11, fill: cc.text }}
                    interval={0}
                    angle={distBars.length > 8 ? -35 : 0}
                    textAnchor={distBars.length > 8 ? "end" : "middle"}
                    height={distBars.length > 8 ? 70 : 40}
                  />
                  <YAxis tick={{ fontSize: 11, fill: cc.text }} />
                  <Tooltip
                    contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }}
                    formatter={(value: number) => [value.toLocaleString(), "samples"]}
                    labelStyle={{ fontWeight: 600 }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {distBars.map((_, i) => (
                      <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <p className="muted">
            No dataset profile available. Start training to collect class distribution data.
          </p>
        )}
      </div>

      {/* Section 2: Hardest Classes */}
      <div className="card">
        <h3>Hardest Classes</h3>
        <p className="muted" style={{ fontSize: 11, marginBottom: 8 }}>
          Top-5 lowest accuracy classes from the latest evaluation. Delta is vs. mean class accuracy.
        </p>
        {hardestClasses.length > 0 ? (
          <div className="hardest-classes-list">
            {hardestClasses.map((c, i) => (
              <div key={c.classId} className="hardest-class-row">
                <span className="hardest-rank">#{i + 1}</span>
                <span
                  className="hardest-dot"
                  style={{ background: BAR_COLORS[Number(c.classId) % BAR_COLORS.length] }}
                />
                <span className="hardest-name">{c.name}</span>
                <span className="hardest-bar-track">
                  <span
                    className="hardest-bar-fill"
                    style={{
                      width: `${Math.min(100, c.accuracy * 100)}%`,
                      background: c.accuracy < 0.5 ? "#dc2626" : c.accuracy < 0.75 ? "#d97706" : "#16a34a",
                    }}
                  />
                </span>
                <span className="hardest-acc">{(c.accuracy * 100).toFixed(1)}%</span>
                <span
                  className="hardest-delta"
                  style={{ color: c.delta < 0 ? "var(--red)" : "var(--green)" }}
                >
                  {c.delta >= 0 ? "+" : ""}
                  {(c.delta * 100).toFixed(1)}pp
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No per-class metrics yet. Data will appear after training starts.</p>
        )}
      </div>

      {/* Section 3: Most Confused Pairs */}
      <div className="card">
        <h3>Most Confused Pairs</h3>
        <p className="muted" style={{ fontSize: 11, marginBottom: 8 }}>
          Chord diagram showing misclassification flow between classes. Thicker ribbons = more confusion.
        </p>
        {latestConfusion ? (
          <ConfusionChord
            matrix={latestConfusion.matrix}
            classNames={chordClassNames}
          />
        ) : (
          <p className="muted">
            No confusion matrix data yet. Data will appear after the first evaluation epoch.
          </p>
        )}
      </div>

      {/* Section 4: Data Quality */}
      <DataQualitySection datasetProfile={datasetProfile} activeRun={activeRun} offline={offline} />
    </div>
  );
}


/* ================================================================
   Data Quality sub-component
   ================================================================ */

const SEVERITY_ICON: Record<string, string> = {
  critical: "\u26D4",  // ⛔
  warning: "\u26A0\uFE0F",  // ⚠️
  info: "\u2139\uFE0F",  // ℹ️
};

const SEVERITY_CLS: Record<string, string> = {
  critical: "dq-critical",
  warning: "dq-warning",
  info: "dq-info",
};

const WARNING_LABELS: Record<string, string> = {
  near_duplicates: "Near-Duplicate Images",
  zero_variance: "Zero-Variance Images",
  contains_nan: "NaN Values Detected",
  contains_inf: "Inf Values Detected",
  near_black: "Near-Black Images",
  near_white: "Near-White Images",
};

function DataQualitySection({
  datasetProfile,
  activeRun,
  offline,
}: {
  datasetProfile?: DatasetProfile;
  activeRun: string;
  offline: boolean;
}) {
  const dq = datasetProfile?.data_quality;
  const [expandedGroup, setExpandedGroup] = useState<number | null>(null);
  const [expandedWarning, setExpandedWarning] = useState<number | null>(null);

  if (!datasetProfile) {
    return (
      <div className="card">
        <h3>Data Quality</h3>
        <p className="muted">
          No dataset profile available. Start training to run quality analysis.
        </p>
      </div>
    );
  }

  if (!dq) {
    return (
      <div className="card">
        <h3>Data Quality</h3>
        <p className="muted">
          Quality analysis was not available for this run.
        </p>
      </div>
    );
  }

  const hasWarnings = dq.warnings.length > 0;
  const overallStatus = !hasWarnings
    ? "pass"
    : dq.warnings.some((w: DataQualityWarning) => w.severity === "critical")
      ? "critical"
      : "warn";

  const toggleGroup = (idx: number) => {
    setExpandedGroup(expandedGroup === idx ? null : idx);
  };

  return (
    <div className="card">
      <h3>Data Quality</h3>
      <p className="muted" style={{ fontSize: 11, marginBottom: 8 }}>
        Automated checks for duplicates, corrupt, and anomalous images.
        Scanned {dq.scanned_samples.toLocaleString()} training samples.
      </p>

      {/* Overall status badge */}
      <div className="insight-summary-row" style={{ marginBottom: 16 }}>
        <span className={`insight-badge ${
          overallStatus === "pass" ? "imb-good"
            : overallStatus === "critical" ? "imb-severe"
              : "imb-moderate"
        }`}>
          {overallStatus === "pass" ? "✓ All Checks Passed" :
            overallStatus === "critical" ? "⛔ Critical Issues Found" :
              "⚠ Warnings Found"}
        </span>
        <span className="insight-chip">
          {dq.scanned_samples.toLocaleString()} scanned
        </span>
        {dq.total_duplicate_pairs > 0 && (
          <span className="insight-chip">
            {dq.total_duplicate_pairs} duplicate pair(s)
          </span>
        )}
        {dq.total_flagged_images > 0 && (
          <span className="insight-chip">
            {dq.total_flagged_images} flagged image(s)
          </span>
        )}
      </div>

      {/* Warnings list */}
      {hasWarnings ? (
        <div className="dq-warnings-list">
          {dq.warnings.map((w: DataQualityWarning, i: number) => {
            const hasImgs = w.image_paths && w.image_paths.length > 0;
            const isWarnExpanded = expandedWarning === i;
            return (
              <div key={i} className={`dq-warning-row ${SEVERITY_CLS[w.severity] ?? ""} ${hasImgs ? "dq-warning-row--clickable" : ""}`}>
                <span className="dq-icon">{SEVERITY_ICON[w.severity] ?? "•"}</span>
                <div className="dq-warning-content">
                  <div
                    className="dq-warning-title"
                    onClick={hasImgs ? () => setExpandedWarning(isWarnExpanded ? null : i) : undefined}
                    style={hasImgs ? { cursor: "pointer" } : undefined}
                  >
                    {hasImgs && (
                      <span className="dq-dup-chevron" style={{ marginRight: 4 }}>
                        {isWarnExpanded ? "▾" : "▸"}
                      </span>
                    )}
                    {WARNING_LABELS[w.type] ?? w.type}
                    <span className="dq-warning-count">{w.count}</span>
                    {hasImgs && !isWarnExpanded && (
                      <span className="dq-dup-preview-hint" style={{ marginLeft: 8 }}>click to preview</span>
                    )}
                  </div>
                  <div className="dq-warning-message">{w.message}</div>
                  {w.indices && w.indices.length > 0 && (
                    <div className="dq-warning-indices">
                      Sample indices: {w.indices.slice(0, 10).join(", ")}
                      {w.indices.length > 10 && ` … +${w.indices.length - 10} more`}
                    </div>
                  )}
                  {isWarnExpanded && hasImgs && (
                    <div className="dq-dup-images" style={{ marginTop: 8, borderRadius: 4 }}>
                      {w.image_paths!.map((p, pi) => (
                        <div key={pi} className="dq-dup-thumb-wrapper">
                          <img
                            src={resolveArtifact(p, activeRun, offline)}
                            alt={`flagged-${w.type}-${pi}`}
                            className="dq-dup-thumb"
                            loading="lazy"
                          />
                          {w.indices && w.indices[pi] !== undefined && (
                            <span className="dq-dup-thumb-label">#{w.indices[pi]}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="dq-pass-message">
          <span style={{ fontSize: 20, marginRight: 8 }}>✓</span>
          {dq.summary}
        </div>
      )}

      {/* Duplicate groups detail with expandable image previews */}
      {dq.duplicate_groups.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h4 style={{ fontSize: 13, marginBottom: 8 }}>Duplicate Groups</h4>
          <p className="muted" style={{ fontSize: 11, marginBottom: 8 }}>
            Groups of images with near-identical perceptual hashes (dHash, Hamming distance ≤ 10).
            Consider deduplicating to avoid training bias and inflated validation metrics.
            {dq.duplicate_groups.some((g: DuplicateGroup) => g.image_paths?.length) &&
              " Click a group to preview the flagged images."}
          </p>
          <div className="dq-dup-groups">
            {dq.duplicate_groups.slice(0, 10).map((g: DuplicateGroup, i: number) => {
              const hasImages = g.image_paths && g.image_paths.length > 0;
              const isExpanded = expandedGroup === i;
              return (
                <div key={i} className={`dq-dup-group ${hasImages ? "dq-dup-group--clickable" : ""} ${isExpanded ? "dq-dup-group--expanded" : ""}`}>
                  <div
                    className="dq-dup-group-header"
                    onClick={hasImages ? () => toggleGroup(i) : undefined}
                    style={hasImages ? { cursor: "pointer" } : undefined}
                  >
                    {hasImages && (
                      <span className="dq-dup-chevron">{isExpanded ? "▾" : "▸"}</span>
                    )}
                    <span className="dq-dup-size">{g.size} images</span>
                    <span className="dq-dup-indices">
                      indices: {g.indices.slice(0, 8).join(", ")}
                      {g.indices.length > 8 && ` … +${g.indices.length - 8} more`}
                    </span>
                    {hasImages && !isExpanded && (
                      <span className="dq-dup-preview-hint">click to preview</span>
                    )}
                  </div>
                  {isExpanded && hasImages && (
                    <div className="dq-dup-images">
                      {g.image_paths!.map((p, pi) => (
                        <div key={pi} className="dq-dup-thumb-wrapper">
                          <img
                            src={resolveArtifact(p, activeRun, offline)}
                            alt={`dup-group-${i}-${pi}`}
                            className="dq-dup-thumb"
                            loading="lazy"
                          />
                          <span className="dq-dup-thumb-label">
                            #{g.indices[pi] !== undefined ? g.indices[pi] : pi}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
            {dq.duplicate_groups.length > 10 && (
              <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                … and {dq.duplicate_groups.length - 10} more group(s).
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
