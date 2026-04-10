import { useMemo, useState } from "react";
import type { PerClassRow, XAIInsightEntry, XAIDiagnosis } from "../types";
import type { RunTask } from "../taskMetrics";

interface Props {
  perClassTimeline: Record<string, PerClassRow[]>;
  xaiInsightsTimeline?: XAIInsightEntry[];
  classNames?: string[];
  task?: RunTask;
}

/* ── Quality breakdown bar component ── */
const BREAKDOWN_LABELS: Record<string, { label: string; color: string }> = {
  accuracy: { label: "Accuracy", color: "#4caf50" },
  focus: { label: "Focus", color: "#2196f3" },
  coverage: { label: "Coverage", color: "#ff9800" },
  coherence: { label: "Coherence", color: "#9c27b0" },
  edge: { label: "Edge", color: "#f44336" },
  consistency: { label: "Consistency", color: "#00bcd4" },
};

function QualityBreakdown({
  breakdown,
  totalScore,
}: {
  breakdown: Record<string, number>;
  totalScore: number;
}) {
  const keys = Object.keys(BREAKDOWN_LABELS).filter((k) => k in breakdown);
  if (keys.length === 0)
    return (
      <span
        className="xai-quality-badge"
        title="XAI Quality Score"
      >
        Q: {(totalScore * 100).toFixed(0)}%
      </span>
    );
  return (
    <div className="xai-quality-breakdown" title={`Quality: ${(totalScore * 100).toFixed(0)}%`}>
      <span className="xai-quality-breakdown-label">
        Q: {(totalScore * 100).toFixed(0)}%
      </span>
      <div className="xai-quality-bar">
        {keys.map((k) => {
          const meta = BREAKDOWN_LABELS[k];
          const val = breakdown[k] ?? 0;
          return (
            <div
              key={k}
              className="xai-quality-segment"
              style={{
                flex: val,
                backgroundColor: meta.color,
                opacity: 0.35 + val * 0.65,
              }}
              title={`${meta.label}: ${(val * 100).toFixed(0)}%`}
            />
          );
        })}
      </div>
      <div className="xai-quality-legend">
        {keys.map((k) => {
          const meta = BREAKDOWN_LABELS[k];
          const val = breakdown[k] ?? 0;
          return (
            <span key={k} className="xai-quality-legend-item">
              <span
                className="xai-quality-legend-dot"
                style={{ backgroundColor: meta.color }}
              />
              {meta.label} {(val * 100).toFixed(0)}%
            </span>
          );
        })}
      </div>
    </div>
  );
}

/* ── Baseline delta chips ── */
function BaselineDeltaChips({ delta }: { delta: Record<string, number> }) {
  const entries = Object.entries(delta).filter(
    ([, v]) => Math.abs(v) > 0.01,
  );
  if (entries.length === 0) return null;

  const labelMap: Record<string, string> = {
    gini: "Focus",
    coverage_pp: "Coverage",
    coherence: "Coherence",
    edge_ratio: "Edge",
    entropy: "Entropy",
  };

  return (
    <div className="xai-baseline-delta">
      <span className="xai-confusion-label">vs. Baseline:</span>
      {entries.map(([key, val]) => {
        const positive =
          key === "entropy" || key === "edge_ratio" ? val < 0 : val > 0;
        const suffix = key === "coverage_pp" ? "pp" : "";
        const display =
          key === "coverage_pp" || key === "entropy"
            ? `${val > 0 ? "+" : ""}${val.toFixed(0)}${suffix}`
            : `${val > 0 ? "+" : ""}${val.toFixed(3)}`;
        return (
          <span
            key={key}
            className={`xai-delta-chip ${positive ? "xai-delta-positive" : "xai-delta-negative"}`}
          >
            {labelMap[key] ?? key} {display}
          </span>
        );
      })}
    </div>
  );
}

/**
 * XAI Insights panel — shows human-readable per-class explanations
 * generated from saliency map analysis, enriched with confusion-matrix
 * data, severity badges, quality scores, and trend indicators.
 */
export function XAIInsights({
  perClassTimeline,
  xaiInsightsTimeline,
  classNames,
  task = "classification",
}: Props) {
  const classKeys = useMemo(
    () => Object.keys(perClassTimeline).sort((a, b) => Number(a) - Number(b)),
    [perClassTimeline],
  );

  /* Gather available checkpoints from xai_insights_timeline */
  const checkpoints = useMemo(() => {
    if (!xaiInsightsTimeline || xaiInsightsTimeline.length === 0) return [];
    return xaiInsightsTimeline.map((e, idx) => ({
      idx,
      label:
        e.branch === "baseline"
          ? `Baseline · e${e.epoch}`
          : `Iter ${e.iteration} · ${e.branch}`,
      ...e,
    }));
  }, [xaiInsightsTimeline]);

  const [selectedCheckpointIdx, setSelectedCheckpointIdx] = useState<number>(
    () => (checkpoints.length > 0 ? checkpoints.length - 1 : -1),
  );

  /* Also gather insights from per_class_timeline as fallback */
  const fallbackInsights = useMemo(() => {
    const result: Record<
      string,
      {
        insight: string;
        branch: string;
        epoch: number;
        iteration: number;
        severity?: string;
        quality_score?: number;
        trend?: string;
        confused_with?: { class: string; count: number }[];
        short_text?: string;
        quality_breakdown?: Record<string, number>;
        augmentation_impact?: string;
        baseline_delta?: Record<string, number>;
      }[]
    > = {};
    for (const [cls, rows] of Object.entries(perClassTimeline)) {
      const withInsight = rows.filter((r) => r.xai_insight);
      if (withInsight.length > 0) {
        result[cls] = withInsight.map((r) => ({
          insight: r.xai_insight!,
          branch: r.branch,
          epoch: r.epoch,
          iteration: r.iteration,
          severity: r.severity,
          quality_score: r.quality_score,
          trend: r.trend,
          confused_with: r.confused_with,
          short_text: r.short_text,
          quality_breakdown: r.quality_breakdown,
          augmentation_impact: r.augmentation_impact,
          baseline_delta: r.baseline_delta,
        }));
      }
    }
    return result;
  }, [perClassTimeline]);

  type InsightItem = {
    classId: string;
    className: string;
    text: string;
    severity?: string;
    quality_score?: number;
    trend?: string;
    confused_with?: { class: string; count: number }[];
    short_text?: string;
    quality_breakdown?: Record<string, number>;
    augmentation_impact?: string;
    baseline_delta?: Record<string, number>;
    ap_50?: number;
    detections?: number;
    mean_confidence?: number;
  };

  /* Active insights to display */
  const activeInsights: InsightItem[] = useMemo(() => {
    // Primary source: dedicated xai_insights_timeline
    if (checkpoints.length > 0 && selectedCheckpointIdx >= 0) {
      const cp = checkpoints[selectedCheckpointIdx];
      if (cp && cp.insights) {
        const diagnoses: Record<string, XAIDiagnosis> | undefined =
          cp.diagnoses;
        return Object.entries(cp.insights)
          .sort(([a], [b]) => Number(a) - Number(b))
          .map(([cls, text]) => {
            const diag = diagnoses?.[cls];
            return {
              classId: cls,
              className:
                classNames && Number(cls) < classNames.length
                  ? classNames[Number(cls)]
                  : `Class ${cls}`,
              text: diag?.text ?? text,
              severity: diag?.severity,
              quality_score: diag?.quality_score,
              trend: diag?.trend,
              confused_with: diag?.confused_with,
              short_text: diag?.short_text,
              quality_breakdown: diag?.quality_breakdown,
              augmentation_impact: diag?.augmentation_impact,
              baseline_delta: diag?.baseline_delta,
              mean_confidence: diag?.mean_confidence,
            };
          });
      }
    }
    // Fallback: latest from per_class_timeline
    const entries: InsightItem[] = [];
    for (const cls of classKeys) {
      const fb = fallbackInsights[cls];
      if (fb && fb.length > 0) {
        const last = fb[fb.length - 1];
        entries.push({
          classId: cls,
          className:
            classNames && Number(cls) < classNames.length
              ? classNames[Number(cls)]
              : `Class ${cls}`,
          text: last.insight,
          severity: last.severity,
          quality_score: last.quality_score,
          trend: last.trend,
          confused_with: last.confused_with,
          short_text: last.short_text,
          quality_breakdown: last.quality_breakdown,
          augmentation_impact: last.augmentation_impact,
          baseline_delta: last.baseline_delta,
        });
      }
    }
    return entries;
  }, [
    checkpoints,
    selectedCheckpointIdx,
    fallbackInsights,
    classKeys,
    classNames,
  ]);

  // Sync selected to latest if it's out of range
  if (
    checkpoints.length > 0 &&
    (selectedCheckpointIdx < 0 || selectedCheckpointIdx >= checkpoints.length)
  ) {
    // Will be fixed on next render
  }

  const hasData = activeInsights.length > 0;

  if (!hasData && classKeys.length === 0) {
    return <p className="muted">No XAI insight data available yet.</p>;
  }

  return (
    <div>
      {/* Checkpoint selector */}
      {checkpoints.length > 1 && (
        <div className="xai-insight-controls">
          <label>
            Checkpoint:
            <select
              value={selectedCheckpointIdx}
              onChange={(e) =>
                setSelectedCheckpointIdx(Number(e.target.value))
              }
            >
              {checkpoints.map((cp) => (
                <option key={cp.idx} value={cp.idx}>
                  {cp.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      {!hasData && (
        <div className="xai-insight-empty">
          <div style={{ fontSize: 28, marginBottom: 8 }}>🧠</div>
          <p>
            XAI insights will appear here once saliency maps have been
            generated. Enable XAI in your config and wait for the first
            checkpoint.
          </p>
        </div>
      )}

      {hasData && (
        <div className="xai-insight-list">
          {activeInsights.map((item) => {
            // Parse quick metadata from text for visual badges (legacy)
            const isScattered = item.text.includes("broadly scattered");
            const isFocused =
              item.text.includes("sharply focused") ||
              item.text.includes("is focused");
            const hasWarning =
              item.text.includes("struggles") ||
              item.text.includes("failed on all") ||
              item.text.includes("irrelevant");
            const hasSuggestion =
              item.text.includes("Suggestion:") ||
              item.text.includes("Suggestions:");

            // Structured severity (enriched)
            const severity = item.severity || (hasWarning ? "critical" : isFocused ? "ok" : "");
            const qualityScore = item.quality_score;
            const trend = item.trend;
            const confusedWith = item.confused_with;
            const qualityBreakdown = item.quality_breakdown;
            const augImpact = item.augmentation_impact;
            const baselineDelta = item.baseline_delta;

            const severityBorder =
              severity === "critical"
                ? "xai-card-critical"
                : severity === "warning"
                  ? "xai-card-warning"
                  : severity === "ok"
                    ? "xai-card-ok"
                    : "";

            // Extract suggestion text from the full diagnosis
            let suggestionText = "";
            const sugIdx = item.text.indexOf("Suggestions:");
            if (sugIdx >= 0) {
              suggestionText = item.text.substring(sugIdx);
            } else {
              const sugIdx2 = item.text.indexOf("Suggestion:");
              if (sugIdx2 >= 0) {
                suggestionText = item.text.substring(sugIdx2);
              }
            }

            // Build main text without the suggestion tail
            const mainText = suggestionText
              ? item.text.substring(0, item.text.indexOf(suggestionText)).trim()
              : item.text;

            return (
              <div
                key={item.classId}
                className={`xai-insight-card ${severityBorder}`}
              >
                <div className="xai-insight-header">
                  <span className="xai-insight-class">{item.className}</span>
                  <span className="xai-insight-id">#{item.classId}</span>

                  {/* Severity badge */}
                  {severity === "ok" && (
                    <span className="tooltip-badge badge-selected">OK</span>
                  )}
                  {severity === "warning" && (
                    <span className="tooltip-badge badge-evaluated">WARNING</span>
                  )}
                  {severity === "critical" && (
                    <span className="tooltip-badge badge-rejected">CRITICAL</span>
                  )}

                  {/* Legacy focus badges (when no severity data) */}
                  {!severity && isFocused && (
                    <span className="tooltip-badge badge-selected">FOCUSED</span>
                  )}
                  {!severity && isScattered && (
                    <span className="tooltip-badge badge-evaluated">SCATTERED</span>
                  )}

                  {/* Trend indicator */}
                  {trend === "improving" && (
                    <span className="xai-trend xai-trend-improving" title="Attention improving">▲</span>
                  )}
                  {trend === "declining" && (
                    <span className="xai-trend xai-trend-declining" title="Attention declining">▼</span>
                  )}
                  {trend === "stable" && (
                    <span className="xai-trend xai-trend-stable" title="Stable">●</span>
                  )}

                  {/* Quality score — breakdown bar or simple badge */}
                  {qualityScore !== undefined &&
                    qualityScore > 0 &&
                    (qualityBreakdown &&
                    Object.keys(qualityBreakdown).length > 0 ? (
                      <QualityBreakdown
                        breakdown={qualityBreakdown}
                        totalScore={qualityScore}
                      />
                    ) : (
                      <span
                        className="xai-quality-badge"
                        title="XAI Quality Score"
                      >
                        Q: {(qualityScore * 100).toFixed(0)}%
                      </span>
                    ))}
                </div>

                <p className="xai-insight-text">{mainText}</p>

                {item.mean_confidence !== undefined && (
                  <div className="xai-confusion-pairs">
                    <span className="xai-confusion-chip">Mean conf {(item.mean_confidence * 100).toFixed(1)}%</span>
                  </div>
                )}

                {/* Augmentation impact */}
                {augImpact && augImpact.length > 0 && (
                  <div className="xai-aug-impact">
                    <span className="xai-aug-impact-icon">⚡</span>
                    {augImpact}
                  </div>
                )}

                {/* Baseline delta chips */}
                {baselineDelta &&
                  Object.keys(baselineDelta).length > 0 && (
                    <BaselineDeltaChips delta={baselineDelta} />
                  )}

                {/* Confusion pairs */}
                {confusedWith && confusedWith.length > 0 && (
                  <div className="xai-confusion-pairs">
                    <span className="xai-confusion-label">Confused with:</span>
                    {confusedWith.map((c) => (
                      <span key={c.class} className="xai-confusion-chip">
                        {c.class}
                        <span className="xai-confusion-count">{c.count}×</span>
                      </span>
                    ))}
                  </div>
                )}

                {hasSuggestion && suggestionText && (
                  <div className="xai-insight-suggestion">
                    💡 {suggestionText}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
