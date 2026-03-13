import { useCallback, useMemo, useState } from "react";
import type { BranchNode, StatePayload } from "../types";
import { resolveArtifact } from "../hooks/useRunState";
import { useChartColors, useTheme } from "../ThemeContext";

/* ---- Lightbox for image zoom ---- */
function ImageLightbox({ src, label, onClose }: { src: string; label: string; onClose: () => void }) {
  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
        <button className="lightbox-close" onClick={onClose}>&times;</button>
        <img src={src} alt={label} />
        <div className="lightbox-label">{label}</div>
      </div>
    </div>
  );
}

interface Props {
  node: BranchNode;
  state: StatePayload;
  activeRun: string;
  offline: boolean;
  onClose: () => void;
}

export function NodeDetailPanel({ node, state, activeRun, offline, onClose }: Props) {
  const cc = useChartColors();
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [lightbox, setLightbox] = useState<{ src: string; label: string } | null>(null);
  const openLightbox = useCallback((src: string, label: string) => setLightbox({ src, label }), []);
  const branchData = state.branches[node.id];
  const metrics = branchData?.metrics;
  const decision = (state.decision_history ?? []).find(
    (d) => d.selected_branch_id === node.id,
  );
  const onBestPath = (state.selected_path ?? []).includes(node.id);
  const runTask = (state.task ?? "classification") as "classification" | "multilabel";
  const primaryMetricKey = "accuracy";

  // Find sample snapshots for this branch
  const snapshot = (state.sample_branch_snapshots ?? {})[node.id];
  const samplePairs = snapshot?.sample_pairs ?? [];

  // Per-class accuracy for this branch from the last epoch_end matching this branch
  const perClassForBranch = useMemo(() => {
    const branchLabel = node.label;
    const timeline = state.per_class_timeline ?? {};
    const result: Record<string, number> = {};
    for (const [cls, rows] of Object.entries(timeline)) {
      const matching = rows.filter((r) => r.branch === branchLabel);
      if (matching.length > 0) {
        result[cls] = matching[matching.length - 1].accuracy;
      }
    }
    return result;
  }, [node.label, state.per_class_timeline]);

  // Confusion matrix for this branch
  const confusionForBranch = useMemo(() => {
    const branchLabel = node.label;
    const entries = (state.confusion_timeline ?? []).filter(
      (c) => c.branch === branchLabel,
    );
    return entries.length > 0 ? entries[entries.length - 1] : null;
  }, [node.label, state.confusion_timeline]);

  return (
    <div className="detail-panel-overlay" onClick={onClose}>
      <div className="detail-panel" onClick={(e) => e.stopPropagation()}>
        <div className="detail-header">
          <h3>{node.label}</h3>
          <button className="close-btn" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="detail-badges">
          <span className={`tooltip-badge ${onBestPath ? "badge-selected" : "badge-evaluated"}`}>
            {onBestPath ? "SELECTED (best path)" : node.status ?? "candidate"}
          </span>
          {node.augmentation && <span className="tooltip-badge badge-default">aug: {node.augmentation}</span>}
          {node.iteration !== undefined && (
            <span className="tooltip-badge badge-default">iteration {node.iteration}</span>
          )}
          {node.best_epoch != null && (
            <span className="tooltip-badge badge-default">
              best @ epoch {node.best_epoch}/{node.total_epochs ?? "?"}
            </span>
          )}
        </div>

        {/* Metrics table */}
        {metrics && (
          <div className="detail-section">
            <h4>Metrics</h4>
            <table className="detail-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>
                      {k === "loss"
                        ? v.toFixed(4)
                        : `${(v * 100).toFixed(2)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Decision context */}
        {decision && (
          <div className="detail-section">
            <h4>Decision Context</h4>
            <p className="muted">
              {decision.decision_reason
                || "Best candidate selected by accuracy."}
            </p>
            {decision.baseline_accuracy !== null && metrics?.[primaryMetricKey] !== undefined && (
              <p>
                Delta vs baseline:{" "}
                <strong>
                  {(((metrics[primaryMetricKey] ?? 0) - decision.baseline_accuracy) * 100).toFixed(2)} pp
                </strong>
              </p>
            )}
          </div>
        )}

        {/* Per-class accuracy */}
        {Object.keys(perClassForBranch).length > 0 && (
          <div className="detail-section">
            <h4>{runTask === "multilabel" ? "Per-Label F1" : "Per-Class Accuracy"}</h4>
            <div className="class-bars">
              {Object.entries(perClassForBranch)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([cls, acc]) => (
                  <div key={cls} className="class-bar-row">
                    <span className="class-label">Class {cls}</span>
                    <div className="class-bar-track">
                      <div
                        className="class-bar-fill"
                        style={{ width: `${Math.min(100, acc * 100)}%` }}
                      />
                    </div>
                    <span className="class-bar-value">{(acc * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Augmentation sample previews with zoom */}
        {samplePairs.length > 0 && (
          <div className="detail-section">
            <h4>Augmentation Samples{node.augmentation ? ` — ${node.augmentation}` : ""}</h4>
            {samplePairs.slice(0, 6).map((pair, i) => {
              const origSrc = resolveArtifact(pair[0], activeRun, offline);
              const augSrc = resolveArtifact(pair[1], activeRun, offline);
              return (
                <div key={i} className="detail-sample-card" style={{ marginBottom: 8 }}>
                  <div className="detail-sample-grid">
                    <div>
                      <div className="detail-sample-label">Original</div>
                      <div
                        className="detail-sample-img-wrap"
                        onClick={() => openLightbox(origSrc, `Original — sample ${i + 1}`)}
                      >
                        <img src={origSrc} alt={`original-${i}`} />
                        <span className="detail-sample-zoom-hint">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
                          Enlarge
                        </span>
                      </div>
                    </div>
                    <div>
                      <div className="detail-sample-label">Augmented{node.augmentation ? ` (${node.augmentation})` : ""}</div>
                      <div
                        className="detail-sample-img-wrap"
                        onClick={() => openLightbox(augSrc, `Augmented${node.augmentation ? ` — ${node.augmentation}` : ""} — sample ${i + 1}`)}
                      >
                        <img src={augSrc} alt={`augmented-${i}`} />
                        <span className="detail-sample-zoom-hint">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
                          Enlarge
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Mini confusion matrix */}
        {confusionForBranch && (
          <div className="detail-section">
            <h4>Confusion Matrix</h4>
            <div className="mini-matrix">
              {confusionForBranch.matrix.map((row, ri) => {
                const maxVal = Math.max(1, ...confusionForBranch.matrix.flat());
                return (
                  <div key={ri} className="matrix-row">
                    {row.map((val, ci) => {
                      const intensity = val / maxVal;
                      const bg =
                        ri === ci
                          ? `rgba(22, 163, 74, ${0.15 + 0.85 * intensity})`
                          : `rgba(37, 99, 235, ${0.10 + 0.70 * intensity})`;
                      const textColor =
                        intensity > 0.4
                          ? cc.matrixTextLight
                          : isDark
                            ? "#e2e8f0"
                            : "#1e293b";
                      const textShadow =
                        intensity > 0.4
                          ? "0 1px 2px rgba(0,0,0,0.5)"
                          : isDark
                            ? "0 1px 2px rgba(0,0,0,0.8)"
                            : "none";
                      return (
                        <span
                          key={ci}
                          className="matrix-cell"
                          style={{
                            backgroundColor: bg,
                            color: textColor,
                            textShadow,
                            border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
                          }}
                          title={`True: ${ri}, Pred: ${ci}, Count: ${val}`}
                        >
                          {val}
                        </span>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Lightbox zoom overlay */}
      {lightbox && (
        <ImageLightbox
          src={lightbox.src}
          label={lightbox.label}
          onClose={() => setLightbox(null)}
        />
      )}
    </div>
  );
}
