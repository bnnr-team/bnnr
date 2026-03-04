import { useMemo } from "react";
import type { DecisionRecord } from "../types";

interface Props {
  decisions: DecisionRecord[];
  selectedPath?: string[];
}

/**
 * Augmentation Stack — shows the cumulative augmentation pipeline
 * that was built up through BNNR decisions.
 * Each decision adds a new augmentation layer to the stack.
 */
export function AugmentationStack({ decisions, selectedPath }: Props) {
  const stack = useMemo(() => {
    const layers: Array<{
      iteration: number;
      augmentation: string;
      accuracy: number;
    }> = [];

    for (const d of decisions) {
      const winnerMetrics =
        d.results[d.selected_branch_label] ??
        d.results[d.selected_branch_id] ?? {};
      layers.push({
        iteration: d.iteration,
        augmentation: d.selected_branch_label,
        accuracy: winnerMetrics.accuracy ?? 0,
      });
    }
    return layers;
  }, [decisions]);

  if (stack.length === 0) {
    return <p className="muted">No augmentations selected yet. Pipeline builds as decisions are made.</p>;
  }

  return (
    <div className="aug-stack">
      {/* Visual pipeline */}
      <div className="pipeline-vis">
        <div className="pipeline-node baseline-node">
          <div className="pipeline-label">Input Data</div>
          <div className="pipeline-sub">Raw images</div>
        </div>
        <div className="pipeline-arrow">→</div>
        {stack.map((layer, i) => (
          <div key={layer.iteration} className="pipeline-segment">
            <div className={`pipeline-node aug-node ${i === stack.length - 1 ? "latest" : ""}`}>
              <div className="pipeline-label">{layer.augmentation}</div>
              <div className="pipeline-sub">
                Iter {layer.iteration} · {(layer.accuracy * 100).toFixed(1)}%
              </div>
            </div>
            {i < stack.length - 1 && <div className="pipeline-arrow">→</div>}
          </div>
        ))}
        <div className="pipeline-arrow">→</div>
        <div className="pipeline-node model-node">
          <div className="pipeline-label">Model</div>
          <div className="pipeline-sub">
            {stack.length > 0 ? `${(stack[stack.length - 1].accuracy * 100).toFixed(1)}% acc` : "—"}
          </div>
        </div>
      </div>

      {/* Text summary */}
      <div className="pipeline-summary">
        <strong>Active pipeline:</strong>{" "}
        {stack.map((l) => l.augmentation).join(" → ")}
        {" "}({stack.length} augmentation{stack.length !== 1 ? "s" : ""})
      </div>
    </div>
  );
}
