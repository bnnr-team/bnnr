import type { BranchNode, DecisionRecord } from "../types";

interface Props {
  node: BranchNode;
  metrics?: Record<string, number>;
  position: { x: number; y: number };
  onBestPath: boolean;
  decision?: DecisionRecord;
}

export function NodeTooltip({ node, metrics, position, onBestPath, decision }: Props) {
  const acc = metrics?.accuracy;
  const f1 = metrics?.f1_macro;
  const loss = metrics?.loss;

  // compute delta vs baseline from decision
  let delta: string | null = null;
  if (decision && typeof decision.baseline_accuracy === "number" && acc !== undefined) {
    const d = (acc - decision.baseline_accuracy) * 100;
    delta = `${d >= 0 ? "+" : ""}${d.toFixed(2)} pp`;
  }

  return (
    <div
      className="node-tooltip"
      style={{
        left: position.x + 12,
        top: position.y - 10,
      }}
    >
      <div className="tooltip-header">
        <strong>{node.label}</strong>
        <span className={`tooltip-badge ${onBestPath ? "badge-selected" : node.status === "rejected" ? "badge-rejected" : node.status === "evaluated" ? "badge-evaluated" : "badge-default"}`}>
          {onBestPath ? "selected" : node.status === "rejected" ? "no improvement" : node.status ?? "candidate"}
        </span>
      </div>
      {node.augmentation && (
        <div className="tooltip-row">aug: {node.augmentation}</div>
      )}
      {metrics && (
        <div className="tooltip-metrics">
          {acc !== undefined && <span>Acc: {(acc * 100).toFixed(1)}%</span>}
          {f1 !== undefined && <span>F1: {(f1 * 100).toFixed(1)}%</span>}
          {loss !== undefined && <span>Loss: {loss.toFixed(3)}</span>}
        </div>
      )}
      {node.best_epoch != null && (
        <div className="tooltip-row" style={{ marginTop: 4, fontWeight: 600 }}>
          Best epoch: {node.best_epoch}/{node.total_epochs ?? "?"}
        </div>
      )}
      {delta && <div className="tooltip-delta">{delta} vs baseline</div>}
    </div>
  );
}
