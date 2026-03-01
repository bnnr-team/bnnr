import { useMemo } from "react";
import type { BranchNode, DecisionRecord, MetricPoint } from "../types";

interface Props {
  decisions: DecisionRecord[];
  timeline: MetricPoint[];
  branchNodes?: BranchNode[];
}

/**
 * Augmentation Impact Summary — shows the ROI of each BNNR decision.
 * For each iteration: winner, best epoch, Δ vs baseline, Δ vs previous step, cumulative gain.
 */
export function AugmentationImpact({ decisions, timeline, branchNodes }: Props) {
  const rows = useMemo(() => {
    if (decisions.length === 0) return [];

    // Build a lookup from branch_id to node for best_epoch info
    const nodeById: Record<string, BranchNode> = {};
    for (const n of branchNodes ?? []) {
      nodeById[n.id] = n;
    }

    // Get baseline accuracy (last baseline epoch)
    const baselineEntries = timeline.filter((r) => r.branch === "baseline");
    const baselineAcc = baselineEntries.length > 0
      ? baselineEntries[baselineEntries.length - 1].accuracy
      : 0;

    let prevAcc = baselineAcc;
    return decisions.map((d) => {
      const winnerMetrics =
        d.results[d.selected_branch_label] ??
        d.results[d.selected_branch_id] ?? {};
      const acc = winnerMetrics.accuracy ?? 0;
      const f1 = winnerMetrics.f1_macro ?? 0;
      const loss = winnerMetrics.loss ?? 0;
      const deltaVsBaseline = acc - baselineAcc;
      const deltaVsPrev = acc - prevAcc;
      const candidateCount = Object.keys(d.results).length;

      // Best epoch from node data
      const winnerNode = nodeById[d.selected_branch_id];
      const bestEpoch = winnerNode?.best_epoch ?? null;
      const totalEpochs = winnerNode?.total_epochs ?? null;

      // Find best and worst candidate
      let bestCand = "";
      let bestAcc = -1;
      let worstCand = "";
      let worstAcc = 2;
      for (const [name, m] of Object.entries(d.results)) {
        const a = m.accuracy ?? 0;
        if (a > bestAcc) { bestAcc = a; bestCand = name; }
        if (a < worstAcc) { worstAcc = a; worstCand = name; }
      }

      const row = {
        iteration: d.iteration,
        winner: d.selected_branch_label,
        accuracy: acc,
        f1,
        loss,
        deltaVsBaseline,
        deltaVsPrev,
        candidateCount,
        bestCand,
        bestAcc,
        worstCand,
        worstAcc,
        bestEpoch,
        totalEpochs,
      };
      prevAcc = acc;
      return row;
    });
  }, [decisions, timeline, branchNodes]);

  if (rows.length === 0) {
    return <p className="muted">Awaiting first augmentation decision. Data appears after candidates are evaluated and a winner is selected.</p>;
  }

  return (
    <div className="ledger-table-wrap">
      <table className="ledger-table impact-table">
        <thead>
          <tr>
            <th>Iter</th>
            <th>Winner</th>
            <th>Best Epoch</th>
            <th>Acc (%)</th>
            <th>Δ vs Baseline</th>
            <th>Δ vs Prev</th>
            <th>F1 (%)</th>
            <th>Loss</th>
            <th>Cands</th>
            <th>Best / Worst Candidate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.iteration}>
              <td>{r.iteration}</td>
              <td><strong>{r.winner}</strong></td>
              <td style={{ fontWeight: 600, color: "#7c3aed" }}>
                {r.bestEpoch != null ? `★ ${r.bestEpoch}/${r.totalEpochs ?? "?"}` : "—"}
              </td>
              <td>{(r.accuracy * 100).toFixed(1)}</td>
              <td style={{
                color: r.deltaVsBaseline >= 0 ? "var(--green)" : "var(--red)",
                fontWeight: 700,
              }}>
                {r.deltaVsBaseline >= 0 ? "+" : ""}{(r.deltaVsBaseline * 100).toFixed(2)}pp
              </td>
              <td style={{
                color: r.deltaVsPrev >= 0 ? "var(--green)" : "var(--red)",
                fontWeight: 600,
              }}>
                {r.deltaVsPrev >= 0 ? "+" : ""}{(r.deltaVsPrev * 100).toFixed(2)}pp
              </td>
              <td>{(r.f1 * 100).toFixed(1)}</td>
              <td>{r.loss.toFixed(3)}</td>
              <td>{r.candidateCount}</td>
              <td style={{ fontSize: 11 }}>
                <span style={{ color: "var(--green)" }}>▲ {r.bestCand} ({(r.bestAcc * 100).toFixed(1)}%)</span>
                {" / "}
                <span style={{ color: "var(--red)" }}>▼ {r.worstCand} ({(r.worstAcc * 100).toFixed(1)}%)</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
