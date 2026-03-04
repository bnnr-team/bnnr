import type { DecisionRecord } from "../types";

interface Props {
  decisions: DecisionRecord[];
}

export function DecisionLedger({ decisions }: Props) {
  if (decisions.length === 0) {
    return <p className="muted">No branch decisions recorded yet.</p>;
  }

  return (
    <div className="ledger-table-wrap">
      <table className="ledger-table">
        <thead>
          <tr>
            <th>Iter</th>
            <th>Winner</th>
            <th>Acc (%)</th>
            <th>Delta (pp)</th>
            <th>Candidates</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((d) => {
            const winnerMetrics =
              d.results[d.selected_branch_label] ??
              d.results[d.selected_branch_id] ??
              {};
            const acc = winnerMetrics.accuracy;
            const baseline = d.baseline_accuracy;
            const delta =
              acc !== undefined && baseline !== null
                ? (acc - baseline) * 100
                : null;
            const candidateCount = Object.keys(d.results).length;
            return (
              <tr key={d.iteration}>
                <td>{d.iteration}</td>
                <td>
                  <strong>{d.selected_branch_label}</strong>
                </td>
                <td>{acc !== undefined ? (acc * 100).toFixed(1) : "—"}</td>
                <td
                  style={{
                    color:
                      delta !== null
                        ? delta >= 0
                          ? "var(--green)"
                          : "var(--red)"
                        : undefined,
                    fontWeight: 600,
                  }}
                >
                  {delta !== null
                    ? `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`
                    : "—"}
                </td>
                <td>{candidateCount} evaluated</td>
                <td className="muted" style={{ maxWidth: 260, fontSize: 12 }}>
                  {d.decision_reason || "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
