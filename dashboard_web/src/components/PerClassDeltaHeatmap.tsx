import { useMemo } from "react";
import type { DecisionRecord, PerClassRow } from "../types";
import { useChartColors, useTheme } from "../ThemeContext";

interface Props {
  decisions: DecisionRecord[];
  perClassTimeline: Record<string, PerClassRow[]>;
}

/**
 * Per-Class Δ Heatmap — shows accuracy change per class after each decision.
 * Rows = classes, Columns = decisions/iterations.
 * Green = improvement, Red = regression.
 */
export function PerClassDeltaHeatmap({ decisions, perClassTimeline }: Props) {
  const cc = useChartColors();
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const { classes, columns, heatData, maxDelta } = useMemo(() => {
    const classSet = new Set<string>();
    Object.keys(perClassTimeline).forEach((k) => classSet.add(k));
    const sortedClasses = Array.from(classSet).sort();
    if (sortedClasses.length === 0 || decisions.length === 0) {
      return { classes: sortedClasses, columns: [] as string[], heatData: {} as Record<string, Record<string, number | null>>, maxDelta: 0 };
    }

    // For each class, for each iteration, compute the delta (accuracy at decision vs before).
    // We look at the last per_class entry for each branch.
    const data: Record<string, Record<string, number | null>> = {};
    const cols: string[] = [];
    let md = 0;

    // Map class -> acc after baseline
    const baselineAccByClass: Record<string, number> = {};
    for (const cls of sortedClasses) {
      const rows = perClassTimeline[cls] ?? [];
      const baselineRows = rows.filter((r) => r.branch === "baseline");
      if (baselineRows.length > 0) {
        baselineAccByClass[cls] = baselineRows[baselineRows.length - 1].accuracy;
      }
    }

    let prevAccByClass = { ...baselineAccByClass };

    for (const d of decisions) {
      const colName = `I${d.iteration}: ${d.selected_branch_label}`;
      cols.push(colName);

      for (const cls of sortedClasses) {
        const rows = perClassTimeline[cls] ?? [];
        // Find accuracy for the winner branch in this iteration
        const winnerRows = rows.filter(
          (r) => r.iteration === d.iteration &&
            (r.branch === d.selected_branch_label || r.branch === d.selected_branch_id),
        );
        if (winnerRows.length > 0) {
          const acc = winnerRows[winnerRows.length - 1].accuracy;
          const prev = prevAccByClass[cls] ?? 0;
          const delta = acc - prev;
          if (!data[cls]) data[cls] = {};
          data[cls][colName] = delta;
          md = Math.max(md, Math.abs(delta));
          prevAccByClass[cls] = acc;
        } else {
          if (!data[cls]) data[cls] = {};
          data[cls][colName] = null;
        }
      }
    }

    return { classes: sortedClasses, columns: cols, heatData: data, maxDelta: md };
  }, [decisions, perClassTimeline]);

  if (classes.length === 0 || columns.length === 0) {
    return <p className="muted">No per-class data available yet.</p>;
  }

  const cellColor = (val: number | null) => {
    if (val === null) return cc.neutralBg;
    if (maxDelta === 0) return cc.neutralBg;
    const norm = Math.min(Math.abs(val) / Math.max(maxDelta, 0.01), 1);
    const intensity = Math.round(norm * 200);
    if (val > 0.001) return `rgba(22, 163, 74, ${0.15 + norm * 0.65})`; // green
    if (val < -0.001) return `rgba(220, 38, 38, ${0.15 + norm * 0.65})`; // red
    return `rgba(148, 163, 184, ${0.1 + intensity * 0.001})`; // neutral
  };

  return (
    <div className="ledger-table-wrap">
      <table className="ledger-table heatmap-table">
        <thead>
          <tr>
            <th>Class</th>
            {columns.map((c) => (
              <th key={c} style={{ writingMode: "vertical-lr", textOrientation: "mixed", fontSize: 10, maxWidth: 48, minWidth: 36 }}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {classes.map((cls) => (
            <tr key={cls}>
              <td style={{ fontWeight: 600, fontSize: 12 }}>{cls}</td>
              {columns.map((col) => {
                const val = heatData[cls]?.[col] ?? null;
                return (
                  <td
                    key={col}
                    style={{
                      backgroundColor: cellColor(val),
                      textAlign: "center",
                      fontSize: 11,
                      fontWeight: 600,
                      color: val !== null && Math.abs(val) > 0.001
                        ? (isDark ? "#e2e8f0" : cc.matrixTextDark)
                        : cc.text,
                      textShadow: val !== null && Math.abs(val) > 0.001 && isDark
                        ? "0 1px 2px rgba(0,0,0,0.8)"
                        : "none",
                    }}
                    title={val !== null ? `${cls}: ${val > 0 ? "+" : ""}${(val * 100).toFixed(1)}pp` : "N/A"}
                  >
                    {val !== null ? `${val > 0 ? "+" : ""}${(val * 100).toFixed(1)}` : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
