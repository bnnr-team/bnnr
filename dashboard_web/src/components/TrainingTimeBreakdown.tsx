import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DecisionRecord, MetricPoint } from "../types";
import { useChartColors } from "../ThemeContext";

interface Props {
  decisions: DecisionRecord[];
  timeline: MetricPoint[];
}

const COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2",
  "#d97706", "#4f46e5", "#0f766e", "#be185d", "#65a30d",
];

/**
 * Training Time Breakdown — shows how many epochs were spent on
 * baseline training vs each augmentation decision iteration.
 */
export function TrainingTimeBreakdown({ decisions, timeline }: Props) {
  const cc = useChartColors();
  const bars = useMemo(() => {
    // Count epochs per phase
    const phases: Array<{ name: string; epochs: number; accuracy: number }> = [];

    // Baseline epochs
    const baselineEpochs = timeline.filter((p) => p.branch === "baseline").length;
    // Use the correct primary metric based on available data
    const det = timeline.some((p) => (p.map_50 ?? 0) > 0) && !timeline.some((p) => p.accuracy > 0);
    const baselineAcc = timeline
      .filter((p) => p.branch === "baseline")
      .reduce((max, p) => Math.max(max, det ? (p.map_50 ?? 0) : p.accuracy), 0);
    if (baselineEpochs > 0) {
      phases.push({ name: "Baseline", epochs: baselineEpochs, accuracy: baselineAcc });
    }

    // Per iteration
    for (const d of decisions) {
      const iterEpochs = timeline.filter((p) => p.iteration === d.iteration).length;
      const winnerMetrics = d.results[d.selected_branch_label] ?? d.results[d.selected_branch_id] ?? {};
      phases.push({
        name: `I${d.iteration}: ${d.selected_branch_label}`,
        epochs: iterEpochs,
        accuracy: det ? (winnerMetrics.map_50 ?? 0) : (winnerMetrics.accuracy ?? 0),
      });
    }

    return phases;
  }, [decisions, timeline]);

  if (bars.length === 0) {
    return <p className="muted">No training data yet.</p>;
  }

  const totalEpochs = bars.reduce((sum, b) => sum + b.epochs, 0);

  return (
    <div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bars}>
            <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: cc.text }} interval={0} angle={bars.length > 4 ? -25 : 0} textAnchor={bars.length > 4 ? "end" : "middle"} height={bars.length > 4 ? 50 : 30} />
            <YAxis label={{ value: "Epochs", angle: -90, position: "insideLeft", fontSize: 11, fill: cc.text }} tick={{ fontSize: 11, fill: cc.text }} />
            <Tooltip
              contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }}
              formatter={(value: number, name: string) => {
                if (name === "epochs") return [`${value} epochs`, "Epochs"];
                return [value, name];
              }}
            />
            <Legend />
            <Bar dataKey="epochs" name="Epochs" radius={[4, 4, 0, 0]}>
              {bars.map((_, i) => (
                <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 8 }}>
        Total: <strong>{totalEpochs}</strong> epochs across{" "}
        <strong>{bars.length}</strong> phases
      </div>
    </div>
  );
}
