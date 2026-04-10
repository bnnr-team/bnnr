import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MetricPoint } from "../types";
import { useChartColors } from "../ThemeContext";
import { primaryMetricValue, type RunTask } from "../taskMetrics";

interface Props {
  timeline: MetricPoint[];
  selectedPath?: string[];
  task?: RunTask;
}

const COLORS = {
  positive: "#16a34a",
  negative: "#dc2626",
  baseline: "#2563eb",
};

/**
 * Metric Gain Rate — shows the primary metric delta (pp) achieved in each
 * phase of training: baseline and each augmentation iteration.
 * Positive bars = improvement, negative = regression.
 */
export function AccuracyGainRate({ timeline, selectedPath, task = "classification" }: Props) {
  const cc = useChartColors();
  const metricLabel = task === "detection" ? "mAP@50" : task === "multilabel" ? "F1 (samples)" : "Accuracy";

  const bars = useMemo(() => {
    if (timeline.length === 0) return [];

    // Determine trunk branches
    const trunkBranches = new Set<string>(["baseline"]);
    for (const p of selectedPath ?? []) {
      const parts = p.split(":");
      if (parts.length > 1) trunkBranches.add(parts.slice(1).join(":"));
      else trunkBranches.add(p);
    }

    // Group trunk points by iteration
    const iterMap = new Map<number, MetricPoint[]>();
    for (const p of timeline) {
      if (!trunkBranches.has(p.branch)) continue;
      if (!iterMap.has(p.iteration)) iterMap.set(p.iteration, []);
      iterMap.get(p.iteration)!.push(p);
    }

    const iterations = Array.from(iterMap.keys()).sort((a, b) => a - b);
    const result: Array<{
      name: string;
      delta_pp: number;
      start_acc: number;
      end_acc: number;
      epochs: number;
      gain_per_epoch: number;
    }> = [];

    let prevEndAcc = 0;

    for (const iter of iterations) {
      const points = iterMap.get(iter)!;
      if (points.length === 0) continue;

      const startAcc = iter === 0 ? 0 : prevEndAcc;
      const endAcc = Math.max(...points.map((p) => primaryMetricValue(p, task)));
      const delta = endAcc - startAcc;
      const epochCount = points.length;
      const branch = points[0].branch;
      const label = iter === 0 ? "Baseline" : branch;

      result.push({
        name: label,
        delta_pp: Math.round(delta * 10000) / 100, // percentage points
        start_acc: Math.round(startAcc * 10000) / 100,
        end_acc: Math.round(endAcc * 10000) / 100,
        epochs: epochCount,
        gain_per_epoch: epochCount > 0 ? Math.round((delta / epochCount) * 10000) / 100 : 0,
      });

      prevEndAcc = endAcc;
    }

    return result;
  }, [timeline, selectedPath, task]);

  if (bars.length === 0) {
    return <p className="muted">No training data yet.</p>;
  }

  const totalGain = bars.reduce((sum, b) => sum + b.delta_pp, 0);

  return (
    <div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bars}>
            <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: cc.text }} interval={0} angle={bars.length > 4 ? -25 : 0} textAnchor={bars.length > 4 ? "end" : "middle"} height={bars.length > 4 ? 50 : 30} />
            <YAxis
              unit="pp"
              tick={{ fontSize: 11, fill: cc.text }}
              label={{ value: `Δ ${metricLabel} (pp)`, angle: -90, position: "insideLeft", fontSize: 11, fill: cc.text }}
            />
            <Tooltip
              contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }}
              formatter={(value: number, _name: string, props) => {
                const d = props.payload;
                if (!d) return [String(value), "Gain"];
                return [
                  `${value > 0 ? "+" : ""}${value.toFixed(2)}pp (${d.start_acc.toFixed(1)}% → ${d.end_acc.toFixed(1)}%)`,
                  "Gain",
                ];
              }}
            />
            <Legend />
            <ReferenceLine y={0} stroke={cc.text} strokeWidth={1} />
            <Bar dataKey="delta_pp" name={`${metricLabel} Gain (pp)`} radius={[4, 4, 0, 0]}>
              {bars.map((b, i) => (
                <Cell
                  key={`cell-${i}`}
                  fill={i === 0 ? COLORS.baseline : b.delta_pp >= 0 ? COLORS.positive : COLORS.negative}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p style={{ fontSize: 12, color: "var(--muted)", margin: "8px 0 4px" }}>
        Shows how much each training phase improved (or regressed) the primary metric.
      </p>
      <div style={{ fontSize: 12, color: "var(--muted)", display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
        <span>
          <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 2, backgroundColor: COLORS.baseline, marginRight: 4, verticalAlign: "middle" }} />
          Baseline
        </span>
        <span>
          <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 2, backgroundColor: COLORS.positive, marginRight: 4, verticalAlign: "middle" }} />
          Improvement
        </span>
        <span>
          <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 2, backgroundColor: COLORS.negative, marginRight: 4, verticalAlign: "middle" }} />
          Regression
        </span>
        <span style={{ marginLeft: "auto" }}>
          Cumulative: <strong style={{ color: totalGain >= 0 ? "var(--green)" : "var(--red)" }}>
            {totalGain >= 0 ? "+" : ""}{totalGain.toFixed(2)}pp
          </strong> across {bars.length} phase{bars.length !== 1 ? "s" : ""}
        </span>
      </div>
    </div>
  );
}
