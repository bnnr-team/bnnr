import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DecisionRecord, MetricPoint } from "../types";
import { useChartColors } from "../ThemeContext";

interface Props {
  timeline: MetricPoint[];
  decisions: DecisionRecord[];
  task?: "classification" | "detection" | "multilabel";
}

/**
 * Loss & Metric Landscape — shows loss and primary metric over the entire
 * training run with vertical markers at each decision point.
 * Focuses on the trunk (baseline + selected branches).
 */
export function LossLandscape({ timeline, decisions, task }: Props) {
  const cc = useChartColors();

  const isDetection = useMemo(() => {
    if (task === "detection") return true;
    if (task === "classification") return false;
    return timeline.some((p) => (p.map_50 ?? 0) > 0) && !timeline.some((p) => p.accuracy > 0);
  }, [timeline, task]);

  const { trunkRows, markers } = useMemo(() => {
    if (timeline.length === 0) return { trunkRows: [], markers: [] as Array<{ x: string; label: string }> };

    // Build trunk: baseline + points from branches that were selected
    const selectedBranches = new Set<string>(["baseline"]);
    for (const d of decisions) {
      selectedBranches.add(d.selected_branch_label);
      // Also handle branch_id format like "iter_1:smugs"
      const parts = d.selected_branch_id.split(":");
      if (parts.length > 1) selectedBranches.add(parts.slice(1).join(":"));
    }

    const rows: Array<{ label: string; loss: number; accuracy_pct: number; idx: number }> = [];
    let idx = 0;
    for (const p of timeline) {
      if (selectedBranches.has(p.branch)) {
        const primaryMetric = isDetection ? (p.map_50 ?? 0) : p.accuracy;
        rows.push({
          label: `it${p.iteration}/e${p.epoch}`,
          loss: p.loss,
          accuracy_pct: Math.max(0, Math.min(100, primaryMetric * 100)),
          idx,
        });
        idx++;
      }
    }

    // Markers at decision points
    const ms: Array<{ x: string; label: string }> = [];
    let prevBranch = "";
    for (const r of rows) {
      const parts = r.label.split("/");
      if (prevBranch !== "" && r.label !== prevBranch) {
        // Check if this is a transition point
        const branchInTimeline = timeline.find((t) =>
          `it${t.iteration}/e${t.epoch}` === r.label && selectedBranches.has(t.branch) && t.branch !== "baseline");
        if (branchInTimeline && branchInTimeline.branch !== "baseline") {
          ms.push({ x: r.label, label: branchInTimeline.branch });
        }
      }
      prevBranch = r.label;
    }

    // Simpler approach: mark at iteration transitions
    const decisionLabels: Array<{ x: string; label: string }> = [];
    for (const d of decisions) {
      // Find the first trunk row with this iteration
      const iter = d.iteration;
      const found = rows.find((r) => r.label.startsWith(`it${iter}/`));
      if (found) {
        decisionLabels.push({ x: found.label, label: `↑ ${d.selected_branch_label}` });
      }
    }

    return { trunkRows: rows, markers: decisionLabels };
  }, [timeline, decisions, isDetection]);

  if (trunkRows.length === 0) {
    return <p className="muted">No data yet.</p>;
  }

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={trunkRows}>
          <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
          <XAxis dataKey="label" tick={{ fontSize: 10, fill: cc.text }} interval="preserveStartEnd" />
          <YAxis yAxisId="loss" orientation="left" tick={{ fontSize: 11, fill: cc.text }} label={{ value: "Loss", angle: -90, position: "insideLeft", fontSize: 11, fill: cc.text }} />
          <YAxis yAxisId="acc" orientation="right" unit="%" domain={[0, 100]} tick={{ fontSize: 11, fill: cc.text }} label={{ value: isDetection ? "mAP@0.5 %" : "Acc %", angle: 90, position: "insideRight", fontSize: 11, fill: cc.text }} />
          <Tooltip contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }} />
          <Legend />
          {markers.map((m, i) => (
            <ReferenceLine
              key={`decision-${i}`}
              yAxisId="loss"
              x={m.x}
              stroke={cc.referenceLine}
              strokeDasharray="4 4"
              strokeWidth={2}
              label={{ value: m.label, position: "top", fontSize: 9, fill: cc.refLabelFill }}
            />
          ))}
          <Line yAxisId="loss" dataKey="loss" stroke="#dc2626" dot={false} name="Loss" strokeWidth={2} />
          <Line yAxisId="acc" dataKey="accuracy_pct" stroke="#16a34a" dot={false} name={isDetection ? "mAP@0.5 (%)" : "Accuracy (%)"} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
