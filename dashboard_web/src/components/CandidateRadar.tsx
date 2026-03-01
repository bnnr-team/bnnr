import { useMemo, useState } from "react";
import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { DecisionRecord } from "../types";
import { useChartColors } from "../ThemeContext";

const COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2",
  "#d97706", "#4f46e5", "#0f766e",
];

interface Props {
  decisions: DecisionRecord[];
  task?: "classification" | "detection" | "multilabel";
}

export function CandidateRadar({ decisions, task = "classification" }: Props) {
  const cc = useChartColors();
  const iterations = decisions.map((d) => d.iteration);
  const [selectedIter, setSelectedIter] = useState<number>(
    iterations[iterations.length - 1] ?? 0,
  );

  const decision = decisions.find((d) => d.iteration === selectedIter);
  const candidates = useMemo(() => {
    if (!decision) return [];
    return Object.keys(decision.results);
  }, [decision]);

  const isDetection = task === "detection";

  const radarData = useMemo(() => {
    if (!decision) return [];

    // Choose axes based on task type
    const metricAxes: { key: string; label: string }[] = isDetection
      ? [
          { key: "map_50", label: "mAP@0.5" },
          { key: "map_50_95", label: "mAP@[.5:.95]" },
        ]
      : [
          { key: "accuracy", label: "Accuracy" },
          { key: "f1_macro", label: "F1 Macro" },
        ];

    // Compute inverted loss as a normalized axis
    const allLoss = Object.values(decision.results).map((r) => r.loss ?? 0);
    const maxLoss = Math.max(...allLoss, 0.001);

    return metricAxes
      .map(({ key, label }) => {
        const row: Record<string, string | number> = { metric: label };
        candidates.forEach((c) => {
          const val = (decision.results[c] as Record<string, number | undefined>)?.[key];
          row[c] = val !== undefined ? Math.round(val * 10000) / 100 : 0; // as %
        });
        return row;
      })
      .concat([
        (() => {
          const row: Record<string, string | number> = { metric: "1-loss (norm)" };
          candidates.forEach((c) => {
            const loss = decision.results[c]?.loss ?? maxLoss;
            row[c] = Math.round((1 - loss / maxLoss) * 10000) / 100;
          });
          return row;
        })(),
      ]);
  }, [decision, candidates, isDetection]);

  if (decisions.length === 0) {
    return <p className="muted">No branch decisions recorded yet.</p>;
  }

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <label>
          Iteration:{" "}
          <select
            value={selectedIter}
            onChange={(e) => setSelectedIter(Number(e.target.value))}
          >
            {iterations.map((it) => (
              <option key={it} value={it}>
                Iteration {it}
              </option>
            ))}
          </select>
        </label>
        {decision && (
          <span style={{ marginLeft: 12, color: "var(--green)", fontWeight: 600 }}>
            Winner: {decision.selected_branch_label}
          </span>
        )}
      </div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke={cc.grid} />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: cc.text }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: cc.text }} />
            <Tooltip contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }} />
            <Legend
              wrapperStyle={{ maxHeight: 56, overflowY: 'auto', overflowX: 'hidden', fontSize: 11, lineHeight: '18px' }}
              iconSize={10}
            />
            {candidates.map((c, i) => (
              <Radar
                key={c}
                name={c}
                dataKey={c}
                stroke={COLORS[i % COLORS.length]}
                fill={COLORS[i % COLORS.length]}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            ))}
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
