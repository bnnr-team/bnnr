import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PerClassRow } from "../types";
import { useChartColors } from "../ThemeContext";

const COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2",
  "#d97706", "#4f46e5", "#0f766e", "#be185d", "#65a30d",
];

interface Props {
  perClassTimeline: Record<string, PerClassRow[]>;
  task?: "classification" | "detection" | "multilabel";
  classNames?: string[];
}

export function PerClassMetrics({ perClassTimeline, task = "classification", classNames }: Props) {
  const cc = useChartColors();
  const isDetection = task === "detection";
  const yLabel = isDetection ? "AP@0.5 %" : task === "multilabel" ? "F1 %" : "Accuracy %";

  const classKeys = useMemo(
    () => Object.keys(perClassTimeline).sort((a, b) => Number(a) - Number(b)),
    [perClassTimeline],
  );
  const [selected, setSelected] = useState<string[]>(() => classKeys.slice(0, 5));

  const chartData = useMemo(() => {
    const epochs = new Map<string, Record<string, number | string>>();
    selected.forEach((cls) => {
      (perClassTimeline[cls] ?? []).forEach((r) => {
        const key = `${r.iteration}:${r.epoch}`;
        const row = epochs.get(key) ?? { label: `it${r.iteration}/e${r.epoch}` };
        row[`c${cls}`] = Math.min(100, r.accuracy * 100);
        epochs.set(key, row);
      });
    });
    return Array.from(epochs.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([, v]) => v);
  }, [selected, perClassTimeline]);

  /** Resolve a human-readable class label */
  function classLabel(cls: string): string {
    const idx = Number(cls);
    if (classNames && idx < classNames.length) return classNames[idx];
    return `class ${cls}`;
  }

  if (classKeys.length === 0) {
    return <p className="muted">No per-class metrics yet.</p>;
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
        <button
          className="btn btn-xs"
          onClick={() => setSelected([...classKeys].slice(0, 10))}
          title="Select the first 10 classes"
        >
          Select All
        </button>
        <button
          className="btn btn-xs"
          onClick={() => setSelected([])}
        >
          Deselect All
        </button>
        <span style={{ fontSize: 11, color: "var(--muted)", alignSelf: "center" }}>
          {selected.length}/{classKeys.length} shown (max 10)
        </span>
      </div>
      <div className="class-selector">
        {classKeys.map((cls) => (
          <label key={cls} className="check-item">
            <input
              type="checkbox"
              checked={selected.includes(cls)}
              onChange={(e) => {
                setSelected((prev) =>
                  e.target.checked
                    ? [...new Set([...prev, cls])].slice(0, 10)
                    : prev.filter((c) => c !== cls),
                );
              }}
            />
            {classLabel(cls)}
          </label>
        ))}
      </div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
            <XAxis dataKey="label" tick={{ fontSize: 11, fill: cc.text }} interval="preserveStartEnd" />
            <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 11, fill: cc.text }} label={{ value: yLabel, angle: -90, position: "insideLeft", style: { fontSize: 11, fill: cc.text } }} />
            <Tooltip contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }} />
            <Legend />
            {selected.map((cls, i) => (
              <Line
                key={cls}
                dataKey={`c${cls}`}
                name={classLabel(cls)}
                stroke={COLORS[i % COLORS.length]}
                dot={false}
                strokeWidth={1.5}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
