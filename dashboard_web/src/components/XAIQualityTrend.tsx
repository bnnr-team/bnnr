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
import type { XAIInsightEntry } from "../types";
import { useChartColors } from "../ThemeContext";

const OVERALL_COLOR = "#f0a069";
const CLASS_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2",
  "#d97706", "#4f46e5", "#0f766e", "#be185d", "#65a30d",
];

interface Props {
  xaiInsightsTimeline?: XAIInsightEntry[];
  classNames?: string[];
}

/**
 * XAI Quality Trend chart — shows how XAI quality score evolves over
 * training checkpoints, with per-class breakdown and overall average.
 *
 * Data is extracted from xai_insights_timeline diagnoses, which contain
 * per-class quality_score values at each checkpoint.
 *
 * Uses the same checkbox-toggle pattern as PerClassMetrics for class
 * selection.  "Overall" is always shown; individual classes are toggled.
 */
export function XAIQualityTrend({ xaiInsightsTimeline, classNames }: Props) {
  const cc = useChartColors();

  /* Build chart data from xai_insights_timeline diagnoses */
  const { chartData, classIds } = useMemo(() => {
    if (!xaiInsightsTimeline || xaiInsightsTimeline.length === 0)
      return { chartData: [], classIds: [] as string[] };

    const allClassIds = new Set<string>();
    const rows: Record<string, unknown>[] = [];

    for (const entry of xaiInsightsTimeline) {
      if (!entry.diagnoses) continue;

      const row: Record<string, unknown> = {
        label:
          entry.branch === "baseline"
            ? `Baseline e${entry.epoch}`
            : `I${entry.iteration} e${entry.epoch}`,
        branch: entry.branch,
        iteration: entry.iteration,
        epoch: entry.epoch,
      };

      let sum = 0;
      let count = 0;
      for (const [cls, diag] of Object.entries(entry.diagnoses)) {
        const q =
          typeof diag === "object" && diag !== null
            ? (diag as Record<string, unknown>).quality_score
            : undefined;
        if (typeof q === "number" && q > 0) {
          const name =
            classNames && Number(cls) < classNames.length
              ? classNames[Number(cls)]
              : `Class ${cls}`;
          row[name] = Math.round(q * 1000) / 10; // percent with 1 decimal
          allClassIds.add(name);
          sum += q;
          count++;
        }
      }
      if (count > 0) {
        row["Overall"] = Math.round((sum / count) * 1000) / 10;
      }

      if (count > 0) rows.push(row);
    }

    const ids = Array.from(allClassIds).sort();
    return { chartData: rows, classIds: ids };
  }, [xaiInsightsTimeline, classNames]);

  /* ---- Class selection state (default: first 5) ---- */
  const [selected, setSelected] = useState<string[]>(() =>
    classIds.slice(0, 5),
  );

  // Sync when classIds change (e.g. first data arrives)
  const [prevClassIds, setPrevClassIds] = useState(classIds);
  if (classIds !== prevClassIds && classIds.length > 0 && selected.length === 0) {
    setSelected(classIds.slice(0, 5));
    setPrevClassIds(classIds);
  }

  if (chartData.length < 2) {
    return (
      <p className="muted" style={{ fontSize: 11 }}>
        XAI quality trend will appear after at least 2 checkpoints with XAI
        data.
      </p>
    );
  }

  return (
    <div>
      {/* Class toggle controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
        <button
          className="btn btn-xs"
          onClick={() => setSelected([...classIds].slice(0, 10))}
          title="Select up to 10 classes"
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
          {selected.length}/{classIds.length} classes shown · Overall always visible
        </span>
      </div>
      <div className="class-selector">
        {classIds.map((cls) => (
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
            {cls}
          </label>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={cc.grid} strokeDasharray="3 3" />
          <XAxis
            dataKey="label"
            tick={{ fill: cc.text, fontSize: 10 }}
            stroke={cc.axisLine}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: cc.text, fontSize: 10 }}
            stroke={cc.axisLine}
            label={{
              value: "Quality %",
              angle: -90,
              position: "insideLeft",
              fill: cc.text,
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              background: cc.tooltipBg,
              border: `1px solid ${cc.tooltipBorder}`,
              borderRadius: 6,
              fontSize: 11,
            }}
            formatter={(value: number) => `${value.toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {/* Overall average line — always shown, thick, highlighted */}
          <Line
            type="monotone"
            dataKey="Overall"
            stroke={OVERALL_COLOR}
            strokeWidth={2.5}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          {/* Per-class lines — only selected classes */}
          {selected.map((cls) => {
            const i = classIds.indexOf(cls);
            return (
              <Line
                key={cls}
                type="monotone"
                dataKey={cls}
                stroke={CLASS_COLORS[i % CLASS_COLORS.length]}
                strokeWidth={1.2}
                dot={{ r: 2 }}
                strokeDasharray={i >= CLASS_COLORS.length ? "5 3" : undefined}
                connectNulls
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
